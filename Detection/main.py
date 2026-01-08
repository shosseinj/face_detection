import cv2
import numpy as np
import argparse
import onnxruntime as ort
from qdrant_client import QdrantClient
from insightface.utils import face_align
from insightface.model_zoo.retinaface import RetinaFace
import av
import matplotlib.pyplot as plt
import os
import cv2
import time
from ultralytics import YOLO
import torchvision, torch

SAVE_DIR_FACE = "detected_faces"
os.makedirs(SAVE_DIR_FACE, exist_ok=True)


SAVE_DIR_PERSON = "detected_PERSON"
os.makedirs(SAVE_DIR_PERSON, exist_ok=True)
torch.cuda.empty_cache()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    parser = argparse.ArgumentParser("Real-time Face Recognition from Webcam")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--det_threshold", type=float, default=0.01, help="Detection threshold")
    parser.add_argument("--collection", type=str, default="n3", help="Qdrant collection name")
    parser.add_argument("--webCam", type=str2bool, default=False, help="Camera device index")
    parser.add_argument("--save_face", type=str2bool, default=False, help="Camera device index")
    parser.add_argument("--save_person", type=str2bool, default=False, help="Camera device index")
    return parser.parse_args()



def load_models(args):
    """Load RetinaFace and ArcFace ONNX models"""
    # Set up ONNX Runtime
    providers = ['CUDAExecutionProvider'] 
    
    # Load RetinaFace detector
    print("Loading RetinaFace detector...")
    # Try to find the model
    import os
    model_paths = [
        "./weights/buffalo_l/det_10g.onnx",
        "./weights/buffalo_l/det_500m.onnx",
        "det_10g.onnx"
    ]
    
    det_model_path = None
    for path in model_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            det_model_path = expanded_path
            break
    
    if det_model_path is None:
        raise FileNotFoundError("Could not find RetinaFace model file")
    
    print(f"Using RetinaFace model: {det_model_path}")
    det_session = ort.InferenceSession(det_model_path, providers=providers)
    
    # Load ArcFace recognizer
    print("Loading ArcFace recognizer...")
    rec_model_paths = [
        "./weights/buffalo_l/w600k_r50.onnx",
        "./weights/buffalo_l/glintr100.onnx",
        "w600k_r50.onnx"
    ]
    
    rec_model_path = None
    for path in rec_model_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            rec_model_path = expanded_path
            break
    
    if rec_model_path is None:
        raise FileNotFoundError("Could not find ArcFace model file")
    
    print(f"Using ArcFace model: {rec_model_path}")
    rec_session = ort.InferenceSession(rec_model_path, providers=providers)
    
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    
    return det_session, rec_session, client

def detect_faces_retinaface(det_session, frame, det_thresh=0.5):
    """Detect faces using RetinaFace ONNX model"""
    # Initialize RetinaFace with the session
    detector = RetinaFace(model_file=None, session=det_session)
    
    # Detect faces - specify input_size
    try:
        # Try with input_size parameter
        bboxes, landmarks = detector.detect(frame, input_size=(320, 320))
    except TypeError:
        try:
            # Try without input_size
            bboxes, landmarks = detector.detect(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    faces = []
    if bboxes is not None and len(bboxes) > 0:
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            
            # Handle different bbox formats
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                score = bbox[4] if len(bbox) > 4 else 0.5
                
                # Apply threshold
                if score < det_thresh:
                    continue
                
                landmark = landmarks[i] if landmarks is not None and i < len(landmarks) else None
                
                faces.append({
                    'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                    'landmark': landmark,
                    'score': score
                })
    
    return faces

def extract_embedding_arcface(rec_session, face_img):
    """Extract face embedding using ArcFace ONNX model"""
    # Preprocess face image (should be 112x112 BGR)
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    
    # Resize to 112x112 if needed
    if face_img.shape[:2] != (112, 112):
        face_img = cv2.resize(face_img, (112, 112))
    
    # Ensure it's BGR (ArcFace models typically expect BGR)
    if face_img.shape[2] == 3:
        # Already BGR (from cv2.imread)
        pass
    
    # Normalize for ArcFace
    face_img = face_img.astype(np.float32)
    face_img = (face_img - 127.5) / 128.0
    
    # Transpose to NCHW format
    face_img = np.transpose(face_img, (2, 0, 1))  # HWC to CHW
    face_img = np.expand_dims(face_img, axis=0)   # Add batch dimension
    
    # Run inference
    input_name = rec_session.get_inputs()[0].name
    output_name = rec_session.get_outputs()[0].name
    
    embedding = rec_session.run([output_name], {input_name: face_img})[0]
    
    # Normalize embedding (L2 normalization)
    embedding = embedding[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def recognize_face(embedding, client, args):
    """Recognize a single face embedding"""
    search_result = client.query_points(
        collection_name=args.collection, 
        query=embedding.tolist(), 
        limit=1
    )
    
    person = "Unknown"
    score = 0.0
    
    if search_result and hasattr(search_result, 'points') and search_result.points:
        best_match = search_result.points[0]
        
        if hasattr(best_match, 'payload') and best_match.payload:
            person = best_match.payload.get("person", "Unknown")
        
        if hasattr(best_match, 'score'):
            score = best_match.score

    if score < 0.3:
        person = "Unknown"
        # score = 0.0
    return person, score

def draw_face_info(frame, bbox, person, score, color):
    """Draw face bounding box and name on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    if person == "Unknown":
        label = "Unknown"
    else:
        label = f"{person} ({score:.2f})"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    text_bg_top = max(y1 - text_height - 10, 0)
    text_bg_bottom = y1
    text_bg_left = x1
    text_bg_right = x1 + text_width
    
    cv2.rectangle(frame,
                 (text_bg_left, text_bg_top),
                 (text_bg_right, text_bg_bottom),
                 color, -1)
    
    cv2.putText(frame, label,
               (x1, y1 - 5),
               font, font_scale, (255, 0, 255), thickness)
    
    return frame


def open_capture(webCam):
    if webCam:
        cap = cv2.VideoCapture(int(0))
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
    
    else:
        RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
        cap = av.open(RTSP_URL, options={"rtsp_transport": "tcp", "flags": "low_delay", "fflags": "nobuffer"}).decode(video=0)

    return cap

import torch
def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU REQUIRED")
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return device



import cv2

def resize_to_fit(frame, max_width, max_height):
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return frame

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)



def frame_generator(container, webCamUsage=True, max_width=3000, max_height=3000): # 1280 *720
    """
    Unified frame generator with resizing.
    - webCamUsage=True: container is cv2.VideoCapture
    - webCamUsage=False: container is PyAV container (RTSP)
    """

    if webCamUsage:
        while True:
            ret, frame = container.read()
            if not ret:
                break

            frame = resize_to_fit(frame, max_width, max_height)
            yield frame

    else:
        for packet in container:
            frame = packet.to_ndarray(format="bgr24")

            frame = resize_to_fit(frame, max_width, max_height)
            yield frame



def detect_faces_retinaface1(det_session, rec_session, client, frame, det_thresh=0.5, args=None):
    """
    Detect faces using RetinaFace ONNX model, align, resize, 
    extract embeddings in batch, and recognize faces.
    Returns a list of (bbox, person, score) tuples.
    """
    # Initialize RetinaFace with the session
    detector = RetinaFace(model_file=None, session=det_session)
    
    # Detect faces
    try:
        bboxes, landmarks = detector.detect(frame, input_size=(640, 640))
    except TypeError:
        try:
            bboxes, landmarks = detector.detect(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    if bboxes is None or len(bboxes) == 0:
        return []

    batch_faces = []
    batch_bboxes = []

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox[:4]
        score = bbox[4] if len(bbox) > 4 else 0.5
        if score < det_thresh:
            continue

        landmark = landmarks[i] if landmarks is not None and i < len(landmarks) else None

        # Align or crop face
        aligned_face = None
        if landmark is not None:
            try:
                aligned_face = face_align.norm_crop(frame, landmark)
            except:
                aligned_face = None

        if aligned_face is None or aligned_face.size == 0:
            margin = 0.2
            h, w = frame.shape[:2]
            x1m = max(0, int(x1 - (x2 - x1) * margin))
            y1m = max(0, int(y1 - (y2 - y1) * margin))
            x2m = min(w, int(x2 + (x2 - x1) * margin))
            y2m = min(h, int(y2 + (y2 - y1) * margin))
            aligned_face = frame[y1m:y2m, x1m:x2m]

        if aligned_face.size == 0:
            continue

        aligned_face = cv2.resize(aligned_face, (112, 112))
        batch_faces.append(aligned_face)
        batch_bboxes.append([x1, y1, x2, y2])

    if not batch_faces:
        return []

    # 1️⃣ Batch preprocessing
    batch_array = np.stack(batch_faces, axis=0).astype(np.float32)
    batch_array = (batch_array - 127.5) / 128.0
    batch_array = np.transpose(batch_array, (0, 3, 1, 2))

    input_name = rec_session.get_inputs()[0].name
    output_name = rec_session.get_outputs()[0].name

    # 2️⃣ Run embeddings in batch
    embeddings = rec_session.run([output_name], {input_name: batch_array})[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 3️⃣ Recognize all faces in batch
    recognized_faces = []
    for bbox, emb in zip(batch_bboxes, embeddings):
        person, score = recognize_face(emb, client, args)
        recognized_faces.append((np.array(bbox, dtype=np.float32), person, score))

    return recognized_faces



def detect_faces_retinaface_gpu2(
    det_session,
    rec_session,
    client,
    frame,
    det_thresh=0.01,
    args=None
):
    """
    RetinaFace detection + batch ArcFace embedding + Qdrant recognition.
    Returns: [(bbox, person, score), ...]
    """


    H, W = frame.shape[:2]

    # ---------------------------
    # 1️⃣ YOLO PERSON DETECTION
    # ---------------------------
    yolo_result = yolo(
        frame,
        imgsz=960,
        conf=0.25,
        iou=0.5,
        classes=[0],   # person only
        device=0,
        verbose=False
    )[0]

    if yolo_result.boxes is None:
        return []
    

    detector = RetinaFace(model_file=None, session=det_session)


    bboxes, landmarks = detector.detect(frame, input_size=(960, 960))
   
    if bboxes is None or len(bboxes) == 0:
        return []

    batch_faces = []
    batch_bboxes = []

    h, w = frame.shape[:2]
    margin = 0.2

    # 2️⃣ Align / crop faces (CPU, unavoidable with current norm_crop)
    for i, bbox in enumerate(bboxes):
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox[:4]
        score = bbox[4] if len(bbox) > 4 else 0.5
        if score < det_thresh:
            continue

        landmark = landmarks[i] if landmarks is not None else None

        aligned = None
        if landmark is not None:
            try:
                aligned = face_align.norm_crop(frame, landmark)
            except:
                aligned = None

        if aligned is None or aligned.size == 0:
            x1m = max(0, int(x1 - (x2 - x1) * margin))
            y1m = max(0, int(y1 - (y2 - y1) * margin))
            x2m = min(w, int(x2 + (x2 - x1) * margin))
            y2m = min(h, int(y2 + (y2 - y1) * margin))
            aligned = frame[y1m:y2m, x1m:x2m]

        if aligned.size == 0:
            continue

        aligned = cv2.resize(aligned, (112, 112))
        batch_faces.append(aligned)
        batch_bboxes.append([x1, y1, x2, y2])

        if args.save_detected:
            face_name = f"face_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, face_name), aligned)

    if not batch_faces:
        return []

    # 3️⃣ Batch ArcFace preprocessing
    batch_array = np.stack(batch_faces).astype(np.float32)
    batch_array = (batch_array - 127.5) / 128.0
    batch_array = batch_array.transpose(0, 3, 1, 2)

    input_name = rec_session.get_inputs()[0].name
    output_name = rec_session.get_outputs()[0].name

    # 4️⃣ Batch ArcFace inference


    embeddings = []
    for i in range(batch_array.shape[0]):
        emb = rec_session.run(
            [output_name],
            {input_name: batch_array[i:i+1]}
        )[0]
        embeddings.append(emb[0])

    embeddings = np.stack(embeddings, axis=0)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)



    # 5️⃣ Qdrant recognition (UNAVOIDABLE loop)
    recognized_faces = []
    for bbox, emb in zip(batch_bboxes, embeddings):
        person, score = recognize_face(emb, client, args)
        recognized_faces.append(
            (np.array(bbox, dtype=np.float32), person, score)
        )

    return recognized_faces





import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

def detect_faces_retinaface_gpu3(
    rec_session,
    client,
    frame,
    det_thresh=0.01,
    args=None,
    yolo=None, 
    detector=None
):
    """
    YOLOv8-person → RetinaFace → ArcFace(batch) → Qdrant
    Returns: [(bbox, person, score)]
    """

    H, W = frame.shape[:2]

    low_res=(640, 384)
    small_frame = cv2.resize(frame, low_res)
    scale_x = W / low_res[0]
    scale_y = H / low_res[1]



    # ---------------------------
    # 1️⃣ YOLO PERSON DETECTION
    # ---------------------------
    yolo_result = yolo(
        small_frame,
        imgsz=low_res,
        conf=0.25,
        iou=0.5,
        classes=[0],   # person only
        device=0,
        verbose=False
    )[0]

    if yolo_result.boxes is None:
        return []


    batch_faces = []
    batch_bboxes = []

    # ---------------------------
    # 2️⃣ LOOP OVER PERSONS
    # ---------------------------
    for box in yolo_result.boxes.xyxy.cpu().numpy():
        px1, py1, px2, py2 = map(int, [
        box[0] * scale_x,
        box[1] * scale_y,
        box[2] * scale_x,
        box[3] * scale_y
    ])

    
        # expand person bbox
        margin = 0.15
        pw, ph = px2 - px1, py2 - py1
        px1 = max(0, int(px1 - pw * margin))
        py1 = max(0, int(py1 - ph * margin))
        px2 = min(W, int(px2 + pw * margin))
        py2 = min(H, int(py2 + ph * margin))

        pw = px2 - px1
        ph = py2 - py1

        # 🔥 GUARD 1: too small person
        if (pw * ph) < 0.01 * (W * H):
            continue

        # 🔥 HEAD REGION ONLY
        head_y2 = py1 + int(0.6 * ph)
        person_crop = frame[py1:head_y2, px1:px2]

        if person_crop.size == 0:
            continue



        
        if args.save_person:
            person_name = f"person_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR_PERSON, person_name), person_crop)
        if person_crop.size == 0:
            continue

        # ---------------------------
        # 3️⃣ RETINAFACE ON PERSON ROI
        # ---------------------------
        try:
            bboxes, landmarks = detector.detect(
                person_crop, input_size=(640, 348)
            )
        except:
            continue

        if bboxes is None:
            continue

        for i, bbox in enumerate(bboxes):
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox[:4]
            score = bbox[4] if len(bbox) > 4 else 1.0
            if score < det_thresh:
                continue

            lm = landmarks[i] if landmarks is not None else None

            # ---------------------------
            # 4️⃣ FACE ALIGN
            # ---------------------------
            aligned = None
            if lm is not None:
                try:
                    aligned = face_align.norm_crop(person_crop, lm)
                except:
                    aligned = None

            if aligned is None or aligned.size == 0:
                aligned = person_crop[int(y1):int(y2), int(x1):int(x2)]

            if aligned.size == 0:
                continue

            aligned = cv2.resize(aligned, (112, 112))
            batch_faces.append(aligned)

            # global bbox
            gx1 = px1 + x1
            gy1 = py1 + y1
            gx2 = px1 + x2
            gy2 = py1 + y2
            batch_bboxes.append([gx1, gy1, gx2, gy2])

    if not batch_faces:
        return []

    # ---------------------------
    # 5️⃣ ARCFACE BATCH
    # ---------------------------
    batch = np.stack(batch_faces).astype(np.float32)
    batch = (batch - 127.5) / 128.0
    batch = np.transpose(batch, (0, 3, 1, 2))

    input_name = rec_session.get_inputs()[0].name
    output_name = rec_session.get_outputs()[0].name

    embeddings = rec_session.run(
        [output_name], {input_name: batch}
    )[0]

    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # ---------------------------
    # 6️⃣ QDRANT SEARCH (still loop, unavoidable)
    # ---------------------------
    results = []
    for bbox, emb in zip(batch_bboxes, embeddings):
        person, score = recognize_face(emb, client, args)
        results.append((np.array(bbox, dtype=np.float32), person, score))

    return results




def detect_faces_retinaface_gpu6(
    frame,
    yolo,
    detector,
    args=None
):
    """
    Use existing face alignment instead of ROI align
    """
    import torch
    
    H, W = frame.shape[:2]
    
    # Person detection
    # person_res = (640, 384)
    person_res = (640, 384)
    small_frame = cv2.resize(frame, person_res)
    
    yolo_result = yolo(
        small_frame,
        imgsz=person_res,
        conf=0.25,
        iou=0.5,
        classes=[0],
        device='cuda:0',
        half=True,
        verbose=False
    )[0]
    
    if yolo_result.boxes is None:
        return []
    
    scale_x = W / person_res[0]
    scale_y = H / person_res[1]
    
    # Keep boxes on GPU for tensor operations
    person_boxes_gpu = yolo_result.boxes.xyxy
    
    # Scale on GPU
    scale_tensor = torch.tensor([scale_x, scale_y, scale_x, scale_y], 
                               device=person_boxes_gpu.device)
    scaled_boxes = person_boxes_gpu * scale_tensor
    
    results = []
    
    for i in range(len(scaled_boxes)):
        box = scaled_boxes[i]
        
        # Convert to ints for cropping
        px1, py1, px2, py2 = map(int, [
            box[0].item(), box[1].item(), box[2].item(), box[3].item()
        ])
        
        # Expand person bbox
        margin = 0.1
        pw, ph = px2 - px1, py2 - py1
        
        px1 = max(0, int(px1 - pw * margin))
        py1 = max(0, int(py1 - ph * margin))
        px2 = min(W, int(px2 + pw * margin))
        py2 = min(H, int(py2 + ph * margin))
        
        pw, ph = px2 - px1, py2 - py1
        
        # Skip too small persons
        if (pw * ph) < 0.005 * (W * H):
            continue
        
        # Extract head region for face detection
        head_y2 = py1 + int(0.6 * ph)
        person_crop = frame[py1:head_y2, px1:px2]
        
        if person_crop.size == 0:
            continue
        
        # Save person crop if needed
        if args and args.save_person:
            person_name = f"person_{int(time.time() * 1000)}_{i}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR_PERSON, person_name), person_crop)
        
        # Face detection with landmarks
        try:
            bboxes, landmarks = detector.detect(
                person_crop, 
                threshold=0.01,
                input_size=(640, 640)
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            continue

        if bboxes is None or len(bboxes) == 0:
            continue
        
        # Process each face
        for j, bbox in enumerate(bboxes):
            score = bbox[4]
            if score < 0.01:
                continue
            
            # Face coordinates in person_crop
            fx1, fy1, fx2, fy2 = map(int, bbox[:4])
            
            # Convert to absolute coordinates
            abs_fx1 = px1 + fx1
            abs_fy1 = py1 + fy1
            abs_fx2 = px1 + fx2
            abs_fy2 = py1 + fy2
            
            # Extract face region from original frame
            face_region = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
            
            if face_region.size == 0:
                continue
            
            # ---------------    for box in yolo_result.boxes.xyxy.cpu().numpy():
        px1, py1, px2, py2 = resizing_box(box, 1,1)
        person_boxes.append(((px1, py1, px2, py2), 'person', 0.9))

        px1_org, py1_org, px2_org, py2_org = resizing_box(box, W / PERSON_RES[0], H / PERSON_RES[1])
        person_boxes_org.append(((px1_org, py1_org, px2_org, py2_org), 'person', 1.0))
# ###################

        face_crop = frame[py1_org: py2_org, px1_org:px2_org]

        if face_crop is None or face_crop.size == 0:
            continue

        if args.save_person :
            person_name = f"person_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR_PERSON, person_name), face_crop)




        # -------------------------
        # Face detection
        # -------------------------
        try:
            bboxes, landmarks = face_detector.detect(
                face_crop,
                threshold=0.01,
                input_size=(face_crop.shape[1], face_crop.shape[0])
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            continue

        if bboxes is None:
            continue


        # -------------------------
        # Scaling factors
        # -------------------------
        crop_h, crop_w = face_crop.shape[:2]



        # -------------------------
        # Process faces
        # -------------------------
        for j, bbox in enumerate(bboxes):
            score = float(bbox[4])
            if score < 0.01:
                continue

            fx1, fy1, fx2, fy2 = map(int, bbox[:4])

            # Clamp to face_crop bounds
            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(crop_w, fx2)
            fy2 = min(crop_h, fy2)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # Absolute coords in original frame
            abs_fx1 = px1_org + fx1
            abs_fy1 = py1_org + fy1
            abs_fx2 = px1_org + fx2
            abs_fy2 = py1_org + fy2


            # -------------------------
            # Face alignment
            # -------------------------
            aligned_face = None

            if face_align is not None and landmarks is not None and j < len(landmarks):
                lm = landmarks[j].copy()

                try:
                    # alignment MUST be on face_crop
                    aligned_face = face_align.norm_crop(face_crop, lm)
                    aligned_face = cv2.resize(aligned_face, (224, 224))
                except Exception:
                    pass

            # Fallback: direct crop
            if aligned_face is None:
                raw_face = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
                # if raw_face is not None and raw_face.size > 0:
                #     aligned_face = cv2.resize(raw_face, (224, 224))

            # if aligned_face is None:
            #     continue
            

            if args.save_face :
                face_name = f"face_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR_FACE, face_name), raw_face)
            # FACE ALIGNMENT using landmarks
            # ---------------------------
            aligned_face = None
            
            if landmarks is not None and j < len(landmarks):
                lm = landmarks[j]
                
                try:
                    # Adjust landmarks to person_crop coordinates
                    # Landmarks from detector are relative to person_crop
                    # We need to use them with person_crop for alignment
                    aligned_face = face_align.norm_crop(person_crop, lm)
                    
                    # Resize to recognition size
                    aligned_face = cv2.resize(aligned_face, (224, 224))
                    
                except Exception as e:
                    print(f"Face alignment error: {e}")
                    # Fallback: use simple crop
                    aligned_face = cv2.resize(face_region, (224, 224))
            else:
                # No landmarks, use simple crop
                aligned_face = cv2.resize(face_region, (224, 224))
            
            # Store results
            results.append((
                (abs_fx1, abs_fy1, abs_fx2, abs_fy2),
                aligned_face,
                float(score)
            ))
    
    return results
























def resizing_box(box, scale_x,scale_y):
    px1, py1, px2, py2 = map(int, [
        box[0] * scale_x,
        box[1] * scale_y,
        box[2] * scale_x,
        box[3] * scale_y
    ])
    return px1, py1, px2, py2

# ///////////////////////////////
def detect_persons(frame, yolo, face_detector,rec_session,client, args=None):
  
    H, W = frame.shape[:2]
    
    # Ultra-low resolutions that are multiples of 32
    if W >= 3840:  # 4K - use 1.5% of pixels
        # (384, 224) = 0.086MP vs 8.3MP original (1%)
        # PERSON_RES = (384, 224)   # Both divisible by 32: 384/32=12, 224/32=7
        PERSON_RES = (640, 384)   
    elif W >= 1920:  # Full HD - use 3% of pixels
        # PERSON_RES = (640, 384)   # Same as above
        PERSON_RES = (640, 384)   
    elif W >= 1280:  # HD
        # PERSON_RES = (320, 192)   # 320/32=10, 192/32=6
        PERSON_RES = (640, 384)   
    else:  # Lower resolutions
        PERSON_RES = (640, 384)   # 256/32=8, 160/32=5

    PERSON_RES = (PERSON_RES[0] // 32 * 32, PERSON_RES[1] // 32 * 32)
    
    small_frame = cv2.resize(frame, PERSON_RES)
    
 
    yolo_result = yolo(
        small_frame,
        imgsz=PERSON_RES,
        conf=0.25,
        iou=0.5,
        classes=[0],  # person only
        device='cuda:0',
        half=True,
        verbose=False,
        # max_det=50  # Limit detections for speed
    )[0]
    
    if yolo_result.boxes is None:
        return []
    
    show_lowResolution = True
    if show_lowResolution:
        scale_x = 1
        scale_y = 1
        out_frame = small_frame
    else:
        scale_x = W / PERSON_RES[0]
        scale_y = H / PERSON_RES[1]
        out_frame = frame
    
    person_boxes = []
    person_boxes_org = []
    margin = 0.01

    

    for box in yolo_result.boxes.xyxy.cpu().numpy():
        px1, py1, px2, py2 = resizing_box(box, 1,1)
        person_boxes.append(((px1, py1, px2, py2), 'person', 0.9))

        px1_org, py1_org, px2_org, py2_org = resizing_box(box, W / PERSON_RES[0], H / PERSON_RES[1])
# ###################

        face_crop = frame[py1_org: py2_org, px1_org:px2_org]

        if face_crop is None or face_crop.size == 0:
            continue

        if args.save_person :
            person_name = f"person_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR_PERSON, person_name), face_crop)




        # -------------------------
        # Face detection
        # -------------------------
        try:
            retina_input = cv2.resize(face_crop, (640,640))
            bboxes, landmarks = face_detector.detect(
                retina_input,
                input_size=(640,640)
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            continue

        if bboxes is None:
            continue


        # -------------------------
        # Scaling factors
        # -------------------------
        crop_h, crop_w = face_crop.shape[:2]



        # -------------------------
        # Process faces
        # -------------------------
        for j, bbox in enumerate(bboxes):
            score = float(bbox[4])
            if score < 0.01:
                continue

            fx1, fy1, fx2, fy2 = map(int, bbox[:4])

            # Clamp to face_crop bounds
            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(crop_w, fx2)
            fy2 = min(crop_h, fy2)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # Absolute coords in original frame
            abs_fx1 = px1_org + fx1
            abs_fy1 = py1_org + fy1
            abs_fx2 = px1_org + fx2
            abs_fy2 = py1_org + fy2

            # -------------------------
            # Raw face crop (NO resize)
            # -------------------------
            raw_face = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]

            if raw_face is None or raw_face.size == 0:
                continue




            # ---------------------------
            # FACE ALIGNMENT (CORRECT)
            # ---------------------------
            aligned = None

            if landmarks is not None and j < len(landmarks):
                lm = landmarks[j].copy()  # landmarks are in face_crop coords

                try:
                    # align MUST be on face_crop, not raw_face
                    aligned = face_align.norm_crop(face_crop, lm)
                except Exception:
                    aligned = None

            # ---------------------------
            # Fallback: use raw face (no alignment)
            # ---------------------------
            if aligned is None or aligned.size == 0:
                aligned = raw_face

            if aligned is None or aligned.size == 0:
                continue

            # optional resize ONLY for recognition
            aligned = cv2.resize(aligned, (112, 112))



            # -------------------------
            # Save face as-is
            # -------------------------
            if args.save_face:
                face_name = f"face_{int(time.time() * 100000)}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR_FACE, face_name), aligned)

            



       

            aligned_norm = (aligned.astype(np.float32) - 127.5) / 128.0
            aligned_norm = np.transpose(aligned_norm, (2, 0, 1))
            aligned_norm = np.expand_dims(aligned_norm, axis=0)  # (1, 3, 112, 112)


    

            # input_name = rec_session.get_inputs()[0].name
            # output_name = rec_session.get_outputs()[0].name

            # embeddings = rec_session.run([output_name], {input_name: aligned_norm})[0]

            # person, score = recognize_face(embeddings[0], client, args)
            person = 'person'
            score=0.9
            person_boxes_org.append(((abs_fx1, abs_fy1 , abs_fx2, abs_fy2), person, score))




    
    return out_frame , person_boxes,person_boxes_org


def detect_faces_in_person(person_boxes, frame, detector, face_align=None, args=None):
    """
    Step 2: Face detection within person regions
    Returns: list of (face_bbox, aligned_face_image, score)
    """
    if not person_boxes:
        return []
    
    H, W = frame.shape[:2]
    results = []
    
    for px1, py1, px2, py2 in person_boxes:
        pw, ph = px2 - px1, py2 - py1
        
        # Skip too small persons
        if (pw * ph) < 0.003 * (W * H):
            continue
        
        # Extract head region (top 60% of person)
        head_y2 = py1 + int(0.6 * ph)
        head_region = frame[py1:head_y2, px1:px2]
        
        if head_region.size == 0:
            continue
        

        
        # Resize for face detection
        hh, hw = head_region.shape[:2]
        if hh > 0 and hw > 0:
            # Target size for face detection
            target_size = min(640, max(hh, hw))
            
            if hh > hw:
                new_height = target_size
                new_width = int(hw * (target_size / hh))
            else:
                new_width = target_size
                new_height = int(hh * (target_size / hw))
            
            face_detect_img = cv2.resize(head_region, (new_width, new_height))
            scale_w = new_width / hw
            scale_h = new_height / hh
        else:
            continue
        
        # Face detection
        try:
            bboxes, landmarks = detector.detect(
                face_detect_img, 
                threshold=0.01,
                input_size=(640, 640)
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            continue
        
        if bboxes is None:
            continue
        
        # Process each face
        for j, bbox in enumerate(bboxes):
            score = bbox[4]
            if score < 0.01:
                continue
            
            # Face coordinates in face_detect_img
            fx1, fy1, fx2, fy2 = map(int, bbox[:4])
            
            # Scale back to head_region coordinates
            fx1 = int(fx1 / scale_w)
            fy1 = int(fy1 / scale_h)
            fx2 = int(fx2 / scale_w)
            fy2 = int(fy2 / scale_h)
            
            # Convert to absolute coordinates
            abs_fx1 = px1 + fx1
            abs_fy1 = py1 + fy1
            abs_fx2 = px1 + fx2
            abs_fy2 = py1 + fy2
            
            # Face alignment
            aligned_face = None
            
            if face_align is not None and landmarks is not None and j < len(landmarks):
                lm = landmarks[j]
                lm_scaled = lm / np.array([scale_w, scale_h])
                
                try:
                    aligned_face = face_align.norm_crop(head_region, lm_scaled)
                    aligned_face = cv2.resize(aligned_face, (224, 224))
                except:
                    # Fallback: crop directly
                    face_crop = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
                    if face_crop.size > 0:
                        aligned_face = cv2.resize(face_crop, (224, 224))
            else:
                # No alignment, crop directly
                face_crop = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
                if face_crop.size > 0:
                    aligned_face = cv2.resize(face_crop, (224, 224))
            
            if aligned_face is not None:
                results.append((
                    (abs_fx1, abs_fy1, abs_fx2, abs_fy2),
                    aligned_face,
                    float(score)
                ))
    
    return results


def recognize_faces(faces, rec_session, client, threshold=0.5):
    """
    Step 3: Face recognition (ArcFace + Qdrant)
    Returns: list of (face_bbox, person_info, score)
    """
    if not faces:
        return []
    
    recognized_results = []
    
    # Batch processing for efficiency
    face_batch = []
    bbox_batch = []
    
    for bbox, face_img, det_score in faces:
        # Prepare face for recognition
        face_tensor = preprocess_face(face_img)  # Your preprocessing
        face_batch.append(face_tensor)
        bbox_batch.append((bbox, det_score))
    
    if not face_batch:
        return []
    
    # Batch inference
    try:
        # Convert batch to numpy array for ONNX
        face_array = np.array(face_batch, dtype=np.float32)
        
        # ArcFace inference
        embeddings = rec_session.run(None, {'input': face_array})[0]
        
        # Search in Qdrant for each embedding
        for i, embedding in enumerate(embeddings):
            bbox, det_score = bbox_batch[i]
            
            # Search in Qdrant
            search_result = client.search(
                collection_name="faces",
                query_vector=embedding.tolist(),
                limit=1
            )
            
            if search_result and search_result[0].score > threshold:
                person_info = search_result[0].payload.get("name", "Unknown")
                recognition_score = search_result[0].score
            else:
                person_info = "Unknown"
                recognition_score = 0.0
            
            recognized_results.append((bbox, person_info, recognition_score))
    
    except Exception as e:
        print(f"Recognition error: {e}")
        # Fallback: just return detection results
        for bbox, det_score in bbox_batch:
            recognized_results.append((bbox, "Unknown", det_score))
    
    return recognized_results


def detect_faces_retinaface_gpu7(frame, yolo, detector, rec_session,client,args=None):
    """
    Complete pipeline using separated functions
    """
    # Step 1: Person detection
    show_type= 'org'
    small_frame , small_person_boxes,person_boxes_org = detect_persons(frame, yolo, detector,rec_session,client, args)
    if show_type=='person':
        output_frame = small_frame
        results = small_person_boxes
    if show_type=='org':
        output_frame = frame
        results = person_boxes_org





    
    return output_frame , results

def main():
    args = parse_args()
    
    print("Loading models...")
    det_session, rec_session, client = load_models(args)

    container = open_capture(args.webCam)

    print("Starting real-time face recognition. Press 'q' to quit.")
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 255, 255), (0, 165, 255)
    ]
    
    
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    yolo = YOLO("yolov8n.pt") 

    import onnxruntime

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider'
    ]

    MODEL_PATH = "/home/shosseinj/Music/retinaface-pytorch/models/retinaface.onnx"
    det_session = onnxruntime.InferenceSession(MODEL_PATH, providers=providers)


    detector = RetinaFace(model_file=None, session=det_session)

    for frame in frame_generator(container, args.webCam):
        # recognized_faces = detect_faces_retinaface_gpu4( rec_session, client, frame, args.threshold, args, yolo, detector)

        frame , recognized_faces = detect_faces_retinaface_gpu7(  frame,  yolo, detector,rec_session,client,args)

        for bbox, person, score in recognized_faces:
            color_idx = hash(person) % len(colors)
            color = colors[color_idx]
            frame = draw_face_info(frame, bbox, person, score, color)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print("Face recognition stopped.")

if __name__ == "__main__":
    main()