import cv2
import numpy as np
import argparse
import onnxruntime as ort
from qdrant_client import QdrantClient
from insightface.utils import face_align
from insightface.model_zoo.retinaface import RetinaFace
import av

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
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--collection", type=str, default="n3", help="Qdrant collection name")
    parser.add_argument("--webCam", type=str2bool, default=False, help="Camera device index")
    return parser.parse_args()



def load_models(args):
    """Load RetinaFace and ArcFace ONNX models"""
    # Set up ONNX Runtime
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu >= 0 else ['CPUExecutionProvider']
    
    # Load RetinaFace detector
    print("Loading RetinaFace detector...")
    # Try to find the model
    import os
    model_paths = [
        "~/.insightface/models/buffalo_l/det_10g.onnx",
        "~/.insightface/models/buffalo_l/det_500m.onnx",
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
        "~/.insightface/models/buffalo_l/w600k_r50.onnx",
        "~/.insightface/models/buffalo_l/glintr100.onnx",
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
        bboxes, landmarks = detector.detect(frame, input_size=(640, 640))
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


def frame_generator(container, webCamUsage=True, max_width=1280, max_height=720):
    """
    Unified frame generator with resizing to fit the monitor/window.
    - webCamUsage=True: container is cv2.VideoCapture
    - webCamUsage=False: container is PyAV container (RTSP)
    - max_width, max_height: maximum size to fit frames to (maintains aspect ratio)
    """

    if webCamUsage:
        while True:
            ret, frame = container.read()
            if not ret:
                break
            yield frame
    else:
        for packet in container:
            frame = packet.to_ndarray(format="bgr24")
            yield frame

import cv2
import numpy as np
import torch
import kornia.geometry.transform as K


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
    det_thresh=0.5,
    args=None
):
    """
    RetinaFace detection + batch ArcFace embedding + Qdrant recognition.
    Returns: [(bbox, person, score), ...]
    """

    detector = RetinaFace(model_file=None, session=det_session)

    # 1️⃣ Detect faces
    try:
        bboxes, landmarks = detector.detect(frame, input_size=(640, 640))
    except TypeError:
        bboxes, landmarks = detector.detect(frame)

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
   
    for frame in frame_generator(container, args.webCam):
        recognized_faces = detect_faces_retinaface_gpu2(det_session, rec_session, client, frame, args.threshold, args)

        # Draw all recognized faces
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