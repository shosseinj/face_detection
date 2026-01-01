import cv2
import argparse
import numpy as np
import torch
from utils.utils import *
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.box_utils import decode, decode_landmarks, nms
from scipy.spatial.distance import cosine

# ===============================
# Arguments
# ===============================
def parse_arguments():
    parser = argparse.ArgumentParser("RetinaFace GPU Real-Time")
    parser.add_argument("--weights", default="./weights/retinaface_mv2.pth")
    parser.add_argument("--network", default="mobilenetv2")
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.4)
    parser.add_argument("--min_face_size", type=int, default=20)
    return parser.parse_args()

# ===============================
# CUDA check
# ===============================
def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU REQUIRED")
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return device

# ===============================
# Resize for RetinaFace
# ===============================
def resize_image(frame, size):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale

# ===============================
# Open camera
# ===============================
def open_capture(source):
    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap

# ===============================
# Main loop
# ===============================
def main(args):
    device = require_cuda()
    cfg = get_config(args.network)

    # -------------------------------
    # RetinaFace model
    # -------------------------------
    model = RetinaFace(cfg=cfg).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    rgb_mean = torch.tensor([104, 117, 123], dtype=torch.float32, device=device).view(1, 1, 3)

    cap = open_capture(args.source)
    cv2.namedWindow("RetinaFace GPU", cv2.WINDOW_NORMAL)

    # -------------------------------
    # InsightFace model
    # -------------------------------
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # -------------------------------
    # Qdrant client
    # -------------------------------
    client = QdrantClient(host="localhost", port=6333)

    # -------------------------------
    # Face cache to skip repeated recognition
    # -------------------------------
    face_cache = []  # list of tuples: (embedding, name)
    update_every = 4  # update recognition every 4 frames
    tracked_faces = []  # will store {"bbox": [...], "name": ..., "embedding": ...} per face
    frame_count = 0
    # with torch.no_grad():
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frame_count += 1
    #         if frame_count % 10 == 0:
    #             original_frame = frame.copy()

    #             # -------------------------------
    #             # RetinaFace detection
    #             # -------------------------------
    #             model_img, scale = resize_image(frame, args.target_size)
    #             h, w = model_img.shape[:2]

    #             img = torch.from_numpy(model_img.astype(np.float32)).to(device)
    #             img = img - rgb_mean
    #             img = img.permute(2, 0, 1).unsqueeze(0)
    #             if args.fp16:
    #                 img = img.half()

    #             with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
    #                 loc, conf, landmarks = model(img)

    #             priors = PriorBox(cfg, image_size=(w, h)).generate_anchors().to(device)
    #             boxes = decode(loc.squeeze(0), priors, cfg["variance"]) * torch.tensor([w,h,w,h], device=device) / scale
    #             landmarks = decode_landmarks(landmarks.squeeze(0), priors, cfg["variance"]) * torch.tensor([w,h]*5, device=device) / scale
    #             scores = conf.squeeze(0)[:,1]

    #             keep = scores > args.conf_threshold
    #             boxes = boxes[keep]
    #             landmarks = landmarks[keep]
    #             scores = scores[keep]

    #             if scores.numel() > 0:
    #                 dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
    #                 keep = nms(dets.cpu().numpy(), 0.4)
    #                 dets = dets[keep].cpu().numpy()
    #                 landmarks = landmarks[keep].cpu().numpy()

    #                 face_imgs = []
    #                 face_boxes = []
    #                 for i in range(dets.shape[0]):
    #                     x1, y1, x2, y2, score = dets[i]
    #                     x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
    #                     face_img = extract_face(original_frame, [x1_i, y1_i, x2_i, y2_i], margin=0.4, min_size=args.min_face_size)
    #                     if face_img is not None:
    #                         face_imgs.append(face_img)
    #                         face_boxes.append([x1_i, y1_i, x2_i, y2_i])

    #                 # -------------------------------
    #                 # InsightFace per-face inference
    #                 # -------------------------------
    #                 faces_results = []
                
    #                 for face_img in face_imgs:
    #                     res = app.get(face_img)  # returns list of 0 or 1 faces
    #                     faces_results.append(res[0] if res else None)

    #                 for i, res in enumerate(faces_results):
    #                     x1_i, y1_i, x2_i, y2_i = face_boxes[i]
    #                     person_name = "Unknown"

    #                     if res:
    #                         emb = res.embedding.astype("float32")

    #                         # Check cache first
    #                         skip_recognition = False
    #                         for cached_emb, cached_name in face_cache:
    #                             if cosine(emb, cached_emb) < args.similarity_threshold:
    #                                 person_name = cached_name
    #                                 skip_recognition = True
    #                                 break

    #                         # If not in cache, query Qdrant
    #                         if not skip_recognition:
    #                             result = client.query_points(collection_name="face_embeddings", query=emb.tolist(), limit=1)
    #                             if result.points:
    #                                 best_match = result.points[0]
    #                                 person_name = best_match.payload.get("person", "Unknown")
    #                             face_cache.append((emb, person_name))

    #                     # Draw
    #                     cv2.rectangle(frame, (x1_i,y1_i), (x2_i,y2_i), (0,255,0), 2)
    #                     text_size = cv2.getTextSize(person_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    #                     cv2.rectangle(frame, (x1_i, y1_i - text_size[1] - 10),
    #                                 (x1_i + text_size[0] + 10, y1_i), (0,255,0), -1)
    #                     cv2.putText(frame, person_name, (x1_i + 5, y1_i - 5),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    #                     for j in range(5):
    #                         cv2.circle(frame, (int(landmarks[i][2*j]), int(landmarks[i][2*j+1])), 2, (0,0,255), -1)
            



           
    #         cv2.imshow("RetinaFace GPU", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             break
    


    face_cache = []
    update_every = 40  # update recognition every 4 frames
    tracked_faces = []  # will store {"bbox": [...], "name": ..., "embedding": ...} per face
    frame_count = 0

    # Store previous frame results
    prev_boxes = []
    prev_landmarks = []
    prev_names = []
    prev_scores = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            original_frame = frame.copy()
            
            # -------------------------------
            # RetinaFace detection (every frame)
            # -------------------------------
            model_img, scale = resize_image(frame, args.target_size)
            h, w = model_img.shape[:2]

            img = torch.from_numpy(model_img.astype(np.float32)).to(device)
            img = img - rgb_mean
            img = img.permute(2, 0, 1).unsqueeze(0)
            if args.fp16:
                img = img.half()

            with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
                loc, conf, landmarks = model(img)

            priors = PriorBox(cfg, image_size=(w, h)).generate_anchors().to(device)
            boxes = decode(loc.squeeze(0), priors, cfg["variance"]) * torch.tensor([w,h,w,h], device=device) / scale
            landmarks = decode_landmarks(landmarks.squeeze(0), priors, cfg["variance"]) * torch.tensor([w,h]*5, device=device) / scale
            scores = conf.squeeze(0)[:,1]

            keep = scores > args.conf_threshold
            boxes = boxes[keep]
            landmarks = landmarks[keep]
            scores = scores[keep]

            current_names = ["Unknown"] * len(boxes) if scores.numel() > 0 else []
            
            if scores.numel() > 0:
                dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
                keep = nms(dets.cpu().numpy(), 0.4)
                dets = dets[keep].cpu().numpy()
                landmarks = landmarks[keep].cpu().numpy()
                boxes_np = boxes[keep].cpu().numpy()

                # Store current detection results
                prev_boxes = boxes_np
                prev_landmarks = landmarks
                prev_scores = scores[keep].cpu().numpy()

                # -------------------------------
                # InsightFace recognition (only every N frames)
                # -------------------------------
                if frame_count % update_every == 0:
                    face_imgs = []
                    for i in range(dets.shape[0]):
                        x1, y1, x2, y2, score = dets[i]
                        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                        face_img = extract_face(original_frame, [x1_i, y1_i, x2_i, y2_i], margin=0.4, min_size=args.min_face_size)
                        face_imgs.append(face_img)
                    
                    # Batch process faces
                    faces_results = []
                    for face_img in face_imgs:
                        if face_img is not None:
                            res = app.get(face_img)
                            faces_results.append(res[0] if res else None)
                        else:
                            faces_results.append(None)
                    
                    # Update names based on recognition
                    for i, res in enumerate(faces_results):
                        if res:
                            emb = res.embedding.astype("float32")
                            
                            # Check cache first
                            skip_recognition = False
                            for cached_emb, cached_name in face_cache:
                                if cosine(emb, cached_emb) < args.similarity_threshold:
                                    current_names[i] = cached_name
                                    skip_recognition = True
                                    break
                            
                            # If not in cache, query Qdrant
                            if not skip_recognition:
                                try:
                                    result = client.query_points(
                                        collection_name="face_embeddings", 
                                        query=emb.tolist(), 
                                        limit=1
                                    )
                                    if result.points:
                                        best_match = result.points[0]
                                        current_names[i] = best_match.payload.get("person", "Unknown")
                                    face_cache.append((emb, current_names[i]))
                                except:
                                    current_names[i] = "Unknown"
                        else:
                            current_names[i] = "Unknown"
                    
                    # Store names for next frames
                    prev_names = current_names.copy()
                else:
                    # Use previous recognition results
                    current_names = prev_names
            
            # Draw boxes and names for current frame
            for i, (box, score) in enumerate(zip(prev_boxes, prev_scores) if len(prev_boxes) > 0 else []):
                x1, y1, x2, y2 = box[:4]
                x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                
                # Draw name (if available)
                if i < len(current_names):
                    person_name = current_names[i]
                    # Add confidence score to name
                    name_with_score = f"{person_name} ({score:.2f})"
                    
                    text_size = cv2.getTextSize(name_with_score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1_i, y1_i - text_size[1] - 10),
                                (x1_i + text_size[0] + 10, y1_i), (0, 255, 0), -1)
                    cv2.putText(frame, name_with_score, (x1_i + 5, y1_i - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw landmarks (if available)
                if i < len(prev_landmarks):
                    for j in range(5):
                        cv2.circle(frame, (int(prev_landmarks[i][2*j]), int(prev_landmarks[i][2*j+1])), 
                                2, (0, 0, 255), -1)
            
            # Show FPS
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("RetinaFace GPU", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_arguments())