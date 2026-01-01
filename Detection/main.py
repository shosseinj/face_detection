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


from scipy.spatial.distance import cdist
import numpy as np

# Add after your imports
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

class SimpleFaceTracker:
    def __init__(self, max_age=10, iou_threshold=0.5, cos_threshold=0.6):
        self.faces = {}  # id: {"bbox": [], "name": "", "age": 0, "embedding": None, "confidence": 0}
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.cos_threshold = cos_threshold
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def update(self, detections, names=None, embeddings=None, scores=None):
        """
        Update tracker with new detections
        
        Args:
            detections: list of [x1, y1, x2, y2]
            names: list of names (optional)
            embeddings: list of face embeddings (optional)
            scores: list of confidence scores (optional)
        """
        # Age all current faces
        for face_id in list(self.faces.keys()):
            self.faces[face_id]["age"] += 1
            if self.faces[face_id]["age"] > self.max_age:
                print(f"Removing face {face_id} due to age")
                del self.faces[face_id]
        
        # If no detections, return current faces
        if not detections:
            return self.faces
        
        # Convert to numpy arrays
        detections_np = np.array(detections)
        
        # Prepare lists for new data
        det_names = names if names else ["Unknown"] * len(detections)
        det_embeddings = embeddings if embeddings else [None] * len(detections)
        det_scores = scores if scores else [0.5] * len(detections)
        
        if self.faces:
            face_ids = list(self.faces.keys())
            face_boxes = np.array([self.faces[fid]["bbox"] for fid in face_ids])
            
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(detections), len(face_ids)))
            for i, det in enumerate(detections):
                for j, face_box in enumerate(face_boxes):
                    iou_matrix[i, j] = self.iou(det, face_box)
            
            # Match based on IoU
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(face_ids)))
            
            # Greedy matching (you could use Hungarian algorithm for better results)
            for i in range(len(detections)):
                if not unmatched_detections:
                    break
                    
                best_iou = 0
                best_j = -1
                
                for j in unmatched_tracks:
                    if iou_matrix[i, j] > best_iou and iou_matrix[i, j] > self.iou_threshold:
                        best_iou = iou_matrix[i, j]
                        best_j = j
                
                if best_j != -1:
                    matches.append((i, best_j))
                    unmatched_detections.remove(i)
                    unmatched_tracks.remove(best_j)
            
            # Update matched faces
            for det_idx, track_idx in matches:
                face_id = face_ids[track_idx]
                self.faces[face_id]["bbox"] = detections[det_idx]
                self.faces[face_id]["age"] = 0
                self.faces[face_id]["confidence"] = det_scores[det_idx]
                
                # Update name and embedding if recognition was done
                if det_names[det_idx] != "Unknown":
                    self.faces[face_id]["name"] = det_names[det_idx]
                if det_embeddings[det_idx] is not None:
                    self.faces[face_id]["embedding"] = det_embeddings[det_idx]
            
            # Create new faces for unmatched detections
            for det_idx in unmatched_detections:
                face_id = self.next_id
                self.faces[face_id] = {
                    "bbox": detections[det_idx],
                    "name": det_names[det_idx],
                    "age": 0,
                    "embedding": det_embeddings[det_idx],
                    "confidence": det_scores[det_idx]
                }
                self.next_id += 1
            
            # Remove very old unmatched tracks
            for track_idx in unmatched_tracks:
                face_id = face_ids[track_idx]
                if self.faces[face_id]["age"] > self.max_age // 2:
                    del self.faces[face_id]
        
        else:
            # First frame - create all new faces
            for i in range(len(detections)):
                face_id = self.next_id
                self.faces[face_id] = {
                    "bbox": detections[i],
                    "name": det_names[i],
                    "age": 0,
                    "embedding": det_embeddings[i],
                    "confidence": det_scores[i]
                }
                self.next_id += 1
        
        return self.faces
    
    def get_active_faces(self):
        """Get all active faces (age <= max_age)"""
        active_faces = {}
        for face_id, face in self.faces.items():
            if face["age"] <= self.max_age:
                active_faces[face_id] = face
        return active_faces
    



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
   


    face_cache = []
    update_every = 40  # update recognition every 4 frames
    tracked_faces = []  # will store {"bbox": [...], "name": ..., "embedding": ...} per face
    frame_count = 0

    # Store previous frame results
    prev_boxes = []
    prev_landmarks = []
    prev_names = []
    prev_scores = []
    prev_face_count = 0          # <- add this line
    prev_boxes      = np.empty((0,5), dtype=np.float32)
    prev_landmarks  = np.empty((0,10), dtype=np.float32)
    prev_names      = []
# -------------------
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
                new_face_count = dets.shape[0]
                face_count_changed = (new_face_count != prev_face_count)
                prev_face_count = new_face_count
                if face_count_changed or frame_count % update_every == 0:
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
            if scores.numel() == 0:          # nothing detected this frame
                prev_boxes   = np.empty((0,5), dtype=np.float32)
                prev_landmarks = np.empty((0,10), dtype=np.float32)
                prev_names   = []            # clear cached names
                current_names = []
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
   
   
   
    # Initialize at the beginning (after your other initializations)
    # tracker = SimpleFaceTracker(max_age=15, iou_threshold=0.3)

    # # Replace your current main loop with this:
    # with torch.no_grad():
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
            
    #         frame_count += 1
    #         original_frame = frame.copy()
            
    #         # -------------------------------
    #         # RetinaFace detection (every frame)
    #         # -------------------------------
    #         model_img, scale = resize_image(frame, args.target_size)
    #         h, w = model_img.shape[:2]

    #         img = torch.from_numpy(model_img.astype(np.float32)).to(device)
    #         img = img - rgb_mean
    #         img = img.permute(2, 0, 1).unsqueeze(0)
    #         if args.fp16:
    #             img = img.half()

    #         with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
    #             loc, conf, landmarks = model(img)

    #         priors = PriorBox(cfg, image_size=(w, h)).generate_anchors().to(device)
    #         boxes = decode(loc.squeeze(0), priors, cfg["variance"]) * torch.tensor([w,h,w,h], device=device) / scale
    #         landmarks = decode_landmarks(landmarks.squeeze(0), priors, cfg["variance"]) * torch.tensor([w,h]*5, device=device) / scale
    #         scores = conf.squeeze(0)[:,1]

    #         keep = scores > args.conf_threshold
    #         boxes = boxes[keep]
    #         landmarks = landmarks[keep]
    #         scores = scores[keep]

    #         current_names = ["Unknown"] * len(boxes) if scores.numel() > 0 else []
    #         current_embeddings = [None] * len(boxes) if scores.numel() > 0 else []
            
    #         if scores.numel() > 0:
    #             dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
    #             keep = nms(dets.cpu().numpy(), 0.4)
    #             dets = dets[keep].cpu().numpy()
    #             landmarks = landmarks[keep].cpu().numpy()
    #             boxes_np = boxes[keep].cpu().numpy()
    #             scores_np = scores[keep].cpu().numpy()

    #             # -------------------------------
    #             # InsightFace recognition (only every N frames)
    #             # -------------------------------
    #             if frame_count % update_every == 0:
    #                 face_imgs = []
    #                 face_indices = []
                    
    #                 # Prepare face images for recognition
    #                 for i in range(dets.shape[0]):
    #                     x1, y1, x2, y2, score = dets[i]
    #                     x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
    #                     face_img = extract_face(original_frame, [x1_i, y1_i, x2_i, y2_i], margin=0.4, min_size=args.min_face_size)
    #                     if face_img is not None:
    #                         face_imgs.append(face_img)
    #                         face_indices.append(i)
                    
    #                 # Batch process faces
    #                 faces_results = []
    #                 for face_img in face_imgs:
    #                     res = app.get(face_img)
    #                     faces_results.append(res[0] if res else None)
                    
    #                 # Update names based on recognition
    #                 for idx, (result_idx, res) in enumerate(zip(face_indices, faces_results)):
    #                     if res:
    #                         emb = res.embedding.astype("float32")
                            
    #                         # Check cache first
    #                         skip_recognition = False
    #                         for cached_emb, cached_name in face_cache:
    #                             if cosine(emb, cached_emb) < args.similarity_threshold:
    #                                 current_names[result_idx] = cached_name
    #                                 current_embeddings[result_idx] = emb
    #                                 skip_recognition = True
    #                                 break
                            
    #                         # If not in cache, query Qdrant
    #                         if not skip_recognition:
    #                             try:
    #                                 result = client.query_points(
    #                                     collection_name="face_embeddings", 
    #                                     query=emb.tolist(), 
    #                                     limit=1
    #                                 )
    #                                 if result.points:
    #                                     best_match = result.points[0]
    #                                     current_names[result_idx] = best_match.payload.get("person", "Unknown")
    #                                 else:
    #                                     current_names[result_idx] = "Unknown"
                                    
    #                                 face_cache.append((emb, current_names[result_idx]))
    #                                 current_embeddings[result_idx] = emb
    #                             except Exception as e:
    #                                 print(f"Qdrant error: {e}")
    #                                 current_names[result_idx] = "Unknown"
    #                     else:
    #                         current_names[result_idx] = "Unknown"
                
    #             # Prepare detections for tracker
    #             detections_list = []
    #             names_list = []
    #             embeddings_list = []
                
    #             for i in range(dets.shape[0]):
    #                 x1, y1, x2, y2, score = dets[i]
    #                 detections_list.append([int(x1), int(y1), int(x2), int(y2)])
    #                 names_list.append(current_names[i] if i < len(current_names) else "Unknown")
    #                 embeddings_list.append(current_embeddings[i] if i < len(current_embeddings) else None)
                
    #             # Update tracker
    #             tracked_faces = tracker.update(
    #                 detections=detections_list,
    #                 names=names_list,
    #                 embeddings=embeddings_list,
    #                 scores=scores_np.tolist()
    #             )
            
    #         else:
    #             # No detections, just update tracker (ages faces)
    #             tracked_faces = tracker.update(detections=[])
            
    #         # Draw tracked faces
    #         for face_id, face in tracked_faces.items():
    #             x1, y1, x2, y2 = face["bbox"]
    #             name = face.get("name", "Unknown")
    #             confidence = face.get("confidence", 0.5)
                
    #             # Draw bounding box with color based on age
    #             age = face.get("age", 0)
    #             if age < 5:
    #                 color = (0, 255, 0)  # Green for fresh tracks
    #             elif age < 10:
    #                 color = (0, 255, 255)  # Yellow for medium age
    #             else:
    #                 color = (0, 165, 255)  # Orange for old tracks
                
    #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
    #             # Draw ID and name
    #             label = f"ID:{face_id} {name}"
    #             if frame_count % update_every == 0:  # Show confidence only on recognition frames
    #                 label += f" ({confidence:.2f})"
                
    #             text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    #             cv2.rectangle(frame, (int(x1), int(y1) - text_size[1] - 10),
    #                         (int(x1) + text_size[0] + 10, int(y1)), color, -1)
    #             cv2.putText(frame, label, (int(x1) + 5, int(y1) - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    #         # Show FPS and stats
    #         cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    #         cv2.putText(frame, f"Active faces: {len(tracked_faces)}", (10, 60),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    #         cv2.imshow("RetinaFace GPU + Tracker", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_arguments())