import cv2
import argparse
import numpy as np
import torch
import os
import sys
from utils.utils import * 

from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.box_utils import decode, decode_landmarks, nms
import jdatetime

# ===============================
# Arguments
# ===============================
def parse_arguments():
    parser = argparse.ArgumentParser("RetinaFace GPU Minimal")

    parser.add_argument("--weights", default="./models/weights/retinaface_mv2.pth")
    parser.add_argument("--network", default="mobilenetv2")
    parser.add_argument("--source", default="0")

    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--target-size", type=int, default=640)
    parser.add_argument("--fp16", action="store_true")
    
    # Add save directory argument
    parser.add_argument("--save-dir", default="../images/detected_faces", 
                       help="Directory to save detected faces")
    parser.add_argument("--save-format", default="jpg", choices=["jpg", "png"],
                       help="Image format for saving faces")
    parser.add_argument("--save-every-n", type=int, default=1,
                       help="Save every N detections (to avoid duplicates)")
                       

    parser.add_argument("--min_face_size", type=int, default=20,
                       help="Minimum face size in pixels to save")

    return parser.parse_args()


# ===============================
# CUDA ONLY
# ===============================
def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU REQUIRED")
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return device

# ===============================
# Resize for model only
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
    cap = cv2.VideoCapture(int(source))  # Remove cv2.CAP_DSHOW
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap


# ===============================
# Main
# ===============================
def main(args):
    device = require_cuda()

    # Create save directory
    save_dir = create_save_directory(args.save_dir)
    
    cfg = get_config(args.network)
    model = RetinaFace(cfg=cfg).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cuda", weights_only=True))

    rgb_mean = (104, 117, 123)

    cap = open_capture(args.source)
    cv2.namedWindow("RetinaFace GPU", cv2.WINDOW_NORMAL)
    
    # Initialize counters
    frame_count = 0
    total_faces_saved = 0
    
    # For tracking to avoid saving duplicates
    last_save_time = {}

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Create a copy for saving (original colors)
            original_frame = frame.copy()
            
            model_img, scale = resize_image(frame, args.target_size)
            h, w = model_img.shape[:2]

            img = np.float32(model_img)
            img -= rgb_mean
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                loc, conf, landmarks = model(img)

            priors = PriorBox(cfg, image_size=(w, h)).generate_anchors().to(device)

            boxes = decode(loc.squeeze(0), priors, cfg["variance"])
            landmarks = decode_landmarks(landmarks.squeeze(0), priors, cfg["variance"])
            scores = conf.squeeze(0)[:, 1]

            boxes *= torch.tensor([w, h, w, h], device=device)
            landmarks *= torch.tensor([w, h] * 5, device=device)

            boxes /= scale
            landmarks /= scale

            keep = scores > args.conf_threshold
            boxes = boxes[keep]
            landmarks = landmarks[keep]
            scores = scores[keep]

            if scores.numel() > 0:
                dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
                keep = nms(dets.cpu().numpy(), 0.4)

                dets = dets[keep].cpu().numpy()
                landmarks = landmarks[keep].cpu().numpy()
               
                for i in range(dets.shape[0]):
                    x1, y1, x2, y2, score = dets[i]
                    
                    # Convert to integers for drawing
                    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw rectangle
                    cv2.rectangle(
                        frame,
                        (x1_i, y1_i),
                        (x2_i, y2_i),
                        (0, 255, 0),
                        2
                    )
                    
                    # Draw score
                    cv2.putText(frame, f"{score:.2f}", 
                              (x1_i, y1_i - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 0), 2)
                    
                    # Draw landmarks
                    for j in range(5):
                        cv2.circle(
                            frame,
                            (int(landmarks[i][2*j]), int(landmarks[i][2*j+1])),
                            2,
                            (0, 0, 255),
                            -1
                        )
                    
                    # Save face if conditions are met
                    should_save = (
                        frame_count % args.save_every_n == 0 
                        and
                        (x2_i - x1_i) >= args.min_face_size and
                        (y2_i - y1_i) >= args.min_face_size
                    )
                    
                    if should_save :
                        # Extract face from original frame
                        face_img = extract_face(
                            original_frame, 
                            [x1_i, y1_i, x2_i, y2_i],
                            margin=0.4,
                            min_size=args.min_face_size
                        )
                        
                        if face_img is not None:
                            # Save the face
                            now_jalali = jdatetime.datetime.now()
                            timestamp = now_jalali.strftime("%Y-%m-%d_%H%M%S_%f")                            
                            save_path = save_face_image(
                                face_img, 
                                save_dir,
                                timestamp +f"{i}"
                            )
                            
                            total_faces_saved += 1
                            # print(f"Saved face {total_faces_saved}: {save_path}")
                
                # Display count on frame
                cv2.putText(frame, f"Faces: {dets.shape[0]}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 255), 2)
                cv2.putText(frame, f"Saved: {total_faces_saved}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 255), 2)
            
            cv2.imshow("RetinaFace GPU", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):  # Manual save on 's' key press
                print("Manual save triggered")
                # You could add manual save logic here

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal faces saved: {total_faces_saved}")
    print(f"Faces saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main(parse_arguments())