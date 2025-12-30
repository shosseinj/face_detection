import cv2
import argparse
import numpy as np
import torch
from face_detection.utils import  * 
from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.box_utils import decode, decode_landmarks, nms


# ===============================
# Arguments
# ===============================
def parse_arguments():
    parser = argparse.ArgumentParser("RetinaFace GPU Minimal")

    parser.add_argument("--weights", default="./weights/retinaface_mv2.pth")
    parser.add_argument("--network", default="mobilenetv2")
    parser.add_argument("--source", default="0")

    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--target-size", type=int, default=640)
    parser.add_argument("--fp16", action="store_true")

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
    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap


# ===============================
# Main
# ===============================
def main(args):
    device = require_cuda()

    cfg = get_config(args.network)
    model = RetinaFace(cfg=cfg).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cuda", weights_only=True))

    rgb_mean = (104, 117, 123)

    cap = open_capture(args.source)
    cv2.namedWindow("RetinaFace GPU", cv2.WINDOW_NORMAL)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                    x1, y1, x2, y2, _ = dets[i]
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    for j in range(5):
                        cv2.circle(
                            frame,
                            (int(landmarks[i][2*j]), int(landmarks[i][2*j+1])),
                            2,
                            (0, 0, 255),
                            -1
                        )

            cv2.imshow("RetinaFace GPU", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_arguments())
