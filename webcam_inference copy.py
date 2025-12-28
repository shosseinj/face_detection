import os
import cv2
import argparse
import numpy as np

import torch
import time
import threading

from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms

import cv2

# found = False
# for i in range(5):
#     cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
#     if cap.isOpened():
#         print(f"Camera found at index {i}")
#         found = True
#         cap.release()
#         break
#     cap.release()

# if not found:
#     print("No working camera found. Check drivers or other apps.")



def parse_arguments():
    parser = argparse.ArgumentParser(description="Retinaface Webcam Inference")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/retinaface_mv2.pth',
        help='Path to the trained model weights'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.4,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )
    
    # Video saving options
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Input video path or Webcam source (default: 0)'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Enable saving the processed video'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./output_video.mp4',
        help='Path to save the output video'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=24.0,
        help='FPS for the output video'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable showing the GUI window (useful on headless servers)'
    )
    parser.add_argument(
        '--low-latency',
        action='store_true',
        help='Enable low-latency mode: background frame grabber and always process latest frame'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Enable mixed-precision (FP16) inference (requires CUDA)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=640,
        help='Square size to resize input images to before inference (default: 640)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['dshow', 'msmf', 'any'],
        default='any',
        help='Preferred video backend to use when opening a camera (dshow/msmf/any)'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(model, image, use_amp=False):
    """Run model inference. If use_amp is True and CUDA is available, use autocast."""
    model.eval()
    if use_amp:
        with torch.cuda.amp.autocast():
            loc, conf, landmarks = model(image)
    else:
        loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks


def resize_image(frame, target_shape=(640, 640)):
    width, height = target_shape

    # Aspect-ratio preserving resize
    im_ratio = float(frame.shape[0]) / frame.shape[1]
    model_ratio = height / width
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)

    resize_factor = float(new_height) / frame.shape[0]
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create blank image and place resized image on it
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:new_height, :new_width, :] = resized_frame

    return image, resize_factor


def main(params):
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # model initialization
    model = RetinaFace(cfg=cfg)
    model.to(device)

    # loading state_dict
    state_dict = torch.load(params.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    def open_capture(source, max_index_search=5, backend_pref='any'):
        """Open a VideoCapture robustly.

        If `source` is a digit (webcam index), try multiple Windows backends
        and search other indices as a fallback. For file paths, try opening
        directly.
        """
        # Map backend constants to friendly names for logging
        backend_names = {
            cv2.CAP_DSHOW: 'CAP_DSHOW',
            cv2.CAP_MSMF: 'CAP_MSMF'
        }

        # Translate backend_pref token to a prioritized list of backends
        if backend_pref == 'dshow':
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        elif backend_pref == 'msmf':
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]
        else:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        if str(source).isdigit():
            idx = int(source)

            # Try the requested index with multiple backends. Only return a
            # capture if we can actually read at least one frame (warm-up).
            for backend in backends:
                if backend is not None:
                    cap = cv2.VideoCapture(idx, backend)
                else:
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    name = backend_names.get(backend, 'default')
                    # Try reading a frame to ensure backend actually returns data
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Opened camera index {idx} with backend {name}")
                        return cap, frame
                    else:
                        print(f"Opened camera index {idx} with backend {name} but could not read frame; trying next backend/index")
                cap.release()

            # Try searching other indices (0..max_index_search-1)
            for i in range(0, max_index_search):
                for backend in backends:
                    if backend is not None:
                        cap = cv2.VideoCapture(i, backend)
                    else:
                        cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        name = backend_names.get(backend, 'default')
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"Opened camera index {i} with backend {name}")
                            return cap, frame
                        else:
                            print(f"Opened camera index {i} with backend {name} but could not read frame; trying next")
                    cap.release()
            return None, None
        else:
            # Treat as file path / stream URL
            if not os.path.exists(source):
                print(f"Error: video file '{source}' not found.")
                return None, None
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Try reading a frame to ensure file/stream is ok
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, frame
                cap.release()
            return None, None

    cap, first_frame = open_capture(params.source, backend_pref=params.backend)


    # Open webcam
    # if params.source.isdigit():
    #     cap = cv2.VideoCapture(int(params.source))
    # else:
    #     cap = cv2.VideoCapture(params.source)
    
    # if not cap.isOpened():
    #     print("Error: Could not open video source.")
    #     return

    if cap is None or not cap.isOpened() or first_frame is None:
        print("Error: Could not open video source or read an initial frame. Check camera connection, permissions, or whether another app is using the camera.")
        return

    frame_height, frame_width = first_frame.shape[:2]

    # Initialize video writer if save option is enabled
    video_writer = None
    if params.save_video:
        # Ensure output directory exists
        output_dir = os.path.dirname(params.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        video_writer = cv2.VideoWriter(params.output_path, fourcc, params.fps, (frame_width, frame_height))
        print(f"Video will be saved to: {params.output_path}")

    # Low-latency background frame grabber (keeps only the latest frame)
    class FrameGrabber(threading.Thread):
        def __init__(self, cap):
            super().__init__(daemon=True)
            self.cap = cap
            self.lock = threading.Lock()
            self.latest = None
            self.running = True

        def run(self):
            while self.running:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest = frame
                else:
                    # small sleep to avoid busy loop when camera stalls
                    time.sleep(0.01)

        def get_latest(self):
            with self.lock:
                if self.latest is None:
                    return None
                return self.latest.copy()

        def stop(self):
            self.running = False

    grabber = None
    if getattr(params, 'low_latency', False) and params.source.isdigit():
        grabber = FrameGrabber(cap)
        grabber.start()
        # use already captured first_frame as initial latest
        with grabber.lock:
            grabber.latest = first_frame

    # Create display window and show initial frame (if display not disabled)
    if not getattr(params, 'no_display', False):
        try:
            cv2.namedWindow('Webcam Inference', cv2.WINDOW_NORMAL)
            cv2.imshow('Webcam Inference', first_frame)
            # Force GUI event processing
            cv2.waitKey(1)
            vis = cv2.getWindowProperty('Webcam Inference', cv2.WND_PROP_VISIBLE)
            if vis < 1:
                print('Warning: display window is not visible. If you are running headless, use --no-display.')
        except Exception as e:
            print(f'Warning: could not create display window: {e}')

    # Process the first frame we already captured
    frame = first_frame

    while True:
        # In low-latency mode, always grab the latest available frame from
        # the background grabber. Otherwise, read from capture directly.
        if grabber is not None:
            frame = grabber.get_latest()
            if frame is None:
                # no frame available yet; small sleep to avoid busy loop
                time.sleep(0.005)
                continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Could not read frame.")
                break

        image, resize_factor = resize_image(frame, target_shape=(params.target_size, params.target_size))

        # Prepare image for inference
        image = np.float32(image)
        img_height, img_width, _ = image.shape
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        # forward pass (use mixed precision if requested and CUDA is available)
        use_amp = params.fp16 and device.type == 'cuda'
        loc, conf, landmarks = inference(model, image, use_amp=use_amp)

        # generate anchor boxes
        priorbox = PriorBox(cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors().to(device)

        # decode boxes and landmarks
        boxes = decode(loc, priors, cfg['variance'])
        landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

        # scale adjustments
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
        boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
        landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

        scores = conf.cpu().numpy()[:, 1]

        # filter by confidence threshold
        inds = scores > params.conf_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # sort by scores
        order = scores.argsort()[::-1][:params.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, params.nms_threshold)

        detections = detections[keep]
        landmarks = landmarks[keep]

        # keep top-k detections and landmarks
        detections = detections[:params.post_nms_topk]
        landmarks = landmarks[:params.post_nms_topk]

        # concatenate detections and landmarks
        detections = np.concatenate((detections, landmarks), axis=1)

        # draw detections on the frame
        draw_detections(frame, detections, params.vis_threshold)

        # Write frame to output video if enabled
        if params.save_video and video_writer is not None:
            video_writer.write(frame)

        # Display the resulting frame (unless disabled)
        if not getattr(params, 'no_display', False):
            try:
                cv2.imshow('Webcam Inference', frame)
            except Exception:
                pass

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if grabber is not None:
        grabber.stop()
        grabber.join(timeout=1.0)
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    if params.save_video:
        print(f"Video saved successfully to {params.output_path}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
