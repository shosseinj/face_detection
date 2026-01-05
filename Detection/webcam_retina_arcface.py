import cv2
import numpy as np
import argparse
import onnxruntime as ort
from qdrant_client import QdrantClient
from insightface.utils import face_align
from insightface.model_zoo.retinaface import RetinaFace

def parse_args():
    parser = argparse.ArgumentParser("Real-time Face Recognition from Webcam")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--collection", type=str, default="n3", help="Qdrant collection name")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    return parser.parse_args()


def resize_frame(frame, target_width=2048):
    """Resize frame for display while maintaining aspect ratio"""
    if frame is None:
        return None
    
    height, width = frame.shape[:2]
    
    # Calculate new dimensions while maintaining aspect ratio
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    
    # Resize
    resized = cv2.resize(frame, (target_width, new_height))
    return resized


# def open_capture(camera_id):

#     rtsp_url = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301/"
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     # cap = cv2.VideoCapture(camera_id)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open camera {camera_id}")
#     # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     # cap.set(cv2.CAP_PROP_FPS, 30)
    

#     # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 5)
#     return cap


# def open_capture(camera_id):
#     # Change channel number or try different stream format
#     # Channel 301 might be HEVC, try channel 101 for H.264
#     rtsp_url = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301/"
#     # Or try substream (usually H.264)
#     # rtsp_url = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/102/"
    
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
#     # Try different backend if FFMPEG has issues
#     # cap = cv2.VideoCapture(rtsp_url, cv2.CAP_GSTREAMER)
    
#     if not cap.isOpened():
#         # Fallback to RTSP without specific channel
#         rtsp_url = "rtsp://Jafari:Asd12345@192.168.110.20:554"
#         cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open camera {camera_id}")
    
#     return cap


# Install: pip install hikvisionapi
from hikvisionapi import Client


from hikvisionapi import Client


# def open_capture(camera_id):
#     rtsp_url = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301/"
    
#     # Add FFMPEG parameters for lower latency
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
#     # Set buffer size and latency options
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimum
#     cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
    
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open camera {camera_id}")
    
#     return cap


from hikvisionapi import Client

def open_capture(camera_id):
    # Hikvision SDK typically uses port 8000
    port = 8000
    
    try:
        print(f"Connecting to Hikvision SDK on port {port}...")
        
        # Create SDK client
        cam = Client(f'http://192.168.110.20:{port}', 'Jafari', 'Asd12345', timeout=10)
        
        # Test connection
        device_info = cam.System.deviceInfo(method='get')
        print(f"Connected! Device: {device_info.get('deviceName', 'Unknown')}")
        
        # Get channel information for channel 3
        try:
            # First, check what channels are available
            channels_info = cam.Streaming.channels(method='get')
            print(f"Total channels available: {len(channels_info)}")
            
            # Channel indexing usually starts from 1
            # Channel 3 typically corresponds to:
            # - Main stream: 301 or 103 depending on camera model
            # - Sub stream: 302 or 203
            
            # Try different channel mappings for channel 3
            channel_mappings = [
                '301',  # Most common: Channel 3, Main stream
                '302',  # Channel 3, Sub stream
                '103',  # Alternative: Channel 1, Stream 3
                '203',  # Channel 2, Stream 3
                '303',  # Channel 3, Stream 3
                '3',    # Simple channel 3
                3,      # Integer channel 3
            ]
            
            # Method 1: Try ISAPI streaming URL (most reliable)
            for channel in channel_mappings:
                try:
                    print(f"Trying channel {channel}...")
                    
                    # Get RTSP URL via ISAPI
                    stream_config = cam.Streaming.channels[channel](method='get')
                    
                    if 'Video' in stream_config:
                        print(f"Channel {channel} config: {stream_config}")
                        
                        # Construct RTSP URL for channel 3
                        # Format: rtsp://username:password@ip:port/Streaming/Channels/301
                        stream_url = f"rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/{channel}"
                        
                        # Try to open the stream
                        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                        if cap.isOpened():
                            print(f"✓ Successfully connected to channel {channel}")
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            return cap
                        else:
                            print(f"  Could not open stream for channel {channel}")
                            
                except Exception as e:
                    print(f"  Channel {channel} failed: {str(e)[:50]}")
                    continue
            
            # Method 2: Direct SDK capture (if RTSP fails)
            print("Trying direct SDK capture...")
            
            # Hikvision SDK direct preview URL
            sdk_urls = [
                f"rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301?transportmode=unicast",
                f"rtsp://Jafari:Asd12345@192.168.110.20:554/ISAPI/Streaming/channels/301",
                f"rtsp://Jafari:Asd12345@192.168.110.20:554/h264/ch3/main/av_stream",
                f"rtsp://Jafari:Asd12345@192.168.110.20:554/3",
            ]
            
            for sdk_url in sdk_urls:
                try:
                    cap = cv2.VideoCapture(sdk_url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        print(f"✓ Connected via SDK URL: {sdk_url}")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 15)
                        return cap
                except:
                    continue
                    
        except Exception as e:
            print(f"Channel configuration error: {e}")
            
    except Exception as e:
        print(f"Hikvision SDK connection failed: {e}")
    
    # Fallback to direct RTSP if SDK fails
    print("Falling back to direct RTSP connection...")
    
    # Direct RTSP URLs for channel 3
    rtsp_urls = [
        "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301",
        "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/302",
        "rtsp://Jafari:Asd12345@192.168.110.20:554/ISAPI/Streaming/channels/301",
        "rtsp://Jafari:Asd12345@192.168.110.20:554/h264/ch3/main/av_stream",
        "rtsp://Jafari:Asd12345@192.168.110.20:554/onvif1",
    ]
    
    for url in rtsp_urls:
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                print(f"✓ Connected via direct RTSP: {url}")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
                return cap
        except:
            continue
    
    raise RuntimeError(f"Cannot connect to camera channel 3")



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

def main():
    args = parse_args()
    
    print("Loading models...")
    det_session, rec_session, client = load_models(args)
    
    print(f"Opening camera {args.camera}...")
    cap = open_capture(args.camera)
    
    print("Starting real-time face recognition. Press 'q' to quit.")
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 255, 255), (0, 165, 255)
    ]
    
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    face_cache = {}
    cache_size = 50
    cache_timeout = 2.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # frame = cv2.flip(frame, 1)
        frame = resize_frame(frame)
        cv2.imshow('Real-time Face Recognition (RetinaFace+ArcFace)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
        # Detect faces with RetinaFace
        faces = detect_faces_retinaface(det_session, frame, args.threshold)
        
        recognized_faces = []
        for i, face in enumerate(faces):
            bbox = face['bbox']
            
            # Check cache
            cache_key = tuple(bbox.astype(int))
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            if cache_key in face_cache:
                cache_entry = face_cache[cache_key]
                if current_time - cache_entry['timestamp'] < cache_timeout:
                    person, score = cache_entry['person'], cache_entry['score']
                else:
                    # Extract and align face
                    if face['landmark'] is not None:
                        try:
                            aligned_face = face_align.norm_crop(frame, face['landmark'])
                        except:
                            # Fallback to bbox crop
                            x1, y1, x2, y2 = map(int, bbox)
                            margin = 0.2
                            h, w = frame.shape[:2]
                            x1 = max(0, int(x1 - (x2 - x1) * margin))
                            y1 = max(0, int(y1 - (y2 - y1) * margin))
                            x2 = min(w, int(x2 + (x2 - x1) * margin))
                            y2 = min(h, int(y2 + (y2 - y1) * margin))
                            aligned_face = frame[y1:y2, x1:x2]
                            if aligned_face.size == 0:
                                continue
                            aligned_face = cv2.resize(aligned_face, (112, 112))
                    else:
                        # Crop using bbox
                        x1, y1, x2, y2 = map(int, bbox)
                        margin = 0.2
                        h, w = frame.shape[:2]
                        x1 = max(0, int(x1 - (x2 - x1) * margin))
                        y1 = max(0, int(y1 - (y2 - y1) * margin))
                        x2 = min(w, int(x2 + (x2 - x1) * margin))
                        y2 = min(h, int(y2 + (y2 - y1) * margin))
                        aligned_face = frame[y1:y2, x1:x2]
                        if aligned_face.size == 0:
                            continue
                        aligned_face = cv2.resize(aligned_face, (112, 112))
                    
                    # Extract embedding
                    embedding = extract_embedding_arcface(rec_session, aligned_face)
                    
                    # Recognize
                    person, score = recognize_face(embedding, client, args)
                    face_cache[cache_key] = {
                        'person': person, 
                        'score': score, 
                        'timestamp': current_time
                    }
            else:
                # Extract and align face
                if face['landmark'] is not None:
                    try:
                        aligned_face = face_align.norm_crop(frame, face['landmark'])
                    except:
                        # Fallback to bbox crop
                        x1, y1, x2, y2 = map(int, bbox)
                        margin = 0.2
                        h, w = frame.shape[:2]
                        x1 = max(0, int(x1 - (x2 - x1) * margin))
                        y1 = max(0, int(y1 - (y2 - y1) * margin))
                        x2 = min(w, int(x2 + (x2 - x1) * margin))
                        y2 = min(h, int(y2 + (y2 - y1) * margin))
                        aligned_face = frame[y1:y2, x1:x2]
                        if aligned_face.size == 0:
                            continue
                        aligned_face = cv2.resize(aligned_face, (112, 112))
                else:
                    # Crop using bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    margin = 0.2
                    h, w = frame.shape[:2]
                    x1 = max(0, int(x1 - (x2 - x1) * margin))
                    y1 = max(0, int(y1 - (y2 - y1) * margin))
                    x2 = min(w, int(x2 + (x2 - x1) * margin))
                    y2 = min(h, int(y2 + (y2 - y1) * margin))
                    aligned_face = frame[y1:y2, x1:x2]
                    if aligned_face.size == 0:
                        continue
                    aligned_face = cv2.resize(aligned_face, (112, 112))
                
                # Extract embedding
                embedding = extract_embedding_arcface(rec_session, aligned_face)
                
                # Recognize
                person, score = recognize_face(embedding, client, args)
                face_cache[cache_key] = {
                    'person': person, 
                    'score': score, 
                    'timestamp': current_time
                }
                
                if len(face_cache) > cache_size:
                    oldest_key = min(face_cache.keys(), 
                                   key=lambda k: face_cache[k]['timestamp'])
                    del face_cache[oldest_key]
            
            recognized_faces.append((bbox, person, score, i))
        
        # Draw all faces
        for bbox, person, score, idx in recognized_faces:
            color_idx = hash(person) % len(colors)
            color = colors[color_idx]
            frame = draw_face_info(frame, bbox, person, score, color)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = cv2.getTickCount()
            time_elapsed = (end_time - start_time) / cv2.getTickFrequency()
            fps = 30 / time_elapsed
            start_time = end_time
        
        # Display info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Det thresh: {args.threshold}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Real-time Face Recognition (RetinaFace+ArcFace)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")

if __name__ == "__main__":
    main()