import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient

def parse_args():
    parser = argparse.ArgumentParser("Real-time Face Recognition from Webcam")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    parser.add_argument("--collection", type=str, default="n3", help="Qdrant collection name")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    return parser.parse_args()

def open_capture(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap

def initialize_models(args):
    """Initialize face analysis model and Qdrant client"""
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"] if args.gpu >= 0 else ["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=args.gpu, det_size=(640, 640))
    
    client = QdrantClient(url="http://localhost:6333")
    
    return app, client

def recognize_face(embedding, client, args):
    """Recognize a single face embedding"""

    
    search_result = client.query_points(
                                        collection_name="n3", 
                                        query= embedding.tolist(), 
                                        limit=1
                                    ).points
    

    
    if search_result and len(search_result) > 0:
        if isinstance(search_result, tuple):
            points = search_result[0]
        else:
            points = search_result
        
        if points and len(points) > 0:
            best_match = points[0]
            person = "Unknown"
            score = 0.0
            
            if hasattr(best_match, 'payload') and best_match.payload:
                person = best_match.payload.get("person", "Unknown")
            if hasattr(best_match, 'score'):
                score = best_match.score
            
            return person, score
    
    return "Unknown", 0.0

def draw_face_info(frame, bbox, person, score, color):
    """Draw face bounding box and name on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    if person == "Unknown":
        label = "Unknown"
    else:
        label = f"{person} ({score:.2f})"
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw background rectangle for text
    text_bg_top = max(y1 - text_height - 10, 0)
    text_bg_bottom = y1
    text_bg_left = x1
    text_bg_right = x1 + text_width
    
    cv2.rectangle(frame,
                 (text_bg_left, text_bg_top),
                 (text_bg_right, text_bg_bottom),
                 color, -1)
    
    # Draw text
    cv2.putText(frame, label,
               (x1, y1 - 5),
               font, font_scale, (255, 255, 255), thickness)
    
    return frame

def main():
    args = parse_args()
    
    print("Initializing models...")
    app, client = initialize_models(args)
    
    print(f"Opening camera {args.camera}...")
    cap = open_capture(args.camera)
    
    print("Starting real-time face recognition. Press 'q' to quit.")
    
    # Colors for different persons
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 255), # White
        (0, 165, 255),  # Orange
    ]
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    # Recognition cache to improve performance
    face_cache = {}
    cache_size = 50
    cache_timeout = 2.0  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Mirror frame for more natural viewing
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces = app.get(frame)
        
        # Process each face
        recognized_faces = []
        for i, face in enumerate(faces):
            bbox = face.bbox
            
            # Check cache first
            cache_key = tuple(bbox.astype(int))
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            if cache_key in face_cache:
                cache_entry = face_cache[cache_key]
                if current_time - cache_entry['timestamp'] < cache_timeout:
                    person, score = cache_entry['person'], cache_entry['score']
                else:
                    # Cache expired, recognize again
                    person, score = recognize_face(face.embedding, client, args)
                    face_cache[cache_key] = {
                        'person': person, 
                        'score': score, 
                        'timestamp': current_time
                    }
            else:
                # Recognize face
                person, score = recognize_face(face.embedding, client, args)
                face_cache[cache_key] = {
                    'person': person, 
                    'score': score, 
                    'timestamp': current_time
                }
                
                # Limit cache size
                if len(face_cache) > cache_size:
                    # Remove oldest entry
                    oldest_key = min(face_cache.keys(), 
                                   key=lambda k: face_cache[k]['timestamp'])
                    del face_cache[oldest_key]
            
            recognized_faces.append((bbox, person, score, i))
        
        # Draw all faces
        for bbox, person, score, idx in recognized_faces:
            color_idx = hash(person) % len(colors)
            color = colors[color_idx]
            frame = draw_face_info(frame, bbox, person, score, color)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = cv2.getTickCount()
            time_elapsed = (end_time - start_time) / cv2.getTickFrequency()
            fps = 30 / time_elapsed
            start_time = end_time
        
        # Display FPS and stats
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {args.threshold}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Real-time Face Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")

if __name__ == "__main__":
    main()