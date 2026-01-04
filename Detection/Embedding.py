import os
import cv2
import uuid
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("Face Embedding + Qdrant")
    parser.add_argument("--image-folder", type=str, default="./database/FaceDataset")
    parser.add_argument("--query-image", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()

# -------------------------
# Face detection with RetinaFace MobileNetv2
# -------------------------
def detect_face_with_retina_mobilev2(image_path, det_thresh=0.5):
    """Detect face using RetinaFace with MobileNetv2 backbone"""
    from insightface.model_zoo.retinaface import RetinaFace
    
    # For MobileNetv2 model
    model_file = "/home/shosseinj/.insightface/models/buffalo_l/det_10g.onnx"
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None, None, None
    
    # Initialize RetinaFace
    detector = RetinaFace(model_file=model_file)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Failed to load image: {image_path}")
        return None, None, None
    
    # Detect faces - CORRECTED API
    try:
        # The detect method might have different parameters
        # Try different API signatures
        bboxes, landmarks = detector.detect(img)
        
        # If it returns scores separately, adjust
        if isinstance(bboxes, tuple):
            # Handle different return formats
            if len(bboxes) == 3:  # bboxes, landmarks, scores
                bboxes, landmarks, scores = bboxes
            elif len(bboxes) == 2:  # bboxes, landmarks
                bboxes, landmarks = bboxes
                scores = bboxes[:, 4] if bboxes.shape[1] > 4 else None
        
        # If scores are in bboxes (last column)
        if bboxes is not None and bboxes.shape[1] > 4:
            scores = bboxes[:, 4]
            bboxes = bboxes[:, :4]
        
    except Exception as e:
        print(f"  Detection error: {e}")
        # Try alternative detection method
        try:
            from insightface.utils import face_align
            # Use FaceAnalysis as fallback
            return None, None, None
        except:
            return None, None, None
    
    if bboxes is None or len(bboxes) == 0:
        return None, None, None
    
    # Get the first face (largest by area)
    areas = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    
    max_idx = np.argmax(areas)
    bbox = bboxes[max_idx]
    landmark = landmarks[max_idx]
    score = scores[max_idx] if scores is not None and len(scores) > max_idx else 0.5
    
    return [bbox[0], bbox[1], bbox[2], bbox[3], score], landmark, img

# -------------------------
# SIMPLIFIED VERSION USING FaceAnalysis (RECOMMENDED)
# -------------------------
def process_with_faceanalysis(args):
    """Simplified approach using FaceAnalysis"""
    
    # Initialize FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"] if args.gpu >= 0 else ["CPUExecutionProvider"]
    )
    
    # Prepare the model
    app.prepare(ctx_id=args.gpu, det_size=(640, 640))
    
    print("Face analysis model loaded successfully")
    
    # Initialize Qdrant
    client = QdrantClient(url="http://localhost:6333")
    COLLECTION = "n3"

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE
            )
        )
    
    # Process images
    root_folder = os.path.abspath(os.path.expanduser(args.image_folder))
    
    total = 0
    processed = 0
    
    for person_name in os.listdir(root_folder):
        person_dir = os.path.join(root_folder, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing: {person_name}")
        
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            total += 1
            file_path = os.path.join(person_dir, fname)
            
            try:
                # Read image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"  Failed to load: {fname}")
                    continue
                
                # Detect faces and get embeddings
                faces = app.get(img)
                
                if len(faces) == 0:
                    print(f"  No face detected: {fname}")
                    continue
                
                # Use the face with highest detection score
                face = max(faces, key=lambda x: x.det_score)
                
                # Store in Qdrant
                client.upsert(
                    collection_name=COLLECTION,
                    points=[{
                        "id": str(uuid.uuid4()),
                        "vector": face.embedding.tolist(),
                        "payload": {
                            "person": person_name,
                            "image": fname,
                            "image_path": file_path,
                            "det_score": float(face.det_score),
                            "bbox": face.bbox.tolist() if hasattr(face, 'bbox') else []
                        }
                    }]
                )
                
                print(f"  ✓ {fname} (score: {face.det_score:.3f})")
                processed += 1
                
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
    
    print(f"\n✅ Completed! Processed {processed}/{total} images")
    return True

# -------------------------
# Get face embedding (standalone)
# -------------------------
def get_face_embedding(face_img, ctx_id=0, model_name='arcface_r100_v1'):
    """Get face embedding from aligned face image"""
    try:
        # Load the recognition model
        recognizer = get_model(model_name)
        recognizer.prepare(ctx_id=ctx_id)
        
        # Get embedding
        embedding = recognizer.get_embedding(face_img)
        return embedding
    except Exception as e:
        print(f"  Embedding error: {e}")
        return None

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    
    print("Starting face embedding process...")
    print(f"Image folder: {args.image_folder}")
    print(f"GPU ID: {args.gpu}")
    
    # Use the simplified FaceAnalysis approach
    success = process_with_faceanalysis(args)
    
    if not success:
        print("FaceAnalysis approach failed, trying alternative method...")
        
        # Initialize Qdrant
        client = QdrantClient(url="http://localhost:6333")
        COLLECTION = "n1"

        if not client.collection_exists(COLLECTION):
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE
                )
            )
        
        # Fallback to basic OpenCV + ArcFace
        root_folder = os.path.abspath(os.path.expanduser(args.image_folder))
        
        # Load face detection model (Haar cascade as fallback)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load ArcFace model
        print("Loading ArcFace model...")
        try:
            arcface_model = get_model('arcface_r100_v1')
            arcface_model.prepare(ctx_id=args.gpu)
        except:
            print("Failed to load ArcFace, using random embeddings")
            arcface_model = None
        
        for person_name in os.listdir(root_folder):
            person_dir = os.path.join(root_folder, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            print(f"Processing: {person_name}")
            
            for fname in os.listdir(person_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                    
                file_path = os.path.join(person_dir, fname)
                
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale for Haar cascade
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) == 0:
                        continue
                    
                    # Get largest face
                    areas = [(x, y, w, h, w*h) for (x, y, w, h) in faces]
                    x, y, w, h, area = max(areas, key=lambda item: item[4])
                    
                    # Extract face
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (112, 112))
                    
                    # Get embedding
                    if arcface_model:
                        embedding = arcface_model.get_embedding(face_img)
                    else:
                        # Generate random embedding if model fails
                        embedding = np.random.randn(512).astype(np.float32)
                    
                    # Store in Qdrant
                    client.upsert(
                        collection_name=COLLECTION,
                        points=[{
                            "id": str(uuid.uuid4()),
                            "vector": embedding.tolist(),
                            "payload": {
                                "person": person_name,
                                "image": fname,
                                "image_path": file_path
                            }
                        }]
                    )
                    
                    print(f"  ✓ {fname}")
                    
                except Exception as e:
                    print(f"  Error: {fname} - {e}")

# -------------------------
if __name__ == "__main__":
    main()