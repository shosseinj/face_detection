from deepface import DeepFace
import os
import numpy as np
from typing import Dict, List, Tuple
import pickle
import tempfile


def build_db(db_dir: str, model_name: str = 'Facenet', detector_backend: str = 'mtcnn', enforce_detection: bool = False):
    """Load images from db_dir and compute embeddings using DeepFace.
    
    Supports two folder structures:
    1. Flat: database/ali.jpg, database/hossein.jpg, etc. (one image per person)
    2. Nested: database/ali/photo1.jpg, database/ali/photo2.jpg, etc. (multiple per person)

    Returns a dict label -> list of embeddings (np.ndarray)
    """
    if not os.path.isdir(db_dir):
        raise FileNotFoundError(f"Database directory '{db_dir}' not found")

    # Some DeepFace versions accept a prebuilt model object, others do not.
    try:
        model = DeepFace.build_model(model_name)
    except Exception:
        model = None
    
    db = {}
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Check if using nested structure (subfolders per person)
    subdirs = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    use_nested = len(subdirs) > 0
    
    if use_nested:
        print(f"Using nested folder structure (subfolders per person)")
        # Process each person's folder
        for person_folder in sorted(subdirs):
            person_path = os.path.join(db_dir, person_folder)
            label = person_folder
            
            # Process all images in this person's folder
            images_for_person = []
            for fname in sorted(os.listdir(person_path)):
                if not fname.lower().endswith(supported):
                    continue
                image_path = os.path.join(person_path, fname)
                images_for_person.append((fname, image_path))
            
            if not images_for_person:
                print(f"Warning: No images found for '{label}' in {person_path}")
                continue
            
            print(f"\nProcessing {len(images_for_person)} images for '{label}':")
            
            # Compute embedding for each image
            for fname, image_path in images_for_person:
                try:
                    rep = DeepFace.represent(
                        img_path=image_path,
                        model_name=model_name,
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=True
                    )
                except Exception as e:
                    print(f"  Warning: Failed to process {fname}: {e}")
                    continue
                
                # Normalize representation
                emb = _normalize_embedding(rep, image_path)
                if emb is not None:
                    db.setdefault(label, []).append(emb)
                    print(f"  ✓ {fname}: embedding computed")
            
            if label in db:
                print(f"  Total: {len(db[label])} embeddings for '{label}'")
    else:
        print(f"Using flat structure (images in root directory)")
        # Original flat structure: images directly in db_dir
        for fname in sorted(os.listdir(db_dir)):
            if not fname.lower().endswith(supported):
                continue
            path = os.path.join(db_dir, fname)
            label = os.path.splitext(fname)[0]
            
            try:
                rep = DeepFace.represent(
                    img_path=path,
                    model_name=model_name,
                    detector_backend='opencv',
                    enforce_detection=False,
                    align=True
                )
            except Exception as e:
                print(f"Warning: DeepFace.represent failed for {path}: {e}")
                continue
            
            emb = _normalize_embedding(rep, path)
            if emb is not None:
                db.setdefault(label, []).append(emb)
                print(f"Loaded reference image for label='{label}' ({path})")
    
    # Summary
    total_embeddings = sum(len(v) for v in db.values())
    print(f"\n=== Database Summary ===")
    print(f"Total identities: {len(db)}")
    print(f"Total embeddings: {total_embeddings}")
    for label in sorted(db.keys()):
        print(f"  {label}: {len(db[label])} embedding(s)")
    
    return db, model


def _normalize_embedding(rep, image_path: str):
    """Helper function to normalize representation into a 1-D numpy array."""
    emb = None
    try:
        if isinstance(rep, dict):
            if 'embedding' in rep:
                emb = np.asarray(rep['embedding'], dtype=np.float32)
            else:
                for v in rep.values():
                    if isinstance(v, (list, np.ndarray)):
                        emb = np.asarray(v, dtype=np.float32)
                        break
        elif isinstance(rep, list):
            first = rep[0] if rep else None
            if isinstance(first, dict) and 'embedding' in first:
                emb = np.asarray(first['embedding'], dtype=np.float32)
            elif isinstance(first, (list, np.ndarray)):
                emb = np.asarray(first, dtype=np.float32)
            else:
                emb = np.asarray(rep, dtype=np.float32)
        elif isinstance(rep, (np.ndarray, list, tuple)):
            emb = np.asarray(rep, dtype=np.float32)
    except Exception as e:
        print(f"Warning: failed to normalize embedding for {image_path}: {e}")
        return None
    
    if emb is None:
        print(f"Warning: no embedding produced for {image_path}")
        return None
    
    try:
        emb = np.asarray(emb, dtype=np.float32)
    except Exception as e:
        print(f"Warning: could not convert embedding for {image_path} to array: {e}")
        return None
    
    return emb


def save_db_cache(db: Dict[str, List[np.ndarray]], cache_path: str = 'embeddings.pkl'):
    """Save precomputed embeddings to disk for faster loading on subsequent runs."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(db, f)
        print(f"Saved embedding cache to {cache_path}")
    except Exception as e:
        print(f"Warning: failed to save embedding cache: {e}")


def load_db_cache(cache_path: str = 'embeddings.pkl') -> Dict[str, List[np.ndarray]]:
    """Load precomputed embeddings from disk if available."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            db = pickle.load(f)
        print(f"Loaded embedding cache from {cache_path}")
        return db
    except Exception as e:
        print(f"Warning: failed to load embedding cache: {e}")
        return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0
    return 1.0 - (num / den)


def identify_face(face_img, db: Dict[str, List[np.ndarray]], model, model_name: str = 'Facenet', detector_backend: str = 'skip', enforce_detection: bool = False, threshold: float = 0.4, debug: bool = False) -> Tuple[str, float, float]:
    """Identify face_img (numpy BGR image) against db. Returns (label, distance, confidence).

    If no match below threshold, returns ('Unknown', best_distance, 0.0)
    Confidence = 1 - normalized_distance (0.0 = no confidence, 1.0 = perfect match)
    """
    import cv2
    import tempfile
    
    # Input validation
    if db is None or len(db) == 0:
        if debug:
            print("identify_face: DB is empty")
        return 'Unknown', 1.0, 0.0
    
    if face_img is None or face_img.size == 0:
        if debug:
            print("identify_face: Invalid face image")
        return 'Unknown', 1.0, 0.0
    
    # Check face image size
    h, w = face_img.shape[:2]
    if h < 32 or w < 32:
        if debug:
            print(f"identify_face: Face too small ({w}x{h}), skipping")
        return 'Unknown', 1.0, 0.0
    
    # Convert BGR -> RGB as DeepFace expects RGB images
    try:
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        if debug:
            print(f"identify_face: failed to convert BGR to RGB: {e}")
        return 'Unknown', 1.0, 0.0

    # Ensure small faces are upscaled so detector/represent works better
    try:
        h, w = rgb.shape[:2]
        min_side = 160
        if min(h, w) < min_side:
            scale = min_side / float(min(h, w))
            new_h = int(h * scale)
            new_w = int(w * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            if debug:
                print(f"identify_face: upscaled face from {w}x{h} to {new_w}x{new_h}")
    except Exception as e:
        if debug:
            print(f"identify_face: failed to upscale face: {e}")

    # Save face to temporary file (DeepFace.represent needs a file path, not numpy array)
    emb = None
    try:
        # Save the face image to a temporary file
        temp_file = None
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_file = tmp.name
            # Convert RGB back to BGR for OpenCV's imwrite
            cv2.imwrite(temp_file, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        if debug:
            print(f"identify_face: calling DeepFace.represent on face with model={model_name}")
        
        # Call represent - DeepFace will handle face alignment internally
        rep = DeepFace.represent(
            img_path=temp_file, 
            model_name=model_name, 
            detector_backend='opencv',  # Use opencv for detection within DeepFace
            enforce_detection=enforce_detection,
            align=True  # Enable face alignment for better accuracy
        )
        
        # Handle return formats - DeepFace usually returns list of dicts
        emb = None
        if isinstance(rep, list) and len(rep) > 0:
            if isinstance(rep[0], dict) and 'embedding' in rep[0]:
                emb = np.asarray(rep[0]['embedding'], dtype=np.float32)
            elif isinstance(rep[0], (list, np.ndarray)):
                emb = np.asarray(rep[0], dtype=np.float32)
        elif isinstance(rep, dict) and 'embedding' in rep:
            emb = np.asarray(rep['embedding'], dtype=np.float32)
        elif isinstance(rep, (list, np.ndarray)):
            emb = np.asarray(rep, dtype=np.float32)
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
        
        if emb is None:
            if debug:
                print(f"identify_face: could not extract embedding from representation: {type(rep)}")
            return 'Unknown', 1.0, 0.0
            
    except Exception as e:
        if debug:
            print(f"identify_face: DeepFace.represent failed: {e}")
        # Try without enforce_detection
        try:
            if debug:
                print(f"identify_face: retrying with enforce_detection=False")
            rep = DeepFace.represent(
                img_path=temp_file if temp_file else face_img,
                model_name=model_name,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            if isinstance(rep, list) and len(rep) > 0:
                if isinstance(rep[0], dict) and 'embedding' in rep[0]:
                    emb = np.asarray(rep[0]['embedding'], dtype=np.float32)
                elif isinstance(rep[0], (list, np.ndarray)):
                    emb = np.asarray(rep[0], dtype=np.float32)
            elif isinstance(rep, dict) and 'embedding' in rep:
                emb = np.asarray(rep['embedding'], dtype=np.float32)
            else:
                emb = np.asarray(rep, dtype=np.float32)
        except Exception as e2:
            if debug:
                print(f"identify_face: retry also failed: {e2}")
            return 'Unknown', 1.0, 0.0
        finally:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

    # Ensure embedding is a flat numpy array
    if emb is None:
        if debug:
            print(f"identify_face: no embedding extracted")
        return 'Unknown', 1.0, 0.0
    
    try:
        emb = np.asarray(emb, dtype=np.float32).flatten()
    except Exception as e:
        if debug:
            print(f"identify_face: failed to flatten embedding: {e}")
        return 'Unknown', 1.0, 0.0
    
    if emb.shape[0] == 0:
        if debug:
            print(f"identify_face: embedding is empty")
        return 'Unknown', 1.0, 0.0
    
    if debug:
        print(f"identify_face: embedding computed (len={emb.shape[0]})")

    # Compare to DB with improved matching algorithm
    # Problem: People with more images get easier matches
    # Solution: Use percentile-based scoring for fairness
    
    best_label = 'Unknown'
    best_score = float('inf')
    distances = {}  # Track distances: {label: (min_dist, mean_dist, num_images)}
    all_distances = {}  # Track all individual distances for analysis
    
    for label, embs in db.items():
        dists = []
        for db_emb in embs:
            d = cosine_distance(emb, db_emb)
            dists.append(d)
        
        min_dist = min(dists)
        mean_dist = np.mean(dists)
        
        # Use a weighted score that prefers minimum distance but penalizes mean
        # This prevents bias towards people with more images
        # Formula: min_distance + (mean_distance - min_distance) * 0.3
        # This keeps emphasis on best match but considers overall fit
        score = min_dist + (mean_dist - min_dist) * 0.2
        
        distances[label] = (min_dist, mean_dist, len(dists))
        all_distances[label] = dists
        
        if score < best_score:
            best_score = score
            best_dist = min_dist
            best_label = label
    
    if debug:
        # Print all distances with detailed breakdown
        print(f"identify_face: Distance Analysis:")
        for label, (min_d, mean_d, count) in sorted(distances.items(), key=lambda x: x[1][0]):
            status = ">" if label == best_label else " "
            print(f"identify_face: [{status}] {label}: min={min_d:.4f}, mean={mean_d:.4f}, images={count}")
        print(f"identify_face: best_match={best_label} dist={best_dist:.4f} score={best_score:.4f} threshold={threshold}")
    
    if best_dist <= threshold:
        # Confidence: 1 - distance (0.0 to 1.0 scale)
        # At threshold 0.50, distance 0.0 = 100% confidence, distance 0.50 = 0% confidence
        confidence = max(0.0, 1.0 - (best_dist / threshold))
        return best_label, float(best_dist), confidence
    # If not matched, return Unknown with 0 confidence
    return 'Unknown', float(best_dist), 0.0
