
import os 
import cv2


def crop_face(frame, box, padding=0.2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    face = frame[y1:y2, x1:x2]
    return face





# ===============================
# Create save directory
# ===============================
def create_save_directory(save_dir):
    """Create directory for saving faces if it doesn't exist"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    return save_dir


# ===============================
# Save detected face
# ===============================
def save_face_image(face_img, save_dir, timestamp):
    """Save a detected face image with informative filename"""
    filename = f"{timestamp}.jpg"    
    # Save the face image
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, face_img)
    
    return save_path


# ===============================
# Extract face from frame
# ===============================
def extract_face(frame, bbox, margin=0.2, min_size=50):
    """
    Extract face from frame with optional margin
    Args:
        frame: Original frame
        bbox: [x1, y1, x2, y2] bounding box coordinates
        margin: Percentage margin to add around face (0.2 = 20%)
        min_size: Minimum face size to extract
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Skip if face is too small
    if face_width < min_size or face_height < min_size:
        return None
    
    # Add margin
    if margin > 0:
        width_margin = int(face_width * margin)
        height_margin = int(face_height * margin)
        
        x1 = max(0, x1 - width_margin)
        y1 = max(0, y1 - height_margin)
        x2 = min(frame.shape[1], x2 + width_margin)
        y2 = min(frame.shape[0], y2 + height_margin)
    
    # Extract face region
    face = frame[y1:y2, x1:x2]
    
    # Ensure face is not empty
    if face.size == 0:
        return None
    
    return face



