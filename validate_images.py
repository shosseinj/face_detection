#!/usr/bin/env python
"""
Image Validator for Face Recognition Database

Validates images before adding to database:
- Checks file format
- Verifies image size
- Detects faces in image
- Reports image quality
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def validate_image(image_path):
    """Validate a single image for face recognition database."""
    if not os.path.exists(image_path):
        return False, f"File not found: {image_path}"
    
    # Check file format
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    if not image_path.lower().endswith(supported):
        return False, f"Unsupported format: {os.path.splitext(image_path)[1]}"
    
    # Load image
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Cannot load image (corrupted or unsupported)"
    except Exception as e:
        return False, f"Error loading image: {e}"
    
    h, w = img.shape[:2]
    
    # Check dimensions
    if h < 100 or w < 100:
        return False, f"Image too small: {w}x{h} (minimum 100x100)"
    
    if h < 200 or w < 200:
        return "warning", f"Image small: {w}x{h} (recommended 200x200+)"
    
    # Try to detect face using OpenCV
    try:
        # Use haarcascade for quick face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return "warning", "No face detected (may still work with DeepFace)"
        elif len(faces) > 1:
            return "warning", f"Multiple faces detected ({len(faces)}), recommend single face"
        elif len(faces) == 1:
            x, y, fw, fh = faces[0]
            face_ratio = (fw * fh) / (h * w)
            if face_ratio < 0.1:
                return "warning", f"Face too small in image (only {face_ratio*100:.1f}% of image)"
            elif face_ratio > 0.6:
                return "ok", f"Good: {w}x{h}, face detected and centered"
            else:
                return "ok", f"Good: {w}x{h}, face detected"
    except Exception as e:
        return "warning", f"Face detection failed: {e}"
    
    return "ok", f"Valid: {w}x{h}"


def validate_directory(db_dir):
    """Validate all images in database directory."""
    print(f"\n=== Validating Database Images ===\n")
    print(f"Database: {db_dir}\n")
    
    if not os.path.isdir(db_dir):
        print(f"Error: Directory not found: {db_dir}")
        return
    
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    subdirs = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    
    if subdirs:
        # Nested structure
        total_valid = 0
        total_warning = 0
        total_error = 0
        
        for person_dir in sorted(subdirs):
            person_path = os.path.join(db_dir, person_dir)
            print(f"\n{person_dir}/")
            print("-" * 50)
            
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(supported)]
            
            if not images:
                print("  WARNING: No images found")
                continue
            
            for img_file in sorted(images):
                img_path = os.path.join(person_path, img_file)
                status, message = validate_image(img_path)
                
                if status == "ok":
                    total_valid += 1
                    print(f"  ✓ {img_file}: {message}")
                elif status == "warning":
                    total_warning += 1
                    print(f"  ~ {img_file}: {message}")
                else:
                    total_error += 1
                    print(f"  ✗ {img_file}: {message}")
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Summary:")
        print(f"  Valid: {total_valid}")
        print(f"  Warnings: {total_warning}")
        print(f"  Errors: {total_error}")
        
    else:
        # Flat structure
        images = [f for f in os.listdir(db_dir) 
                 if os.path.isfile(os.path.join(db_dir, f)) and f.lower().endswith(supported)]
        
        total_valid = 0
        total_warning = 0
        total_error = 0
        
        print("Flat Structure - Images in root:")
        print("-" * 50)
        
        for img_file in sorted(images):
            img_path = os.path.join(db_dir, img_file)
            status, message = validate_image(img_path)
            
            if status == "ok":
                total_valid += 1
                print(f"  ✓ {img_file}: {message}")
            elif status == "warning":
                total_warning += 1
                print(f"  ~ {img_file}: {message}")
            else:
                total_error += 1
                print(f"  ✗ {img_file}: {message}")
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Summary:")
        print(f"  Valid: {total_valid}")
        print(f"  Warnings: {total_warning}")
        print(f"  Errors: {total_error}")


def validate_single_image(image_path):
    """Validate and display info for a single image."""
    print(f"\n=== Image Validation ===\n")
    print(f"File: {image_path}\n")
    
    status, message = validate_image(image_path)
    
    print(f"Status: {status.upper()}")
    print(f"Message: {message}\n")
    
    # Load and display info
    try:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        size_kb = os.path.getsize(image_path) / 1024
        print(f"Dimensions: {w}x{h}")
        print(f"File size: {size_kb:.1f} KB")
    except Exception as e:
        print(f"Could not load image info: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate images for face recognition database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_images.py database
  python validate_images.py database/ali/photo1.jpg
        """
    )
    
    parser.add_argument('path', nargs='?', default='database',
                       help='Image file or database directory')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        validate_single_image(args.path)
    else:
        validate_directory(args.path)


if __name__ == '__main__':
    main()
