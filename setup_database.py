#!/usr/bin/env python
"""
Database Setup Helper for Face Recognition

This script helps organize and prepare database images for face recognition.

Supports two structures:
1. Flat: database/ali.jpg, database/hossein.jpg
2. Nested: database/ali/photo1.jpg, database/ali/photo2.jpg

Usage:
    python setup_database.py --help
    python setup_database.py --check database
    python setup_database.py --organize-flat-to-nested database
    python setup_database.py --create-structure database
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


def check_database(db_dir):
    """Check and report database structure and contents."""
    if not os.path.isdir(db_dir):
        print(f"Error: Database directory '{db_dir}' not found")
        return False
    
    print(f"\n=== Database Check for '{db_dir}' ===\n")
    
    # List all items
    items = os.listdir(db_dir)
    subdirs = [d for d in items if os.path.isdir(os.path.join(db_dir, d))]
    files = [f for f in items if os.path.isfile(os.path.join(db_dir, f))]
    
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in files if f.lower().endswith(supported)]
    
    print(f"Subdirectories: {len(subdirs)}")
    print(f"Image files: {len(image_files)}")
    print(f"Other files: {len(files) - len(image_files)}\n")
    
    # Determine structure
    if len(subdirs) > 0 and len(image_files) == 0:
        print("Structure: NESTED (person folders with images inside)")
        print("\nDirectory structure:")
        for person_dir in sorted(subdirs):
            person_path = os.path.join(db_dir, person_dir)
            person_images = [f for f in os.listdir(person_path) 
                           if f.lower().endswith(supported)]
            print(f"  {person_dir}/: {len(person_images)} image(s)")
            for img in sorted(person_images):
                img_path = os.path.join(person_path, img)
                size = os.path.getsize(img_path) / 1024  # KB
                print(f"    - {img} ({size:.1f} KB)")
    else:
        print("Structure: FLAT (images directly in database directory)")
        print("\nImage files:")
        for img in sorted(image_files):
            img_path = os.path.join(db_dir, img)
            size = os.path.getsize(img_path) / 1024  # KB
            name = os.path.splitext(img)[0]
            print(f"  {img} ({size:.1f} KB) -> Person: '{name}'")
    
    return True


def create_nested_structure(db_dir):
    """Create a nested directory structure with examples."""
    print(f"\nCreating nested directory structure in '{db_dir}'...\n")
    
    if not os.path.isdir(db_dir):
        os.makedirs(db_dir)
        print(f"Created directory: {db_dir}")
    
    # Create example person directories
    example_people = ['ali', 'hossein', 'pouria', 'reza']
    
    for person in example_people:
        person_dir = os.path.join(db_dir, person)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"Created: {person_dir}/")
        else:
            print(f"Exists: {person_dir}/")
    
    print(f"\nDirectory structure created!")
    print("\nHow to use:")
    print(f"1. Place images for each person in their folder:")
    print(f"   database/ali/photo1.jpg")
    print(f"   database/ali/photo2.jpg")
    print(f"   database/ali/photo3.jpg")
    print(f"\n2. Then run the recognition:")
    print(f"   python webcam_inference.py --source 0 --recognition --db-dir database")
    print(f"\n3. Run with debug to see embeddings:")
    print(f"   python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug")
    
    return True


def convert_flat_to_nested(db_dir):
    """Convert flat structure to nested structure."""
    print(f"\nConverting flat structure to nested in '{db_dir}'...\n")
    
    if not os.path.isdir(db_dir):
        print(f"Error: Database directory '{db_dir}' not found")
        return False
    
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(db_dir) 
                  if os.path.isfile(os.path.join(db_dir, f)) and f.lower().endswith(supported)]
    
    if not image_files:
        print("No image files found in flat structure")
        return False
    
    # Create a backup
    backup_dir = f"{db_dir}_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(db_dir, backup_dir)
        print(f"Backup created: {backup_dir}\n")
    
    # Convert each image to its own folder
    for img in image_files:
        person_name = os.path.splitext(img)[0]
        person_dir = os.path.join(db_dir, person_name)
        
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"Created: {person_dir}/")
        
        src = os.path.join(db_dir, img)
        dst = os.path.join(person_dir, img)
        
        if os.path.exists(dst):
            print(f"  {img} already exists in {person_name}/ (skipped)")
        else:
            shutil.move(src, dst)
            print(f"  Moved: {img} -> {person_name}/{img}")
    
    print(f"\nConversion complete!")
    print(f"Backup saved to: {backup_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Database setup helper for face recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_database.py --check database
  python setup_database.py --create-structure database
  python setup_database.py --organize-flat-to-nested database
        """
    )
    
    parser.add_argument('database_dir', nargs='?', default='database',
                       help='Database directory path (default: database)')
    
    parser.add_argument('--check', action='store_true',
                       help='Check and report database structure')
    
    parser.add_argument('--create-structure', action='store_true',
                       help='Create nested directory structure with example folders')
    
    parser.add_argument('--organize-flat-to-nested', action='store_true',
                       help='Convert flat structure to nested structure')
    
    args = parser.parse_args()
    db_dir = args.database_dir
    
    # Default action: check database
    if not (args.check or args.create_structure or args.organize_flat_to_nested):
        args.check = True
    
    if args.check:
        check_database(db_dir)
    
    if args.create_structure:
        create_nested_structure(db_dir)
    
    if args.organize_flat_to_nested:
        convert_flat_to_nested(db_dir)


if __name__ == '__main__':
    main()
