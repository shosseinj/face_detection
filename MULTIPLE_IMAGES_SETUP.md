# Multiple Images Per Person - Setup Guide

## Overview

The system now supports **multiple images per person** from different angles and lighting conditions. This significantly improves recognition accuracy!

## Database Structure

Your database is now organized with nested folders:

```
database/
├── ali/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── hossein/
│   ├── angle1.jpg
│   ├── angle2.jpg
│   └── angle3.jpg
├── pouria/
│   ├── frontal.jpg
│   ├── left_angle.jpg
│   └── right_angle.jpg
└── reza/
    ├── 1.jpg
    ├── 2.jpg
    └── 3.jpg
```

## How to Add More Images

### Step 1: Organize Your Images

For each person, create a folder with their name and add multiple photos:

```
database/
└── ali/                    <- Create this folder
    ├── ali.jpg            <- Existing photo
    ├── ali_angle2.jpg     <- Add more photos
    ├── ali_angle3.jpg
    └── ali_light2.jpg
```

### Step 2: Image Requirements

For best results, each person should have:

- **2-5 images minimum** per person (more is better)
- **Different angles**: frontal, left, right
- **Different lighting**: normal, bright, slightly dim
- **Different distances**: close-up, medium, slightly far
- **Clear face**: centered, good quality
- **Minimal accessories**: avoid hats, glasses if possible

### Step 3: Rebuild Embeddings

Delete the cache and rebuild:

```bash
del embeddings.pkl
python webcam_inference.py --source 0 --recognition --db-dir database
```

Or just rebuild embeddings without camera:

```bash
python -c "from utils.recognition_deepface import build_db, save_db_cache; db, m = build_db('database'); save_db_cache(db)"
```

### Step 4: Test Recognition

Run with debug mode to see how embeddings are being used:

```bash
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug
```

## How It Works

### Embedding Computation

For each person's folder:

1. Each image gets its own embedding (128-dimensional vector)
2. Embeddings are face-aligned for consistency
3. All embeddings are stored in memory and cached to disk

Example output when building DB:

```
Using nested folder structure (subfolders per person)

Processing 3 images for 'ali':
  ✓ photo1.jpg: embedding computed
  ✓ photo2.jpg: embedding computed
  ✓ photo3.jpg: embedding computed
  Total: 3 embeddings for 'ali'

Processing 2 images for 'hossein':
  ✓ angle1.jpg: embedding computed
  ✓ angle2.jpg: embedding computed
  Total: 2 embeddings for 'hossein'

=== Database Summary ===
Total identities: 4
Total embeddings: 7
  ali: 3 embedding(s)
  hossein: 2 embedding(s)
  pouria: 1 embedding(s)
  reza: 1 embedding(s)
```

### Recognition Matching

During recognition:

1. Captured face is converted to embedding
2. **Compare against ALL database embeddings** (not just one per person)
3. Find the **minimum distance** across all embeddings for each person
4. **Best match wins** if distance <= threshold

Example:

```
identify_face: [>] ali: 0.1834          <- Best match (lowest distance)
identify_face: [ ] hossein: 0.6234
identify_face: [ ] pouria: 0.7841
identify_face: [ ] reza: 0.8902
identify_face: best_match=ali dist=0.1834 threshold=0.55
```

With multiple embeddings per person:

```
ali embeddings: [0.1834, 0.2156, 0.2301]  -> Min distance: 0.1834
hossein embeddings: [0.6234, 0.6412]       -> Min distance: 0.6234
pouria embeddings: [0.7841]                -> Min distance: 0.7841
reza embeddings: [0.8902]                  -> Min distance: 0.8902
```

## Database Management

### Check Database Structure

```bash
python setup_database.py --check database
```

Output:

```
=== Database Check for 'database' ===

Subdirectories: 4
Image files: 0

Structure: NESTED (person folders with images inside)

Directory structure:
  ali/: 3 image(s)
    - photo1.jpg (61.6 KB)
    - photo2.jpg (65.2 KB)
    - photo3.jpg (58.9 KB)
  hossein/: 2 image(s)
    - angle1.jpg (76.4 KB)
    - angle2.jpg (72.1 KB)
  ...
```

### Add Images to Existing Person

Simply copy new images to their folder:

```
database/ali/
├── ali.jpg (existing)
├── ali_angle2.jpg (existing)
└── ali_new_photo.jpg (NEW - add this)
```

Then rebuild embeddings:

```bash
del embeddings.pkl
# Run recognition to rebuild
```

### Remove a Person

Delete their folder:

```bash
rmdir /s database\person_name
del embeddings.pkl
```

### Convert Between Structures

If you need to go back to flat structure (one image per person):

```bash
python setup_database.py --organize-nested-to-flat database
```

## Advanced Tips

### Image Quality Tips

- **Best angles**: Frontal (0°), Left (45°), Right (45°)
- **Best lighting**: Natural light, overhead light, side light
- **Best distances**: Head fills 50-80% of image
- **Resolution**: 200x200 pixels minimum, 400x400 recommended

### Training for Different Conditions

If recognition fails in certain lighting:

- Add images taken in that lighting
- Add images at different angles
- Add images at different distances

### Threshold Adjustment

With multiple embeddings, you can use lower thresholds:

```bash
# Strict matching
python webcam_inference.py --source 0 --recognition --db-dir database --rec-threshold 0.40

# Balanced (default)
python webcam_inference.py --source 0 --recognition --db-dir database --rec-threshold 0.55

# Lenient
python webcam_inference.py --source 0 --recognition --db-dir database --rec-threshold 0.65
```

## Troubleshooting

### "Unknown" for correct person

**Cause**: Your live face angle/lighting doesn't match database images

**Solution**:

- Add images from the angle you use
- Add images in similar lighting
- Lower threshold: `--rec-threshold 0.60`

### Misidentification (wrong name)

**Cause**: Person folders are too similar, or threshold too high

**Solution**:

- Add more diverse images per person
- Raise threshold: `--rec-threshold 0.45`
- Ensure all people look significantly different

### Slow embedding computation

**Cause**: Too many images in database

**Solution**:

- Use `--rec-skip-frames 3` to identify every 3rd frame
- Reduce image count per person (3-5 is optimal)
- Use smaller target size: `--target-size 224`

## Performance Notes

### Computation Time

- Building database: ~500ms per image on CPU, ~200ms on GPU
- Recognition: ~100ms per comparison (compares against all embeddings)
- Storage: ~1KB per embedding

Example with 4 people, 3 images each (12 embeddings):

- Build time: ~6 seconds
- Per-face recognition: ~200ms

### Memory Usage

- Each embedding: ~1KB
- 100 embeddings: ~100KB
- 1000 embeddings: ~1MB

## Next Steps

1. **Add more images** to each person's folder (2-3 minimum)
2. **Delete embeddings.pkl** to force rebuild
3. **Run recognition** to rebuild database with new embeddings
4. **Test** with debug mode to see matching distances
5. **Adjust threshold** if needed based on results

## Commands Reference

```bash
# Check database structure
python setup_database.py --check database

# Rebuild embeddings only (no camera needed)
python -c "from utils.recognition_deepface import build_db, save_db_cache; db, m = build_db('database'); save_db_cache(db)"

# Run recognition with debug output
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug --rec-skip-frames 2

# Run with custom threshold
python webcam_inference.py --source 0 --recognition --db-dir database --rec-threshold 0.50

# Run with different model
python webcam_inference.py --source 0 --recognition --db-dir database --rec-model ArcFace
```
