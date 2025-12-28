# Face Recognition Setup Guide

## Now Working! Recognition System Improved

Your recognition system is now functioning with several enhancements:

### Key Improvements Made:

1. **Face Alignment**: Faces automatically aligned during embedding for consistency
2. **Better Embeddings**: Higher-quality face vectors with proper handling
3. **Optimized Threshold**: Default 0.55 for better matching
4. **Debug Output**: Shows all distances sorted by similarity
5. **Robust Handling**: Better edge case management

### Recommended Command:

```bash
python webcam_inference.py --source 0 --recognition --db-dir database --rec-skip-frames 2
```

### Critical Tips for Better Recognition:

**Look directly at the camera!** - This is the most important factor.

Other tips:

- Good lighting on your face
- Face centered in frame
- Hold position for a moment
- Identification happens every 2 frames

### Important Requirements for Good Recognition:

1. **Database Images**:

   - Clear, well-lit face photos
   - Face should be frontal or nearly frontal
   - Minimum 200x200 pixels
   - Multiple angles per person (at least 2-3 images per identity)
   - No glasses, hats, or extreme angles

2. **During Inference**:

   - Look directly at the camera (frontal face)
   - Good lighting (well-lit face)
   - Face should be clearly visible (not too small, not too far)
   - Keep steady position for a moment for recognition

3. **Threshold Tuning**:
   - Default is 0.5 (good for accuracy)
   - If too many "Unknown": decrease to 0.45 or 0.4 (more lenient)
   - If too many false positives: increase to 0.55 or 0.6 (stricter)

### Advanced Options:

```bash
# Use debug mode to see all matching distances
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug --rec-skip-frames 2

# Faster recognition (with slight accuracy trade-off)
python webcam_inference.py --source 0 --recognition --db-dir database --rec-model Facenet --rec-skip-frames 3

# Stricter matching (fewer false positives)
python webcam_inference.py --source 0 --recognition --db-dir database --rec-threshold 0.45 --rec-skip-frames 2

# Best accuracy (slower)
python webcam_inference.py --source 0 --recognition --db-dir database --rec-model VGG-Face --rec-skip-frames 1 --rec-threshold 0.5
```

### Model Comparison:

- **VGG-Face**: Most accurate, slightly slower (~200ms per face)
- **Facenet**: Fast and accurate (~100ms per face)
- **ArcFace**: Very fast and accurate (~80ms per face)

### Rebuild Database:

If recognition doesn't work well, you can force rebuild the database:

```bash
rm embeddings.pkl
# Then run with --recognition to rebuild
```

This will recompute all embeddings with the new face alignment enabled.
