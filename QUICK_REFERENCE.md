# Quick Reference Card

## 🎯 Start Recognition

```bash
python webcam_inference.py --source 0 --recognition --db-dir database
```

## 📁 Database Structure

```
database/
├── ali/              (Add photos here)
│   └── ali.jpg
├── hossein/          (Already has 9 photos)
│   ├── photo1.jpg
│   └── ... (8 more)
├── pouria/           (Add photos here)
│   └── pouria.jpg
└── reza/             (Add photos here)
    └── reza.jpg
```

## ➕ Add More Images

1. Copy photo to: `database/ali/new_photo.jpg`
2. Delete: `embeddings.pkl`
3. Run recognition (rebuilds automatically)

## 🔍 Check Database

```bash
python setup_database.py --check database
```

## ✓ Validate Images

```bash
python validate_images.py database
```

## 🐛 Debug Recognition

```bash
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug
```

## ⚙️ Fine-Tune Recognition

```bash
# Stricter (fewer false matches)
--rec-threshold 0.45

# More lenient (fewer "Unknown")
--rec-threshold 0.65

# Faster (every 3rd frame)
--rec-skip-frames 3

# GPU acceleration
--rec-gpu
```

## 📊 Current Status

- **Total People**: 4
- **Total Images**: 13
- **Best Coverage**: Hossein (9 images)
- **Status**: ✅ Ready

## 📈 Improvement Plan

1. Ali: Add 4-5 photos (different angles/lighting)
2. Pouria: Add 4-5 photos
3. Reza: Add 4-5 photos
4. Rebuild embeddings
5. Test and adjust threshold if needed

## 🚀 Full Command (Recommended)

```bash
python webcam_inference.py --source 0 \
  --recognition \
  --db-dir database \
  --rec-skip-frames 2 \
  --rec-threshold 0.55 \
  --target-size 320
```

## 💾 Rebuild Embeddings Only

```bash
del embeddings.pkl
python -c "from utils.recognition_deepface import build_db, save_db_cache; db, m = build_db('database'); save_db_cache(db)"
```

## 📚 Documentation

- `README_MULTIIMAGE.md` - Full overview
- `QUICKSTART.md` - Quick start
- `MULTIPLE_IMAGES_SETUP.md` - Detailed guide
- `RECOGNITION_SETUP.md` - Troubleshooting
- `CODE_IMPLEMENTATION.md` - Technical details

## 🎓 How It Works

```
Live Face
    ↓
Extract Embedding (128 numbers)
    ↓
Compare against database embeddings
    ↓
Find minimum distance for each person
    ↓
Person with lowest distance = Match
    ↓
Display name (if distance < threshold)
```

## ⚡ Tips

- Look at camera (frontal face best)
- Good lighting helps
- More images = better accuracy
- Different angles = robust recognition
- Hossein has 9 images - that's why recognition works best!

## 🔧 Tools

| Tool     | Command                                                                 |
| -------- | ----------------------------------------------------------------------- |
| Check DB | `python setup_database.py --check database`                             |
| Validate | `python validate_images.py database`                                    |
| Run      | `python webcam_inference.py --source 0 --recognition --db-dir database` |
| Debug    | Add `--rec-debug` to see distances                                      |

---

**Everything set up and ready to use!** 🎉

Just add more images to each person's folder for better accuracy.
