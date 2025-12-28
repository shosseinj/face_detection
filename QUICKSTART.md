# Face Recognition System - Quick Start

## What's New

Your face recognition system now supports **multiple images per person** for better accuracy!

## Current Database Structure

```
database/
├── ali/               (1 image)
├── hossein/           (1 image)
├── pouria/            (1 image)
└── reza/              (1 image)
```

## Add More Images

### Easy Steps:

1. **Open Explorer**: Go to `database/ali/` folder
2. **Add images**: Copy or move photos of Ali to this folder
3. **Delete cache**: Remove `embeddings.pkl` file
4. **Run recognition**: Start webcam inference - embeddings will rebuild automatically

### Example:

```
database/ali/
├── ali.jpg           (original)
├── ali_photo2.jpg    (ADD THIS)
├── ali_photo3.jpg    (ADD THIS)
├── ali_angle_left.jpg (ADD THIS)
└── ali_angle_right.jpg (ADD THIS)
```

## How Many Images?

- **Minimum**: 1 image per person (already have this)
- **Good**: 2-3 images per person (different angles)
- **Better**: 4-5 images per person (different lighting + angles)
- **Optimal**: 5-10 images per person (covers all conditions)

## Best Image Types

For each person, try to get:

1. **Frontal** (looking at camera)
2. **Left angle** (45° left)
3. **Right angle** (45° right)
4. **Different lighting** (bright vs normal)
5. **Different distances** (close vs normal)

## Run Recognition

### Basic (with camera):

```bash
python webcam_inference.py --source 0 --recognition --db-dir database
```

### With Debug (see distances):

```bash
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug
```

### Rebuild Database Only (no camera):

```bash
python -c "from utils.recognition_deepface import build_db, save_db_cache; db, m = build_db('database'); save_db_cache(db)"
```

## Check Your Database

```bash
python setup_database.py --check database
```

Shows:

- How many people in database
- How many images per person
- Total embeddings stored

## How It Works

1. **Embedding Creation**: Each image becomes a 128-dimensional vector
2. **Storage**: All vectors saved to `embeddings.pkl` (fast loading)
3. **Comparison**: Live face compared against ALL embeddings
4. **Best Match**: Lowest distance wins (if below threshold)

Example comparison with 2 images per person:

```
Ali embeddings: [0.15, 0.18]      -> Min: 0.15 ✓ MATCH
Hossein: [0.62, 0.65]              -> Min: 0.62
Pouria: [0.78, 0.81]               -> Min: 0.78
Result: "Ali" (lowest distance)
```

## Important Notes

- Delete `embeddings.pkl` when adding new images
- First run will be slow (computing embeddings)
- Subsequent runs are fast (loaded from cache)
- Look at camera for best recognition (frontal face)
- Good lighting helps accuracy

## File Organization

```
I:\AI\retinaface-pytorch\
├── database/               <- Your images here
│   ├── ali/
│   ├── hossein/
│   ├── pouria/
│   └── reza/
├── webcam_inference.py     <- Main script
├── setup_database.py       <- Database management
├── embeddings.pkl          <- Cache (auto-created)
├── MULTIPLE_IMAGES_SETUP.md <- Detailed guide
└── ...
```

## Common Commands

```bash
# Check database
python setup_database.py --check database

# Run recognition
python webcam_inference.py --source 0 --recognition --db-dir database

# Run with debug (see all distances)
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug

# Rebuild embeddings
del embeddings.pkl
python webcam_inference.py --source 0 --recognition --db-dir database

# Check which files to add
ls database/ali/
```

## Next Steps

1. Add 2-3 more photos to each person's folder
2. Delete `embeddings.pkl`
3. Run: `python webcam_inference.py --source 0 --recognition --db-dir database`
4. Test recognition - should be more accurate!

For more details, see: `MULTIPLE_IMAGES_SETUP.md`
