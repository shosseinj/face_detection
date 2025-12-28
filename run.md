```
python webcam_inference.py -n mobilenetv2 -w retinaface_mv2.pth --source 0 --low-latency
```

```
.venv\Scripts\python.exe -c "
from utils.recognition_deepface import build_db, save_db_cache
import os

db, model = build_db('database', model_name='Facenet', detector_backend='opencv', enforce_detection=False)
save_db_cache(db, 'embeddings.pkl')
"

```

```
.venv\Scripts\python.exe validate_images.py database
```

`python webcam_inference.py --source 0 --recognition --db-dir database`

```

python setup_database.py --check database
python validate_images.py database
```

```
# Run recognition
python webcam_inference.py --source 0 --recognition --db-dir database

# Check database
python setup_database.py --check database

# Validate images
python validate_images.py database

# Debug mode
python webcam_inference.py --source 0 --recognition --db-dir database --rec-debug

# Rebuild embeddings
del embeddings.pkl
python -c "from utils.recognition_deepface import build_db, save_db_cache; db, m = build_db('database'); save_db_cache(db)"

```
