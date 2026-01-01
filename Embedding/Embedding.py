import os
import cv2
import uuid
import numpy as np
import argparse
from insightface.app import FaceAnalysis
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
# Main
# -------------------------
def main():
    args = parse_args()

    # -------------------------
    # InsightFace (GPU only)
    # -------------------------
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"]
    )
    app.prepare(ctx_id=args.gpu)

    # -------------------------
    # Qdrant
    # -------------------------
    client = QdrantClient(url="http://localhost:6333")

    COLLECTION = "face_embeddings"



    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE
            )
        )


    # -------------------------
    # Insert embeddings
    # -------------------------
    
    import os
    import cv2
    import uuid

    root_folder = os.path.abspath(os.path.expanduser(args.image_folder))

    for person_name in os.listdir(root_folder):
        person_dir = os.path.join(root_folder, person_name)

        # skip files, only folders
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing person: {person_name}")

        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = os.path.join(person_dir, fname)
            img = cv2.imread(file_path)

            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            emb = faces[0].embedding.astype("float32")

            client.upsert(
                collection_name=COLLECTION,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": emb.tolist(),
                    "payload": {
                        "person": person_name,   # folder name
                        "image": fname,
                        "image_path": file_path
                    }
                }]
            )

    print("✅ Embeddings stored in Qdrant")
    
# -------------------------
if __name__ == "__main__":
    main()