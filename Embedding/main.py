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
    parser.add_argument("--image-folder", type=str, default="../database/hossein/")
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

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )

    # -------------------------
    # Insert embeddings
    # -------------------------
    for fname in os.listdir(args.image_folder):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        path = os.path.join(args.image_folder, fname)
        img = cv2.imread(path)
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
                    "person": os.path.basename(args.image_folder),
                    "image": fname
                }
            }]
        )

    print("Embeddings stored in Qdrant")

    # -------------------------
    # Query
    # -------------------------
    if args.query_image:
        img = cv2.imread(args.query_image)
        faces = app.get(img)

        if len(faces) == 0:
            print("No face in query image")
            return

        q_emb = faces[0].embedding.astype("float32")

        hits = client.search(
            collection_name=COLLECTION,
            query_vector=q_emb.tolist(),
            limit=args.top_k
        )

        print("\nTop matches:")
        for h in hits:
            similarity = 1 - h.score  # cosine distance → similarity
            print(f"{h.payload['image']} | similarity={similarity:.3f}")

# -------------------------
if __name__ == "__main__":
    main()