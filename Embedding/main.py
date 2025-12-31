import os
import cv2
import numpy as np
import faiss
import torch
import argparse
from insightface.app import FaceAnalysis

# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("Face Embedding & FAISS GPU Storage")
    parser.add_argument("--image-folder", type=str, default="./database/hossein/",
                        help="Folder with face images")
    parser.add_argument("--query-image", type=str, default=None,
                        help="Query image path for nearest neighbor search")
    parser.add_argument("--embedding-dim", type=int, default=512,
                        help="Buffalo_L embedding dimension")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for FAISS and Buffalo_L")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to return")
    return parser.parse_args()

# -------------------------
# Main function
# -------------------------
def main():
    args = parse_args()

    # -------------------------
    # Initialize Buffalo_L
    # -------------------------
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=args.gpu)

    # -------------------------
    # Initialize FAISS GPU index
    # -------------------------
    res = faiss.StandardGpuResources()  # GPU resources
    index_cpu = faiss.IndexFlatL2(args.embedding_dim)  # CPU index
    index = faiss.index_cpu_to_gpu(res, args.gpu, index_cpu)
    face_id_map = []

    # -------------------------
    # Process images
    # -------------------------
    for fname in os.listdir(args.image_folder):
        if fname.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(args.image_folder, fname)
            img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
            faces = app.get(img)
            if len(faces) == 0:
                continue

            emb = faces[0].embedding.astype("float32")
            index.add(np.expand_dims(emb, axis=0))
            face_id_map.append(fname)

    print(f"Total embeddings stored: {index.ntotal}")

    # -------------------------
    # Query nearest neighbors (optional)
    # -------------------------
    if args.query_image:
        query_img = cv2.imread(args.query_image)[:, :, ::-1]
        faces = app.get(query_img)
        if len(faces) > 0:
            q_emb = faces[0].embedding.astype("float32")
            D, I = index.search(np.expand_dims(q_emb, axis=0), k=args.top_k)
            print("Distances:", D)
            print("Nearest images:", [face_id_map[i] for i in I[0]])
        else:
            print("No face detected in query image.")

if __name__ == "__main__":
    main()