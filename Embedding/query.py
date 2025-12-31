import argparse
import cv2
import numpy as np
import os
import sys
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS = 1
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)


def stack_images_horizontally(imgs, texts, target_h=200):
    """
    Resize all images to the same height, add text labels, and concatenate horizontally.
    """
    resized = []
    for img, label in zip(imgs, texts):
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        im = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # draw text background bar
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(im, (0, 0), (tw + 4, th + 6), TEXT_BG, -1)
        cv2.putText(im, label, (2, th + 2), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        resized.append(im)
    return cv2.hconcat(resized)


def display_with_cv2(query_img_path, matches, top_k=5):
    imgs, texts = [cv2.imread(query_img_path)], ["Query"]
    for idx, m in enumerate(matches[:top_k]):
        img_path = m["image_path"]
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                label = f"Match{idx+1} {m['similarity']:.3f}"
                imgs.append(img)
                texts.append(label)
        else:
            print(f"Warning: not found {img_path}")
    if len(imgs) == 1:
        print("No valid matches to display")
        return
    canvas = stack_images_horizontally(imgs, texts)
    cv2.imshow("Face search results", canvas)
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_collage(query_img_path, matches, save_path, top_k=5):
    imgs, texts = [cv2.imread(query_img_path)], ["Query"]
    for idx, m in enumerate(matches[:top_k]):
        img_path = m["image_path"]
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                label = f"Match{idx+1} {m['similarity']:.3f}"
                imgs.append(img)
                texts.append(label)
    if len(imgs) == 1:
        print("Nothing to save")
        return
    canvas = stack_images_horizontally(imgs, texts)
    cv2.imwrite(save_path, canvas)
    print(f"Collage saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_image", type=str, default="./database/pouria/p.jpeg")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--display", action="store_true", help="Show result window")
    parser.add_argument("--save", type=str, default="", help="Save collage to this file")
    args = parser.parse_args()

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    client = QdrantClient(host="localhost", port=6333)

    img = cv2.imread(args.query_image)
    if img is None:
        print("Cannot read query image")
        return

    faces = app.get(img)
    if not faces:
        print("No face detected in query image")
        return

    q_emb = faces[0].embedding.astype("float32")
    collection_name = "face_embeddings"

    result = client.query_points(
        collection_name=collection_name,
        query=q_emb.tolist(),
        limit=args.top_k
    )

    matches = []
    print("\n" + "=" * 60)
    print(f"Query: {args.query_image}")
    print("=" * 60)
    for idx, pt in enumerate(result.points):
        sim = 1 - pt.score
        path = pt.payload.get("image_path", "Unknown")
        matches.append({"image_path": path, "similarity": sim, "id": pt.id})
        print(f"{idx+1}. {path}  (sim={sim:.3f})")
    print("-" * 60)

    if args.display or True:
        display_with_cv2(args.query_image, matches, args.top_k)

    if args.save:
        save_collage(args.query_image, matches, args.save, args.top_k)


if __name__ == "__main__":
    main()