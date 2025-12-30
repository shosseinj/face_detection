





import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
from PIL import Image
import os

# --- Load recognizer ---
device = 'cuda'
recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Prepare storage ---
embeddings_dict = {}

# --- Load face detector (Haar cascade example, you can replace with RetinaFace) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Webcam capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    key = -1  # Initialize key for this frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        # Convert to tensor
        face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).unsqueeze(0).float().to(device)

        # Compute embedding
        with torch.no_grad():
            embedding = recognizer(face_tensor)

        # Ask user to enter name
        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1) & 0xFF  # <- Moved here

        if key == ord('n'):  # Press 'n' to name this person
            person_name = input("Enter name of the person: ").strip()
            if person_name in embeddings_dict:
                embeddings_dict[person_name].append(embedding)
            else:
                embeddings_dict[person_name] = [embedding]
            print(f"Captured embedding for {person_name}")

    # Display frame even if no faces
    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF  # Ensure key is read even if no faces

    if key == ord('q'):
        break

# --- Release webcam ---
cap.release()
cv2.destroyAllWindows()

# --- Average embeddings per person and save ---
final_embeddings = {}
for name, embs in embeddings_dict.items():
    final_embeddings[name] = torch.stack(embs).mean(0).cpu()

torch.save(final_embeddings, "personnel_embeddings_webcam.pt")
print("Saved webcam embeddings to personnel_embeddings_webcam.pt")
