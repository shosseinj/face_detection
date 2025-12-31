import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
from PIL import Image

# --- Load FaceNet model ---
device = 'cuda'
recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Storage ---
embeddings_dict = {}

# --- Face detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Webcam capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Instructions:")
print(" - Place a single person's face in front of the camera.")
print(" - Press 'n' to save the embedding for the detected face.")
print(" - Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Only use the first detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n') and len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            embedding = recognizer(face_tensor)

        person_name = input("Enter name of the person: ").strip()
        if person_name in embeddings_dict:
            embeddings_dict[person_name].append(embedding)
        else:
            embeddings_dict[person_name] = [embedding]

        print(f"Captured embedding for {person_name}")
        print("Remove the person and show the next person to capture...")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Average embeddings per person and save
final_embeddings = {name: torch.stack(embs).mean(0).cpu() for name, embs in embeddings_dict.items()}
torch.save(final_embeddings, "personnel_embeddings_webcam.pt")
print("Saved webcam embeddings to personnel_embeddings_webcam.pt")
