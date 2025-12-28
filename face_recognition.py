import torch
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
from PIL import Image

# --- Load recognizer ---
device = 'cuda'
recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Load precomputed embeddings ---
embeddings_dict = torch.load("personnel_embeddings_webcam.pt")  # name -> embedding tensor

# --- Initialize face detector (Haar cascade example) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        # Convert to tensor
        face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            face_embedding = recognizer(face_tensor)
            face_embedding = F.normalize(face_embedding, dim=1)

        # --- Recognition ---
        name = "Unknown"
        max_sim = -1.0
        for person_name, person_emb in embeddings_dict.items():
            person_emb = F.normalize(person_emb.unsqueeze(0).to(device), dim=1)
            sim = F.cosine_similarity(face_embedding, person_emb, dim=1)
            sim_val = sim.item()
            if sim_val > 0.6 and sim_val > max_sim:  # threshold 0.6
                max_sim = sim_val
                name = person_name

        # --- Draw name above face ---
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
