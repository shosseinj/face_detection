import torch
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
from PIL import Image

# --- Load model ---
device = 'cuda'
recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Load stored embeddings ---
embeddings_dict = torch.load("personnel_embeddings_webcam.pt")  # name -> embedding tensor

# --- Face detector ---
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            face_embedding = recognizer(face_tensor)
            if face_embedding.dim() == 1:
                face_embedding = face_embedding.unsqueeze(0)
            face_embedding = F.normalize(face_embedding, dim=1)

       # Recognition
        name = "Unknown"
        max_sim = -1.0
        for person_name, person_emb in embeddings_dict.items():
            # Make sure person_emb is 2D
            if person_emb.dim() == 1:
                person_emb = person_emb.unsqueeze(0)  # (1, 512)
            elif person_emb.dim() == 3:
                person_emb = person_emb.squeeze(0)    # remove extra dimension if present

            person_emb = F.normalize(person_emb.to(device), dim=1)

            # Make sure face_embedding is 2D
            if face_embedding.dim() == 1:
                face_embedding = face_embedding.unsqueeze(0)
            face_embedding = F.normalize(face_embedding, dim=1)

            sim = F.cosine_similarity(face_embedding, person_emb, dim=1)  # shape: (1,)
            sim_val = sim.item()  # now safe

            if sim_val > 0.9 and sim_val > max_sim:
                max_sim = sim_val
                name = person_name


        # Draw name
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
