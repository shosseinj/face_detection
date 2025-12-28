# import torch
# import numpy as np
# from facenet_pytorch import InceptionResnetV1
# from PIL import Image
# import os

# def open_capture(source, max_index_search=5, backend_pref='any'):
#         """Open a VideoCapture robustly.

#         If `source` is a digit (webcam index), try multiple Windows backends
#         and search other indices as a fallback. For file paths, try opening
#         directly.
#         """
#         # Map backend constants to friendly names for logging
#         backend_names = {
#             cv2.CAP_DSHOW: 'CAP_DSHOW',
#             cv2.CAP_MSMF: 'CAP_MSMF'
#         }

#         # Translate backend_pref token to a prioritized list of backends
#         if backend_pref == 'dshow':
#             backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
#         elif backend_pref == 'msmf':
#             backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]
#         else:
#             backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

#         if str(source).isdigit():
#             idx = int(source)

#             # Try the requested index with multiple backends. Only return a
#             # capture if we can actually read at least one frame (warm-up).
#             for backend in backends:
#                 if backend is not None:
#                     cap = cv2.VideoCapture(idx, backend)
#                 else:
#                     cap = cv2.VideoCapture(idx)
#                 if cap.isOpened():
#                     name = backend_names.get(backend, 'default')
#                     # Try reading a frame to ensure backend actually returns data
#                     ret, frame = cap.read()
#                     if ret and frame is not None:
#                         print(f"Opened camera index {idx} with backend {name}")
#                         return cap, frame
#                     else:
#                         print(f"Opened camera index {idx} with backend {name} but could not read frame; trying next backend/index")
#                 cap.release()

#             # Try searching other indices (0..max_index_search-1)
#             for i in range(0, max_index_search):
#                 for backend in backends:
#                     if backend is not None:
#                         cap = cv2.VideoCapture(i, backend)
#                     else:
#                         cap = cv2.VideoCapture(i)
#                     if cap.isOpened():
#                         name = backend_names.get(backend, 'default')
#                         ret, frame = cap.read()
#                         if ret and frame is not None:
#                             print(f"Opened camera index {i} with backend {name}")
#                             return cap, frame
#                         else:
#                             print(f"Opened camera index {i} with backend {name} but could not read frame; trying next")
#                     cap.release()
#             return None, None
#         else:
#             # Treat as file path / stream URL
#             if not os.path.exists(source):
#                 print(f"Error: video file '{source}' not found.")
#                 return None, None
#             cap = cv2.VideoCapture(source)
#             if cap.isOpened():
#                 # Try reading a frame to ensure file/stream is ok
#                 ret, frame = cap.read()
#                 if ret and frame is not None:
#                     return cap, frame
#                 cap.release()
#             return None, None


#     # Low-latency background frame grabber (keeps only the latest frame)

# # Load face recognition model (pretrained)
# device = 'cuda'
# recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # Database folder: each subfolder = one employee
# database_dir = "./database/"
# embeddings_dict = {}

# for person_name in os.listdir(database_dir):
#     person_folder = os.path.join(database_dir, person_name)
#     if not os.path.isdir(person_folder):
#         continue

#     embeddings_list = []

#     for file in os.listdir(person_folder):
#         if not file.endswith(".jpg") and not file.endswith(".png"):
#             continue

#         img_path = os.path.join(person_folder, file)
#         img = Image.open(img_path).convert('RGB')
#         img = img.resize((160, 160))  # Resize to model input

#         # Convert to tensor and send to GPU
#         img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device)

#         # Compute embedding
#         with torch.no_grad():
#             embedding = recognizer(img_tensor)
#             embeddings_list.append(embedding)

#     if embeddings_list:
#         # Average embeddings for this person
#         embeddings_dict[person_name] = torch.stack(embeddings_list).mean(0)

# # Save embeddings to disk (CPU tensors to make it portable)
# cpu_embeddings = {k: v.cpu() for k, v in embeddings_dict.items()}
# torch.save(cpu_embeddings, "personnel_embeddings.pt")
# print("Saved embeddings to personnel_embeddings.pt")
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
