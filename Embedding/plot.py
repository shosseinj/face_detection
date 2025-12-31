import cv2
import os

image_path = "/home/shosseinj/Music/retinaface-pytorch/database/hossein/hossein.jpg"

img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load image!")
    exit()

print(f"Success! Image shape: {img.shape}")

# Display with OpenCV
cv2.imshow('Query Image', img)
print("Press any key to close the window...")
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()