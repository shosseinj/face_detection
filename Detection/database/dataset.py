import kagglehub

# Download latest version
path = kagglehub.dataset_download("stoicstatic/face-recognition-dataset")

print("Path to dataset files:", path)