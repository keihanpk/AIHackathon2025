import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "alikalwar/heart-attack-risk-prediction-cleaned-dataset"
)

print("Path to dataset files:", path)
