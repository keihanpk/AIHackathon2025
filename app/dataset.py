import pandas as pd
import os


# Hardcoded dataset directory path
data_dir = os.path.expanduser(
    "~/.cache/kagglehub/datasets/alikalwar/heart-attack-risk-prediction-cleaned-dataset/versions/1/heart-attack-risk-prediction-dataset.csv"
)


class Dataset:
    def __init__(self, transform=None):
        self.data = pd.read_csv(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
