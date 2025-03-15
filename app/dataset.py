import pandas as pd
import os


# Hardcoded dataset directory path
data_dir = os.path.expanduser(
    "~/.cache/kagglehub/datasets/iamsouravbanerjee/heart-attack-prediction-dataset/versions/2/heart_attack_prediction_dataset.csv"
)


class Dataset:
    def __init__(self, transform=None):
        self.data = pd.read_csv(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
