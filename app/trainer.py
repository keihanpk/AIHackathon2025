import dataset as DS
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn as nn
import torch.optim as optim

print(len(DS.Dataset()))
print(DS.Dataset()[0])


# Check if CUDA (NVIDIA) or Metal (Apple M1/M2) GPU is available
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU (CUDA) for NVIDIA GPUs
        print("Using GPU (CUDA):", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Metal API for Apple Silicon GPUs
        print("Using GPU (Metal API): Apple M1/M2 GPU")
    else:
        device = torch.device("cpu")  # Use CPU if no GPU available
        print("No GPU available. Using CPU.")
    return device


# Example neural network model
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)


# Load and preprocess your dataset
def preprocess_data():
    df = DS.Dataset().data
    # df = pd.read_csv("heart_disease.csv")

    # Handle categorical columns: 'Sex', 'Diet', 'Country', 'Continent'
    label_encoder = LabelEncoder()

    # Encode categorical columns
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Diet"] = label_encoder.fit_transform(df["Diet"])
    df["Country"] = label_encoder.fit_transform(df["Country"])
    df["Continent"] = label_encoder.fit_transform(df["Continent"])
    df["Hemisphere"] = label_encoder.fit_transform(df["Hemisphere"])

    # Split 'Blood Pressure' into two columns: systolic and diastolic
    # Blood Pressure is in the format '158/88', so split by '/'
    df[["Systolic BP", "Diastolic BP"]] = df["Blood Pressure"].str.split(
        "/", expand=True
    )
    # Convert these columns to numeric type
    df["Systolic BP"] = pd.to_numeric(df["Systolic BP"], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(df["Diastolic BP"], errors="coerce")

    # Drop the original 'Blood Pressure' column
    df = df.drop("Blood Pressure", axis=1)

    # Split data into features and target variable
    X = df.drop(
        ["Heart Attack Risk", "Patient ID"], axis=1
    )  # Drop 'Patient ID' and 'Heart Attack Risk' columns
    y = df["Heart Attack Risk"]  # Target variable

    # Scale features (standardize)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]


# Training function
def train_model(device, train_loader, model, criterion, optimizer, num_epochs=10):
    model.to(device)  # Move model to the chosen device
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")


# Main execution
if __name__ == "__main__":
    device = get_device()  # Automatically determine whether to use GPU or CPU
    train_loader, test_loader, input_size = preprocess_data()

    model = HeartDiseaseModel(input_size=input_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(device, train_loader, model, criterion, optimizer, num_epochs=20)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = (
                outputs > 0.5
            ).float()  # Convert probabilities to binary predictions
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print("Current directory:", current_directory)
    torch.save(
        model.state_dict(),
        current_directory + "/models/heart_disease_model(v1.0.0).pth",
    )  # Save the model
