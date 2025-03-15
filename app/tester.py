import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import json

import os

# Assuming the model and preprocessing are already defined
# Load the model (use the same code from previous)


def preprocess_new_data(data):
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    # Preprocess the input data similarly to how the model was trained
    # data is a dictionary or pandas Series for a new individual

    # Example: Convert categorical variables to numerical values using label encoder
    data["Sex"] = label_encoder.fit_transform([data["Sex"]])
    data["Diet"] = label_encoder.fit_transform([data["Diet"]])
    data["Country"] = label_encoder.fit_transform([data["Country"]])
    data["Continent"] = label_encoder.fit_transform([data["Continent"]])
    data["Hemisphere"] = label_encoder.fit_transform([data["Hemisphere"]])

    # Split 'Blood Pressure' into systolic and diastolic
    systolic, diastolic = map(int, data["Blood Pressure"].split("/"))
    data["Systolic BP"] = systolic
    data["Diastolic BP"] = diastolic

    data.pop("Blood Pressure", None)
    # data.drop("Blood Pressure", axis=1, inplace=True)

    # Convert the data to a DataFrame for easier processing
    df = pd.DataFrame([data])

    # Scale the features using the same scaler as during training
    df_scaled = scaler.fit_transform(df)

    return torch.tensor(df_scaled, dtype=torch.float32)


def predict_heart_attack_risk(new_data, model):
    # Preprocess the new data to match the format of the training data
    input_tensor = preprocess_new_data(new_data)
    # Pass the data through the model to get the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)

    # Convert the output probability to binary (0 or 1)
    prediction = (
        output > 0.5
    ).float()  # If the probability > 0.5, classify as 1, else 0

    return int(prediction.item())  # Return 0 or 1 as the final prediction


# Example usage:

# Load label encoders and scaler (these were used during training)
# Assuming you saved them as 'scaler.pkl' and 'label_encoder.pkl'

# Define the new person's data
new_person_data = {
    "Age": 67,
    "Sex": "Male",
    "Cholesterol": 208,
    "Blood Pressure": "158/88",
    "Heart Rate": 72,
    "Diabetes": 0,
    "Family History": 0,
    "Smoking": 1,
    "Obesity": 0,
    "Alcohol Consumption": 0,
    "Exercise Hours Per Week": 4.168189,
    "Diet": "Average",
    "Previous Heart Problems": 0,
    "Medication Use": 0,
    "Stress Level": 9,
    "Sedentary Hours Per Day": 6.615001,
    "Income": 261404,
    "BMI": 31.251233,
    "Triglycerides": 286,
    "Physical Activity Days Per Week": 0,
    "Sleep Hours Per Day": 6,
    "Country": "Argentina",
    "Continent": "South America",
    "Hemisphere": "Southern Hemisphere",
}


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


# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

json_file_path = os.path.join(current_directory, "testData", "testData.json")
with open(json_file_path, "r") as file:
    test_cases = json.load(file)


model = HeartDiseaseModel(input_size=25)  # Assuming 25 input features
model_path = os.path.join(
    current_directory, "models", "heart_disease_model(v1.0.0).pth"
)


model.load_state_dict(torch.load(model_path))  # Load weights into model
model.eval()  # Load the trained model


def test_model():
    for case_name, test_data in test_cases.items():
        prediction = predict_heart_attack_risk(test_data, model)
        print(f"{case_name}: Predicted Risk = {prediction:.2f}")


test_model()
