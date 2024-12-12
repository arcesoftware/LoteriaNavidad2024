import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Prepare the dataset
data = {
    "Year": list(range(1960, 2024)),
    "Favored_Number": [
        30, 45, 92, 62, 15, 28, 82, 16, 30, 3, 94, 54, 89, 33, 31, 19, 51, 52, 50, 74,
        44, 4, 96, 15, 17, 30, 40, 94, 59, 7, 37, 50, 25, 91, 75, 39, 8, 65, 72, 6, 9,
        83, 33, 20, 62, 3, 40, 61, 66, 90, 41, 25, 70, 67, 29, 63, 93, 6, 19, 15, 66,
        19, 0, 94
    ]
}

df = pd.DataFrame(data)

# Normalize the data
min_year = df["Year"].min()
max_year = df["Year"].max()
df["Year"] = (df["Year"] - min_year) / (max_year - min_year)

min_number = df["Favored_Number"].min()
max_number = df["Favored_Number"].max()
df["Favored_Number"] = (df["Favored_Number"] - min_number) / (max_number - min_number)

# Convert data to tensors
X = torch.tensor(df["Year"].values, dtype=torch.float32).unsqueeze(1)  # Features (Year)
y = torch.tensor(df["Favored_Number"].values, dtype=torch.float32).unsqueeze(1)  # Target (Favored Number)

# Define the neural network model
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = Predictor()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Predict the favored number for 2024
next_year = torch.tensor([(2024 - min_year) / (max_year - min_year)], dtype=torch.float32).unsqueeze(1)
model.eval()
with torch.no_grad():
    predicted_number = model(next_year).item()

# Denormalize the predicted number
predicted_favored_number = round(predicted_number * (max_number - min_number) + min_number)

# Print the predicted favored number
print(f"The predicted favored number for 2024 is: {predicted_favored_number}")
