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

# Define normalization and denormalization functions
def normalize(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# Normalize the data
min_year, max_year = df["Year"].min(), df["Year"].max()
df["Year"] = normalize(df["Year"], min_year, max_year)

min_number, max_number = df["Favored_Number"].min(), df["Favored_Number"].max()
df["Favored_Number"] = normalize(df["Favored_Number"], min_number, max_number)

# Convert data to tensors
X = torch.tensor(df["Year"].values, dtype=torch.float32).unsqueeze(1)  # Features (Year)
y = torch.tensor(df["Favored_Number"].values, dtype=torch.float32).unsqueeze(1)  # Target (Favored Number)

# Split data into training and validation sets
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define the neural network model
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = Predictor()

# Define loss function and optimizer
learning_rate = 0.01
num_epochs = 10000
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization

# Early stopping parameters
best_val_loss = float('inf')
patience, patience_counter = 50, 0

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    train_loss = criterion(predictions, y_train)
    train_loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Predict the favored number for 2024
next_year_normalized = normalize(2024, min_year, max_year)
next_year_tensor = torch.tensor([[next_year_normalized]], dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_number_normalized = model(next_year_tensor).item()

# Clamp the normalized prediction to ensure it falls between 0 and 1
predicted_number_normalized = torch.clamp(
    torch.tensor(predicted_number_normalized),
    min=0.0,
    max=1.0
).item()

# Denormalize the clamped prediction
predicted_favored_number = round(denormalize(predicted_number_normalized, min_number, max_number))

# Print the predicted favored number
print(f"The predicted favored number for 2024 is: {predicted_favored_number}")

