# Lottery 2024 Predictor Project 

![2024]([https://github.com/arcesoftware/LoteriaNavidad2024/blob/main/loto.py](https://github.com/arcesoftware/LoteriaNavidad2024/blob/main/2024.webp))


This project implements a neural network model to predict a "Favored Number" for a given year based on historical data using PyTorch. The dataset is small but represents the only real data available for this phenomenon.

## Features
- **Data Normalization:** Converts input and target data into a normalized range for better training efficiency.
- **Model Architecture:** A simple neural network with:
  - Input layer
  - Two hidden layers with ReLU activations
  - Dropout regularization to prevent overfitting
  - Output layer for the prediction.
- **Training Optimization:**
  - Mean Squared Error (MSE) as the loss function
  - Adam optimizer with L2 regularization
- **Early Stopping:** Stops training when validation loss stagnates to prevent overfitting.
- **Validation Split:** Uses 80% of data for training and 20% for validation.

## Dataset
The dataset contains the years from 1960 to 2023 and their corresponding "Favored Numbers." The dataset is normalized for model input but retains its original scale for the final prediction.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/favored-number-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd favored-number-prediction
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the script:
    ```bash
    python predict_favored_number.py
    ```
2. The script trains the model and predicts the "Favored Number" for the year 2024. The result is displayed in the terminal.

## How It Works
1. **Data Preparation:** The historical data is normalized to a [0, 1] range using the minimum and maximum values of the dataset.
2. **Training:**
   - The dataset is split into training and validation sets.
   - The model is trained using MSE loss and Adam optimizer with early stopping to minimize overfitting.
3. **Prediction:**
   - The model predicts the normalized value for 2024.
   - The predicted value is clamped to ensure it falls within the valid range of [0, 1].
   - The normalized prediction is denormalized back to the original scale for display.

## Example Output

  ```bash
Epoch [50/1000], Train Loss: 0.0023, Val Loss: 0.0051 ... Early stopping at epoch 350 The predicted favored number for 2024 is: 66
 ```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions and improvements.

## License
This project is licensed under the MIT License.

---

*Note: This project uses a small dataset for demonstration purposes, and predictions may not be highly accurate due to data limitations.*
