"""
Script Summary:
---------------
This script demonstrates time series forecasting on the classic airline passengers dataset using a PyTorch LSTM model. It covers data loading, preprocessing (sequence creation), model definition, training (including a batch overfitting sanity check), evaluation, and visualization of results. The script utilizes helper functions from the `auxtorch` package for model inspection and plotting.

Main Steps:
-----------
1. Loads monthly airline passenger data from a CSV URL.
2. Splits the data into training and test sets.
3. Converts the time series into supervised learning sequences.
4. Defines an LSTM-based regression model for sequence prediction.
5. Performs a sanity check by overfitting a single batch.
6. Trains the model on the full dataset and tracks RMSE for train/test sets.
7. Visualizes training progress and prediction results.

Motivation:
-------------
https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/?utm_source=chatgpt.com
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auxtorch import (
    count_parameters,
    predict,
    print_model_parameters,
    print_model_structure,
)
from helpers import (
    plot_train_vs_test_error,
    plot_true_vs_pred_scatter,
    plot_true_vs_pred_timeseries,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def split_train_test(data, test_size_ratio=0.3):
    train_size = int(len(data) * (1 - test_size_ratio))
    train = data[0:train_size]
    test = data[train_size : len(data)]
    return train, test


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(data.shape[0] - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])  # Predict the next single value
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


class AirPassengerModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, batch_first=True):
        super().__init__()
        self.recur = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        self.ffn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.recur(x)
        x = self.ffn(x[:, -1, :])
        return x


TEST_SIZE_RATIO = 0.3
SEQUENCE_LENGTH = 12
NUM_LSTM_LAYERS = 1
HIDDEN_SIZE = 72
NUM_EPOCHS = 10_000
LEARNING_RATE = 0.001

dataset_url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)

# ===== Load and Prepare Dataset =====
df = pd.read_csv(dataset_url, index_col="Month", parse_dates=True)
dv = df.values
train, test = split_train_test(dv, test_size_ratio=TEST_SIZE_RATIO)

X_train, y_train = create_sequences(train, seq_length=SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test, seq_length=SEQUENCE_LENGTH)
[x.shape for x in [X_train, y_train, X_test, y_test]]

train_loader = DataLoader(
    dataset=TensorDataset(X_train, y_train), batch_size=16, shuffle=True
)
test_loader = DataLoader(
    dataset=TensorDataset(X_test, y_test), batch_size=16, shuffle=False
)

# ===== Model Initialization and Training Setup =====
model = AirPassengerModel(num_layers=NUM_LSTM_LAYERS, hidden_size=HIDDEN_SIZE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
train_error = []
test_error = []

# ===== Model Summary and Parameter Inspection =====
print(model)
print_model_structure(model)
print_model_parameters(model)  # 50 hidden units * 4 gates = 200
count_parameters(model)

# ===== Overfitting on a Single Batch (Sanity Check) =====
Xb, yb = next(iter(train_loader))
Xb = Xb.clone().detach()
yb = yb.clone().detach()

for epoch in range(NUM_EPOCHS):
    model.train()
    y_pred = model(Xb)
    loss = loss_fn(y_pred, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 != 0:
        continue
    y_pred_train = predict(model, Xb, return_numpy=False)
    assert y_pred_train.shape == yb.shape
    train_rmse = np.sqrt(loss_fn(y_pred_train, yb).item())
    train_error.append(train_rmse)
    print(f"epoch: {epoch + 1}/{NUM_EPOCHS}, train_rmse: {train_rmse:.4f}")


# Scatterplot between y_pred_train and yb (from overfitting batch)
plot_true_vs_pred_scatter(
    yb, y_pred_train, title="Scatterplot: True vs Predicted (Overfit Batch)"
)

# ===== Model Initialization and Training Setup =====
model = AirPassengerModel()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
train_error = []
test_error = []

# ===== Training on Full Dataset =====
for epoch in range(NUM_EPOCHS):
    model.train()
    for Xb, yb in train_loader:
        y_pred = model(Xb)
        loss = loss_fn(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 1 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        assert y_pred_train.shape == y_train.shape
        train_rmse = np.sqrt(loss_fn(y_pred_train, y_train).item())
        y_pred_test = model(X_test)
        assert y_pred_test.shape == y_test.shape
        test_rmse = np.sqrt(loss_fn(y_pred_test, y_test).item())
    train_error.append(train_rmse)
    test_error.append(test_rmse)
    print(
        f"epoch: {epoch + 1}/{NUM_EPOCHS}, train_rmse: {train_rmse:.4f}, test_rmse: {test_rmse:.4f}"
    )

# Plot training and test error over epochs
plot_train_vs_test_error(train_error, test_error)

# Plot timeseries of true vs predicted values on test set
y_true, y_pred = [], []
for Xb, yb in test_loader:
    yp = predict(model, Xb, return_numpy=False)
    loss = loss_fn(yp, yb)
    print(f"test batch rmse: {np.sqrt(loss.item()):.4f}")
    y_pred.append(yp)
    y_true.append(yb)

# Concatenate all batches for plotting
y_true_all = torch.cat(y_true).cpu().numpy()
y_pred_all = torch.cat(y_pred).cpu().numpy()

plot_true_vs_pred_timeseries(y_true_all, y_pred_all)
