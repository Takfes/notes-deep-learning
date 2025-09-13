import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
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
        return x.squeeze()  # Remove extra dimensions to match target shape


dataset_url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)

df = pd.read_csv(dataset_url, index_col="Month", parse_dates=True)
dv = df.values
train, test = split_train_test(dv, test_size_ratio=0.3)

X_train, y_train = create_sequences(train, seq_length=3)
X_test, y_test = create_sequences(test, seq_length=3)
[x.shape for x in [X_train, y_train, X_test, y_test]]

train_loader = DataLoader(
    dataset=TensorDataset(X_train, y_train), batch_size=16, shuffle=True
)
test_loader = DataLoader(
    dataset=TensorDataset(X_test, y_test), batch_size=16, shuffle=False
)

# training loop
model = AirPassengerModel()
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    for Xb, yb in train_loader:
        y_pred = model(Xb)
        loss = loss_fn(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if epoch % 1 != 0:
    #     continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))

# ! check shapes
recur = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
xx = next(iter(train_loader))[0]
output, (h_n, c_n) = recur(xx)
output.shape
h_n.shape
c_n.shape
