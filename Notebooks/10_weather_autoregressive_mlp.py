import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from auxtorch import (
    count_parameters,
    plot_train_vs_test_error,
    plot_true_vs_pred_scatter,
    predict,
    print_model_parameters,
    print_model_structure,
)
from torch.utils.data import DataLoader
from weather_dataset import WeatherDataset


# ===== Define functions and Classes =====
class CustomMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (64,),
        output_dim: int = 1,
        activation: nn.Module = nn.Tanh,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))  # normalize hidden activations
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))  # output head
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===== Define Constants =====
DATASET_FILE = "https://raw.githubusercontent.com/LukeDitria/pytorch_tutorials/refs/heads/main/section12_sequential/data/weather.csv"
SPLIT_DATE = "2023-01-01"
DAY_RANGE = 15
DAYS_IN = 14

EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ===== Prepare Data =====
dataset_train = WeatherDataset(
    dataset_file=DATASET_FILE,
    day_range=DAY_RANGE,
    split_date=pd.to_datetime(SPLIT_DATE),
    train_test="train",
)

dataset_test = WeatherDataset(
    dataset_file=DATASET_FILE,
    day_range=DAY_RANGE,
    split_date=pd.to_datetime(SPLIT_DATE),
    train_test="test",
)

print(f"Number of training examples: {len(dataset_train)}")
print(f"Number of testing examples: {len(dataset_test)}")

data_loader_train = DataLoader(
    dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
data_loader_test = DataLoader(
    dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)

# ===== Visualize Data =====
fig = plt.figure(figsize=(10, 5))
_ = plt.title("Melbourne Max Daily Temperature (C)")
_ = dataset_train.dataset["Maximum temperature (Degree C)"].plot()
_ = dataset_test.dataset["Maximum temperature (Degree C)"].plot()
_ = plt.legend(["Train", "Test"])


model = CustomMLP(
    input_dim=DAYS_IN * 2, hidden_dims=(128, 64), output_dim=1, use_layernorm=True
)

days, months, datas = next(iter(data_loader_train))

datas.shape
datas.reshape(datas.shape[0], datas.shape[1] * 2).shape
