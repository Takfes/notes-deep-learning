import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.activation(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden


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
