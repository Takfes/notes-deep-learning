# =============================================================================
# pretorch.py
#
# Custom PyTorch modules for deep learning experiments and model prototyping.
# Includes residual MLP blocks, flexible MLP architectures, and a simple RNN cell.
# Designed for clarity, extensibility, and teaching purposes.
# =============================================================================

import torch
import torch.nn as nn


class ResBlockMLP(nn.Module):
    """
    Residual block for MLP architectures with normalization and skip connection.

    Attributes:
        norm1 (nn.LayerNorm): Layer normalization for input.
        fc1 (nn.Linear): First linear layer (input to half).
        norm2 (nn.LayerNorm): Layer normalization for hidden.
        fc2 (nn.Linear): Second linear layer (hidden to output).
        fc3 (nn.Linear): Skip connection (input to output).
        act (nn.Module): Activation function (ELU).
    """

    norm1: nn.LayerNorm
    fc1: nn.Linear
    norm2: nn.LayerNorm
    fc2: nn.Linear
    fc3: nn.Linear
    act: nn.Module

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes the residual MLP block.

        Args:
            input_size (int): Input feature dimension.
            output_size (int): Output feature dimension.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size)  # does not change size
        self.fc1 = nn.Linear(input_size, input_size // 2)  # reduce by half
        self.norm2 = nn.LayerNorm(input_size // 2)  # does not change size
        self.fc2 = nn.Linear(input_size // 2, output_size)  # output size
        self.fc3 = nn.Linear(input_size, output_size)  # skip - input to output size
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_size).
        """
        x = self.act(self.norm1(x))
        skip = self.fc3(x)
        x = self.act(self.norm2(self.fc1(x)))
        x = self.fc2(x)
        return x + skip


class CustomMLP(nn.Module):
    """
    Flexible multi-layer perceptron (MLP) with optional normalization and dropout.

    Attributes:
        net (nn.Sequential): The sequential MLP network.
    """

    net: nn.Sequential

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (64,),
        output_dim: int = 1,
        activation: type = nn.Tanh,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> None:
        """
        Initializes the custom MLP.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dims (tuple): Hidden layer sizes.
            output_dim (int): Output feature dimension.
            activation (type): Activation function class (e.g., nn.Tanh).
            dropout (float): Dropout probability.
            use_layernorm (bool): Whether to use layer normalization.
        """
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))  # Linear Layer
            if use_layernorm:
                layers.append(nn.LayerNorm(h))  # normalize hidden activations
            layers.append(activation())  # Activation Function
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # Optional Dropout
            prev = h

        layers.append(nn.Linear(prev, output_dim))  # output head
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim).
        """
        return self.net(x)


class RNNCell(nn.Module):
    """
    Simple RNN cell with tanh activation and output head.

    Attributes:
        input_size (int): Input feature dimension.
        hidden_size (int): Hidden state dimension.
        i2h (nn.Linear): Input-to-hidden linear layer.
        h2o (nn.Linear): Hidden-to-output linear layer.
        activation (nn.Module): Activation function (Tanh).
    """

    input_size: int
    hidden_size: int
    i2h: nn.Linear
    h2o: nn.Linear
    activation: nn.Module

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Initializes the RNN cell.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden state dimension.
        """
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> tuple:
        """
        Forward pass for the RNN cell.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, input_size).
            hidden (torch.Tensor): Hidden state tensor of shape (batch, hidden_size).

        Returns:
            tuple: (output tensor, new hidden state)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat((input, hidden), 1)
        hidden = self.activation(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden
