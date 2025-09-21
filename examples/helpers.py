# =============================================================================
# helpers.py
#
# Utility functions for time series data preparation and visualization.
#
# This module provides helpers for creating supervised learning sequences from time series data
# and for visualizing model predictions and errors. All functions are designed for clarity,
# reproducibility, and teaching purposes in machine learning workflows.
# =============================================================================
import numpy as np
import torch


def create_sequences(
    data: np.ndarray, seq_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate input/output sequences for time series prediction.

    Args:
        data (np.ndarray): 1D array of time series data.
        seq_length (int): Length of each input sequence.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of (X, y) where
            X is input sequences of shape (num_samples, seq_length),
            y is target values of shape (num_samples,).
    """
    X: list[np.ndarray] = []
    y: list[np.float32] = []
    # Iterate over the time series to create input/output pairs
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    # Convert lists to torch tensors
    X_tensor: torch.Tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor: torch.Tensor = torch.tensor(np.array(y), dtype=torch.float32)
    return X_tensor, y_tensor


def plot_true_vs_pred_scatter(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    title: str = "Scatterplot: True vs Predicted",
) -> None:
    """
    Plot a scatterplot comparing true and predicted values.

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        title (str): Plot title.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    # Scatter plot of true vs predicted values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true.numpy(), y_pred.numpy(), alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_train_vs_test_error(train_error: list[float], test_error: list[float]) -> None:
    """
    Plot train and test RMSE over epochs.

    Args:
        train_error (list[float]): Training RMSE values per epoch.
        test_error (list[float]): Test RMSE values per epoch.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    # Plot RMSE for train and test sets over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_error, label="Train RMSE")
    plt.plot(test_error, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Train vs Test RMSE Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_vs_pred_timeseries(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    title: str = "True vs Predicted Time Series",
) -> None:
    """
    Plot true and predicted values as time series.

    Args:
        y_true (torch.Tensor | np.ndarray): True values (tensor or array-like).
        y_pred (torch.Tensor | np.ndarray): Predicted values (tensor or array-like).
        title (str): Plot title.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    # Plot time series of true and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Passengers")
    plt.legend()
    plt.show()
