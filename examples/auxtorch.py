# =============================================================================
# auxtorch.py
#
# Utility functions for PyTorch experiments and model analysis.
# Provides device selection, model parameter inspection, prediction helpers,
# computational graph visualization, gradient clipping, and plotting routines.
# Designed for clarity and reproducibility in teaching and experimentation.
# =============================================================================

from typing import Any, List, Optional

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Detects and returns the best available torch device (cuda, mps, or cpu).

    Args:
        verbose (bool): If True, prints which device is selected.

    Returns:
        torch.device: The selected device.
    """
    # Check for CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        msg = f"Using GPU: {name}"
    # Check for Apple Silicon GPU (MPS)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        msg = "Using Apple Silicon GPU"
    else:
        device = torch.device("cpu")
        msg = "Using CPU"
    if verbose:
        print(msg)
    return device


def count_parameters(model):
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: Total number of parameters.
    """
    # Sum the number of elements for each parameter tensor in the model
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def print_model_parameters(model):
    """
    Prints the name, shape, and gradient requirement of each model parameter.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        None
    """
    for name, p in model.named_parameters():
        print(f"{name:30s} {tuple(p.shape)} requires_grad={p.requires_grad}")


def print_model_structure(model):
    """
    Prints the hierarchical structure of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        None
    """
    for name, m in model.named_modules():
        print(name, "->", m)


def predict(model, x, return_numpy=True):
    """
    Runs model prediction on input data, optionally returning numpy array.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        x (Any): Input data (tensor or convertible to tensor).
        return_numpy (bool): If True, returns numpy array; else torch tensor.

    Returns:
        Any: Model predictions as numpy array or torch tensor.
    """
    # Ensure input is a torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    # Set model to evaluation mode and disable gradients
    model.eval()
    with torch.no_grad():
        predictions = model(x)
    # Convert to numpy if requested
    if return_numpy:
        return predictions.numpy()
    return predictions


def render_graph(tensor, params=None):
    """
    Visualizes the computational graph of a tensor using torchviz.

    Args:
        tensor (torch.Tensor): Output tensor to visualize.
        params (dict, optional): Model parameters for annotation.

    Returns:
        Any: torchviz graph object.
    """
    from torchviz import make_dot

    # Visualize the graph
    dot = make_dot(tensor, params=params)
    # Render the graph to a file (requires Graphviz installed on your system)
    dot.render("computational_graph", format="png")
    # Display in Jupyter Notebook (if using one)
    return dot


def clip_gradients(model, max_norm, norm_type=2):
    """
    Clips gradients of model parameters to prevent exploding gradients.

    Args:
        model (torch.nn.Module): The model whose gradients to clip.
        max_norm (float): Maximum allowed norm.
        norm_type (float): Type of norm (default: 2).

    Returns:
        None
    """
    # Clip gradients in-place for all model parameters
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)


def plot_true_vs_pred_scatter(y_true, y_pred, title="Scatterplot: True vs Predicted"):
    """
    Plots a scatterplot comparing true and predicted values.

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        title (str): Plot title.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true.numpy(), y_pred.numpy(), alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_train_vs_test_error(train_error, test_error):
    """
    Plots train and test RMSE over epochs.

    Args:
        train_error (List[float]): Training RMSE values per epoch.
        test_error (List[float]): Test RMSE values per epoch.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_error, label="Train RMSE")
    plt.plot(test_error, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Train vs Test RMSE Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_vs_pred_timeseries(y_true, y_pred, title="True vs Predicted Time Series"):
    """
    Plots true and predicted values as time series.

    Args:
        y_true (Any): True values (tensor or array-like).
        y_pred (Any): Predicted values (tensor or array-like).
        title (str): Plot title.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Passengers")
    plt.legend()
    plt.show()
