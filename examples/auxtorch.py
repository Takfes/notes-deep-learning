# =============================================================================
# auxtorch.py
#
# Utility functions for PyTorch experiments and model analysis.
#
# This module provides a collection of helpers to support deep learning workflows, including device management, model inspection, prediction, computational graph visualization, and gradient operations. All functions are designed for clarity, reproducibility, and teaching purposes.
# =============================================================================

from typing import Any, List, Optional

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Detect and return the best available torch device (CUDA, MPS, or CPU).

    Args:
        verbose (bool): If True, prints which device is selected.

    Returns:
        torch.device: The selected device.
    """
    # Check for CUDA GPU
    if torch.cuda.is_available():
        device: torch.device = torch.device("cuda")
        name: str = torch.cuda.get_device_name(0)
        msg: str = f"Using GPU: {name}"
    # Check for Apple Silicon GPU (MPS)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device: torch.device = torch.device("mps")
        msg: str = "Using Apple Silicon GPU"
    else:
        device: torch.device = torch.device("cpu")
        msg: str = "Using CPU"
    if verbose:
        print(msg)
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: Total number of parameters.
    """
    # Sum the number of elements for each parameter tensor in the model
    total_params: int = sum(p.numel() for p in model.parameters())
    return total_params


def print_model_parameters(model: torch.nn.Module) -> None:
    """
    Print the name, shape, and gradient requirement of each model parameter.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        None
    """
    for name, p in model.named_parameters():
        print(f"{name:30s} {tuple(p.shape)} requires_grad={p.requires_grad}")


def print_model_structure(model: torch.nn.Module) -> None:
    """
    Print the hierarchical structure of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        None
    """
    for name, m in model.named_modules():
        print(name, "->", m)


def predict(model: torch.nn.Module, x: Any, return_numpy: bool = True) -> Any:
    """
    Run model prediction on input data, optionally returning a numpy array.

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


def render_graph(tensor: torch.Tensor, params: Optional[dict] = None) -> Any:
    """
    Visualize the computational graph of a tensor using torchviz.

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


def clip_gradients(
    model: torch.nn.Module, max_norm: float, norm_type: float = 2
) -> None:
    """
    Clip gradients of model parameters to prevent exploding gradients.

    Args:
        model (torch.nn.Module): The model whose gradients to clip.
        max_norm (float): Maximum allowed norm.
        norm_type (float): Type of norm (default: 2).

    Returns:
        None
    """
    # Clip gradients in-place for all model parameters
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)
