import torch


def count_parameters(model):
    return sum(x.flatten().shape[0] for x in model.parameters() if hasattr(x, "shape"))


def print_model_parameters(model):
    for name, p in model.named_parameters():
        print(f"{name:30s} {tuple(p.shape)} requires_grad={p.requires_grad}")


def print_model_structure(model):
    for name, m in model.named_modules():
        print(name, "->", m)


def predict(model, x, return_numpy=True):
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
    from torchviz import make_dot

    # Visualize the graph
    dot = make_dot(tensor, params=params)
    # Render the graph to a file (requires Graphviz installed on your system)
    dot.render("computational_graph", format="png")
    # Display in Jupyter Notebook (if using one)
    return dot


def clip_gradients(model, max_norm, norm_type=2):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)


def plot_true_vs_pred_scatter(y_true, y_pred, title="Scatterplot: True vs Predicted"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true.numpy(), y_pred.numpy(), alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_train_vs_test_error(train_error, test_error):
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
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Passengers")
    plt.legend()
    plt.show()
