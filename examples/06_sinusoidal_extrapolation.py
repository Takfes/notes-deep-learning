import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auxtorch import count_parameters, get_device, print_model_parameters
from tqdm import tqdm

# TODO: test pytorch RNN vs custom RNN
# TODO: test multistep vs single step prediction
# TODO: test teacher forcing vs no teacher forcing

# ===== Generate Sinusoidal Data =====
N = 500
timedim = torch.linspace(0, 20 * np.pi, N)
data = torch.sin(timedim + torch.cos(timedim))

# plot the data data
plt.figure(figsize=(15, 4))
plt.plot(data)

# find a good sequence length
pd.DataFrame(data.numpy())[pd.DataFrame(data.numpy())[0] == 1]
seq_length_plot = 50
# pd.DataFrame(data.numpy()).to_clipboard()

# zoom in - plot the data data to determine sequence size
seq_start = 12
seq_end = seq_start + seq_length_plot
plt.figure(figsize=(15, 4))
plt.xlim([seq_start, seq_end])
plt.plot(data)

# ===== Create Models =====


class BuiltinRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        y = self.out(out)
        return y, h.detach()


# ===== Prepare Training Objects =====

INPUT_SIZE = 1
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1
BATCH_SIZE = 1
SEQ_LENGTH = seq_length_plot
LEARNING_RATE = 0.001
EPOCHS = 30

device = get_device()

model = BuiltinRNN(
    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
)
count_parameters(model)
print_model_parameters(model)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

logger_epoch_loss = np.zeros(EPOCHS)

# ===== Training Loop =====

for epoch in tqdm(range(EPOCHS)):
    model.train()
    h = None
    logger_individual_loss = []
    for i in range(N - SEQ_LENGTH):
        # grab train data, xs and ys
        xs = data[i : i + SEQ_LENGTH].reshape(SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE)
        ytrue = data[i + SEQ_LENGTH].reshape(1, 1)
        # forwardprop and loss
        preds, h = model(xs, h)
        yhat = preds[-1]
        loss = loss_fn(yhat, ytrue)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log loss
        logger_individual_loss.append(loss.item())
    # average losses from this epoch
    logger_epoch_loss[epoch] = np.mean(logger_individual_loss)


# ===== Plot Losses =====
plt.plot(logger_epoch_loss, "s-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model loss")
plt.show()


# ==== Evaluate Model =====
h = np.zeros((N, HIDDEN_SIZE))

yHat = np.zeros(N)
hh = None
for timei in range(N - SEQ_LENGTH):
    # grab a snippet of data
    X = data[timei : timei + SEQ_LENGTH].view(SEQ_LENGTH, 1, 1)

    # forward pass and loss
    model.eval()
    with torch.no_grad():
        yy, hh = model(X, hh)
        yHat[timei + SEQ_LENGTH] = yy[-1]
        h[timei + SEQ_LENGTH, :] = hh.detach()


## plot!
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(data, "bs-", label="Actual data", markersize=3)
ax[0].plot(yHat, "ro-", label="Predicted", markersize=3)
ax[0].set_ylim([-1.1, 1.1])
ax[0].legend()

ax[1].plot(data - yHat, "k^")
ax[1].set_ylim([-1.1, 1.1])
ax[1].set_title("Errors")

ax[2].plot(data[SEQ_LENGTH:], yHat[SEQ_LENGTH:], "mo", markersize=3)
ax[2].set_xlabel("Real data")
ax[2].set_ylabel("Predicted data")
r = np.corrcoef(data[SEQ_LENGTH:], yHat[SEQ_LENGTH:])
ax[2].set_title(f"r={r[0, 1]:.2f} (NO Simpson's paradox!)")

plt.suptitle("Performance on training data", fontweight="bold", fontsize=20, y=1.1)
plt.tight_layout()
plt.show()


# ==== Hidden States =====
# show the hidden "states" (units activations)
plt.figure(figsize=(16, 5))
plt.plot(h, "s-", markersize=3)
plt.xlabel("Sequence index")
plt.ylabel("State value (a.u.)")
plt.title("Each line is a different hidden unit")
plt.show()


# ==== Test Model with new sinusoidal data =====
timedim = torch.linspace(0, 30 * np.pi, N)
newdata = torch.sin(timedim + torch.sin(timedim))

# loop over time and predict each subsequent value
yHat = np.zeros(N)
h = None
model.eval()
for timei in range(N - SEQ_LENGTH):
    # grab a snippet of data
    X = newdata[timei : timei + SEQ_LENGTH].view(SEQ_LENGTH, 1, 1)
    # forward pass and loss (don't need hidden states here)
    model.eval()
    with torch.no_grad():
        yy, h = model(X, h)
        yHat[timei + SEQ_LENGTH] = yy[-1]


# plotting
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(newdata, "bs-", label="Actual data", markersize=3)
ax[0].plot(yHat, "ro-", label="Predicted", markersize=3)
ax[0].set_ylim([-1.1, 1.1])
ax[0].legend()

ax[1].plot(newdata - yHat, "k^", markersize=3)
ax[1].set_ylim([-1.1, 1.1])
ax[1].set_title("Errors")

ax[2].plot(newdata[SEQ_LENGTH:], yHat[SEQ_LENGTH:], "mo", markersize=3)
ax[2].set_xlabel("Real data")
ax[2].set_ylabel("Predicted data")
r = np.corrcoef(newdata[SEQ_LENGTH:], yHat[SEQ_LENGTH:])
ax[2].set_title(f"r={r[0, 1]:.2f}")

plt.suptitle("Performance on unseen test data", fontweight="bold", fontsize=20, y=1.1)
plt.show()
