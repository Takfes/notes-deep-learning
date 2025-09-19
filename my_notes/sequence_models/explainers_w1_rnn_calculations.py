import pandas as pd
import torch
import torch.nn as nn

# ===== Define input =====
# (batch_size, seq_len, input_size)
# Here: (2, 4, 4)
X = torch.tensor(
    [
        [[2, -1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 2, -1, 2]],
        [[3, -2, 1, 0], [2, -1, 2, -1], [2, 0, 1, -1], [2, 1, 0, 1]],
    ],
    dtype=torch.float32,
)

torch.manual_seed(1990)

rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1, batch_first=True)

for name, param in rnn.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:]} \n")

dict(rnn.named_parameters())

dict(rnn.named_parameters()).get("weight_ih_l0")
dict(rnn.named_parameters()).get("bias_ih_l0")

dict(rnn.named_parameters()).get("weight_hh_l0")
dict(rnn.named_parameters()).get("bias_hh_l0")

# ===== Copy weights to clipboard =====
pd.DataFrame(
    dict(rnn.named_parameters()).get("weight_ih_l0").detach().numpy()
).to_clipboard(index=False, header=False)

pd.DataFrame(
    dict(rnn.named_parameters()).get("bias_ih_l0").detach().numpy()
).to_clipboard(index=False, header=False)

pd.DataFrame(
    dict(rnn.named_parameters()).get("weight_hh_l0").detach().numpy()
).to_clipboard(index=False, header=False)

pd.DataFrame(
    dict(rnn.named_parameters()).get("bias_hh_l0").detach().numpy()
).to_clipboard(index=False, header=False)

# ===== Forward pass =====
output, hn = rnn(X)

output
hn
