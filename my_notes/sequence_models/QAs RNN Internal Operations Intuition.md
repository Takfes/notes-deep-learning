
## RNNs tensor shapes are independent of sequence length

### TLDR
- With RNN/LSTM, your parameter count depends only on hidden_size and feature_size, not on sequence length. the number of learnable parameters is independent of seq_len.
- The same weight matrices (input→hidden, hidden→hidden, biases) are shared across all time steps. Therefore, seq_len affects computational cost (more steps to compute, more memory for intermediate states), but not the number of weights.
- The seq_len dimension is needed so the model knows how many time steps to iterate over. 

### Details
- You have a weight matrix $W_{ax} \in \mathbb{R}^{h \times d}$ mapping input features (dimension $d$) to hidden state dimension $h$. This matrix is used at every time step and does not change with the number of time steps; it is reused for each input $x(t)$.
- Similarly, the weight matrix $W_{aa} \in \mathbb{R}^{h \times h}$ is used for mapping the previous hidden state to the next hidden state at every time step.
- Thus, the number of parameters = size of $W_{ax} + W_{aa}$ + biases + output weights, all independent of seq_len.
- The seq_len multiplies the computational work: more time steps mean more matrix multiplies (forward and backward passes), and more memory to store hidden states and activations. However, the learnable parameters are fixed once you set $d$, $h$, (and possibly the number of layers or directions).


## RNNs process multiple example in parallel but different time steps in sequence, because of hidden state dependency

- RNNs don’t batch across time-steps in the sense of simultaneously computing all timesteps’ hidden states with one big matrix multiply alone, because each time step depends on the previous hidden state — To allow temporal dependencies: the hidden state flows from time step t−1 to t.
- RNNs can and do process multiple examples in parallel per time step (batches), but this is across the various training examples in the batch, and not across different steps of the same example.

## RNNs — Batched Shapes (maths-like notation : weights-left)

Let batch size $N$, input dim $d$, hidden dim $h$, output dim $o$.

### Update / Output (batched - two inputs)

$$
\boxed{
A^{\langle t\rangle}
= g\!\left(
W_{aa}\,A^{\langle t-1\rangle}
\;+\;
W_{ax}\,X^{\langle t\rangle}
\;+\;
b_a\,\mathbf{1}_{1\times N}
\right)
}
$$

$$
\boxed{
Y^{\langle t\rangle}
= g_{\text{out}}\!\left(
W_{ya}\,A^{\langle t\rangle}
\;+\;
b_y\,\mathbf{1}_{1\times N}
\right)
}
$$

### Symbols & Shapes (batched at step $t$)

| Item                                                                                       | Shape                                           |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| $X^{\langle t\rangle}=\big[x^{\langle t\rangle}_1,\dots,x^{\langle t\rangle}_N\big]$       | $\mathbb{R}^{d\times N}$                        |
| $A^{\langle t-1\rangle}=\big[a^{\langle t-1\rangle}_1,\dots,a^{\langle t-1\rangle}_N\big]$ | $\mathbb{R}^{h\times N}$                        |
| $A^{\langle t\rangle}$                                                                     | $\mathbb{R}^{h\times N}$                        |
| $Y^{\langle t\rangle}$                                                                     | $\mathbb{R}^{o\times N}$                        |
| $W_{ax}$ (input→hidden)                                                                    | $\mathbb{R}^{h\times d}$                        |
| $W_{aa}$ (hidden→hidden)                                                                   | $\mathbb{R}^{h\times h}$                        |
| $b_a$                                                                                      | $\mathbb{R}^{h\times 1}$ (broadcast across $N$) |
| $W_{ya}$ (hidden→output)                                                                   | $\mathbb{R}^{o\times h}$                        |
| $b_y$                                                                                      | $\mathbb{R}^{o\times 1}$ (broadcast across $N$) |
| $W_a=[W_{aa}\;W_{ax}]$                                                                     | $\mathbb{R}^{h\times (h+d)}$                    |


## How the **batch** dimension $N$ fits in

With a batch of $N$ examples and sequence length $L$, your input is

$$
X \in \mathbb{R}^{N\times L\times d}.
$$

At time $t$, take the slice $X_t = X[:,t,:] \in \mathbb{R}^{N\times d}$.

* **Math (weights-left) view**: treat each step’s mini-batch as matrices of **column vectors**:
  $X_t^{\top}\in\mathbb{R}^{d\times N}$, $A_{t-1}^{\top}\in\mathbb{R}^{h\times N}$.
  Then

  $$
  A_t^{\top}=g\!\left(W_{aa}\,A_{t-1}^{\top}+W_{ax}\,X_t^{\top}+b_a\,\mathbf{1}_N^{\top}\right)\in\mathbb{R}^{h\times N},
  $$

  and transpose back to $A_t\in\mathbb{R}^{N\times h}$.

* **PyTorch (inputs-left) view** (what the code actually does):
  $X_t W_{ih}^{\top}$ with $X_t\in\mathbb{R}^{N\times d}$, $W_{ih}\in\mathbb{R}^{h\times d}$
  gives $\mathbb{R}^{N\times h}$; similarly $A_{t-1} W_{hh}^{\top}$ gives $\mathbb{R}^{N\times h}$. ([PyTorch Documentation][1])

PyTorch’s `matmul`/`bmm` implement batched matrix multiplication when tensors have leading batch dims; that’s how many examples are handled in parallel. ([PyTorch Documentation][3])

---

## Numeric example

Say $N=5$, $L=10$, $d=2$, $h=8$.

* Input: $X\in\mathbb{R}^{5\times10\times2}$
* Weights: $W_{ax}\in\mathbb{R}^{8\times2}$, $W_{aa}\in\mathbb{R}^{8\times8}$
* Step $t$: $X_t\in\mathbb{R}^{5\times2}$, $A_{t-1}\in\mathbb{R}^{5\times8}$
* Compute:
  $X_t W_{ih}^{\top}\in\mathbb{R}^{5\times8}$, $A_{t-1} W_{hh}^{\top}\in\mathbb{R}^{5\times8}$ → sum, add bias, nonlinearity → $A_t\in\mathbb{R}^{5\times8}$. ([PyTorch Documentation][1])
