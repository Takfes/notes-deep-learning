
## (many-to-one) - Sequence classification - “finish the sequence, then a Linear”
- [Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- What happens: run the RNN across the whole sequence, take the last hidden state (or the last time-step output), and pass it once through a Linear (classifier/regressor).
- You only use the last state/head. That’s the “finish the sequence, then Linear” pattern, and it’s documented in the classification tutorial.
- Many-to-one (classification/regression of the whole sequence): take output[:, -1, :] or h_n[-1], then a single Linear(H, C_or_1) once.

## (many-to-many over time) - Autoregressive - “Linear every step + per-step loss + one BPTT.”
- [Generating Names with a Character-Level RNN/Autoregressive Generation](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- What happens: at each time step you produce a distribution (e.g., next character), compute a loss for that step, sum losses over steps, and backprop once (BPTT).
- This is why those custom examples “repeat a series of operations at each step”: they want an output at each step (next-char prediction). That’s the generation tutorial’s design.
- In autoregressive tasks (generation), you do use a head at every step and sum the losses—as in the generation tutorial and many custom notebooks
- Can I attach a Linear “at every time step” without writing a Python loop?
- With nn.RNN/GRU/LSTM, the forward pass returns the hidden features for all time steps in one tensor. PyTorch’s nn.Linear(in_features, out_features) applies to the last dimension of an input of any shape. So you can project all time steps in one shot
- Many-to-many (per-step outputs): apply Linear(H, C) to the entire [B, T, H] tensor as shown; no Python loop needed.
- [Autoregressive RNN/Predicting Sequential Data With an RNN](https://www.youtube.com/watch?v=YUGbMdfgpx0&list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa&index=23),[code](https://github.com/LukeDitria/pytorch_tutorials/blob/main/section12_sequential/solutions/Pytorch2_Autoregressive_RNN.ipynb)
- this is yet another standard autoregressive pattern, however this time, it swaps the “distribution” that we saw for the next token generation for “scalar value,” since the task is regression not categorical prediction. At each step the RNN + Linear emit a scalar regression output, not a probability distribution.