
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


## Where the “prediction” happens

* The LSTM cell *by itself* does *not* compute a “final prediction” of the task (unless your task is: “give me the hidden state”—but almost always, your task has some output space, e.g. next value, classification, etc.).
* To make prediction(s), you attach a **“head”** (usually a `Linear` layer, maybe plus activation) to some hidden state:

    * If you want prediction at every timestep → apply the head to each $h_t$ in `output`.
    * If you want one summary prediction → apply the head to `h_n` (or output at last time-step).

---

## The “Insightful Moment”

> *There is no built-in “prediction” under the hood; the LSTM cell itself does not produce “predictions” unless you add a head (e.g. a Linear layer) on top of some $h_t$.*

* When you see diagrams that show an arrow from $h_t$ (hidden state) to $y^{\langle t\rangle}$, that arrow typically represents this **prediction head**. It’s *not part* of the RNN/LSTM unit’s internal calculation.

* The recurrence, gates, $c_t, h_t$ are just internal states and activations. The model designer then uses those (e.g. $h_t$ or $h_n$) to compute whatever it is they want (classification, regression, translation, etc.), via additional layers.

---

## Examples: How you choose which hidden/output state to use, and why

| Scenario                                                                    | Which hidden/output you use                                                                         | Why                                                                                                                                                        |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Many-to-many tasks (e.g. tagging every time step)                           | Use `output` (all $h_t$) → apply head to each $h_t$                                                 | Because you need a prediction at each time step based on full history up to that point.                                                                    |
| Many-to-one tasks (e.g. sentiment classification, or “forecast next value”) | Use `h_n` (last hidden), or equivalently `output[:, -1, :]` (for batch\_first)                      | Because you want one summary of whole sequence.                                                                                                            |
| Encoder-Decoder / seq2seq                                                   | Encoder gives `h_n` and `c_n` to initialize decoder(s). May also pass whole `output` for attention. | Because `h_n, c_n` represent summary memory that decoder uses; attention makes use of all hidden states in `output` so decoder can decide where to attend. |

---

## Distinctions & Clarify the Confusions

* **“Two outputs vs three outputs” in diagrams**:

  * Some diagrams show two outputs: just $h_t$ (hidden) and possibly a prediction $y_t$.
  * Some show three: $h_t$, $c_t$, and $y_t$ (or gates etc).

  The difference is whether cell states are drawn, and whether a prediction head is drawn. PyTorch always gives you hidden + cell (for LSTM), not the gates or prediction head (unless you build one).

---

## Putting it all in a clean narrative (so it sticks)

You can think of an LSTM as doing **internal storytelling**:

* At each word/frame/time $t$, it reads input $x_t$, updates internal “long memory” $c_t$, and makes a working memory view $h_t$.
* If its job is “tell me something about each word/time”, you use $h_t$ at each step. If its job is “tell me something after the whole story”, you use $h_n$ (plus $c_n$, if needed).

Prediction is always external to the cell: it’s what you build *on top* of some hidden state(s). Diagrams often show that arrow from hidden → output/prediction, but in PyTorch that arrow is your Linear (or whatever) layer. The internal LSTM doesn’t “decide” the output shape/prediction target—it simply produces the hidden states.