
## motivation - [NLP From Scratch: Generating Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html). How does this custom architecture differe vs an out-of-the-box nn.RNN implementation? 

### nn.RNN loops through timesteps internally, maintaining and updating a hidden state
- What nn.RNN would do instead; **nn.RNN encapsulates the recurrence internally**. You pass it the **whole sequence tensor** and it returns output and h_n.
- “Is recurrence handled by the way we set up training?”; Partly. The recurrence is implemented in the forward (the i2h mapping that consumes h_{t-1}), and the **training code drives it timestep by timestep**
- Check a high level implementation of [nn.RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html?utm_source=chatgpt.com). Iterates over sequence length (and layers)

### When using nn.RNN, you typically provide input as a batch of sequences.
- In nn.RNN you would run the input through the network through dedicated data structures/matrices, instead of running through examples a word at a time(?)
- Padding: You manually extend each sequence in a batch to the same fixed length by adding a special token <PAD> to shorter sequences.
- Packing: pack_padded_sequence() in PyTorch convert a padded batch into a compact representation that only includes valid timesteps. The RNN then skips computations on padding, yielding both efficiency and cleaner handling of hidden state outputs.
- [Why do we "pack" the sequences in PyTorch?](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch)


### Iterative loss accumulation is needed for the custom implementation with the loop.  
- For each character in the word, call rnn(category, input_char_t, hidden) to get (output_t, hidden_t).
- Accumulate loss at each step against the next character.
- Call loss.backward() once at the end → backpropagation-through-time (BPTT) through the unrolled steps. *Check BPTT QAs for more detail*


### Practical Differences 

- Unrolling & shapes
    - CustomRNN/RNNCell: you loop over t, use inputs shaped (B, input_size) each step.
    - nn.RNN: you pass (B,L,input_size) once; it loops internally (fast, fused), returns (B,L,hidden) and h_n

- the gist: your class = cell + manual unroll; nn.RNN = optimized unroll + extras. Add a linear head to either to get task predictions.

### When to prefer which

- Use your cell (or nn.RNNCell) when you need step-wise custom logic (teacher forcing, scheduled sampling, constraints, attention fused per step, TBPTT control). 

- Use nn.RNN for simplicity & speed when you just need a vanilla Elman RNN over a sequence (multi-layer, bidirectional, packed sequences, cuDNN). 

---

Here’s the core idea, super compact.

### Why a **GRUCell** in a decoder?

A decoder generates outputs **step-by-step** (autoregressively). At each step you often need fine control to:

* **Teacher-force** (use ground-truth vs previous prediction).
* **Inject step-specific exogenous features** (future calendars, known covariates).
* **Fuse attention context** from the encoder at each step.
* Do **scheduled sampling**, masking, or custom logic.

`GRUCell` gives you a **single-time-step GRU**: you call it in a Python loop, decide inputs each step, and keep the hidden state yourself.

### GRUCell vs GRU (key differences)

* **`nn.GRU`**: processes a whole sequence in one call (`(T,B,*) → (T,B,*)`), uses cuDNN, **faster**. You get less step-wise control (unless you unroll manually).
* **`nn.GRUCell`**: processes **one step** (`(B,*) → (B,*)`). You write the loop. **Maximum flexibility**, usually **slower**.

Both share parameters across time; `GRU` just wraps the loop in optimized C++/cuDNN.

---

### Minimal decoder with **GRUCell**

```python
import torch, torch.nn as nn
class GRUCellDecoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, hidden)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, h0, steps, y_prev=None, x_future=None, teacher_forcing=False):
        """
        h0: (B,H) init from encoder
        steps: int forecast horizon
        y_prev: (B,1) first input seed (e.g., last known target)
        x_future: (steps,B,Exo) known future features per step or None
        """
        B, H = h0.shape
        h, y = h0, []
        inp = y_prev  # (B,1)
        for t in range(steps):
            xt = torch.cat([inp, x_future[t]] , dim=-1) if x_future is not None else inp
            h = self.cell(xt, h)            # (B,H)
            yt = self.proj(h)               # (B,1)
            y.append(yt)
            inp = y_prev[:,t:t+1] if (teacher_forcing and y_prev is not None) else yt
        return torch.stack(y, dim=1)        # (B,steps,1)
```

### Same idea with **nn.GRU** (less control, faster)

```python
class GRUDecoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.rnn  = nn.GRU(in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, h0, inputs):  # inputs: (B,steps,in_dim) prebuilt with TF/covariates
        out, _ = self.rnn(inputs, h0.unsqueeze(0))  # out: (B,steps,H)
        return self.proj(out)                       # (B,steps,out_dim)
```

With `nn.GRU` you must **pre-construct the whole input sequence** (e.g., mix teacher forcing & covariates ahead of time), which makes per-step choices trickier.

---

### When to pick which

* Need **step-wise tricks** (teacher forcing, scheduled sampling, attention, constraints, quantiles per step)? → **`GRUCell` or custom cell.**
* Need **speed** and straightforward sequences? → **`nn.GRU` (or `nn.LSTM`)**.
* Need **global receptive field / long horizon** with known future covariates? → **Transformer/TCN/SSM** decoders.

That’s it: `GRUCell` = control; `GRU` = throughput. For bespoke decoding logic, write your own small recurrence or use an attention-augmented step loop.

