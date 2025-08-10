# Self Attention (SA), Masked Self Attention (MSA) and Cross Attention
- [source](https://www.youtube.com/watch?v=uvEax6XwfJc&list=PLCip3d1iHEMXcAZPhPSb6Br0dykmPKcji&t=3s)

All of the 1) Encoder self-attention (bidirectional), 2) Decoder masked self-attention (autoregressive) and 3) Decoder cross-attention (encoder–decoder attention) are implementations of the **Multi Head Attention**, which in turn is based on **the Scaled dot-product attention**. Their main differences lies in their inputs, purpose, masks.

### 1) Encoder self-attention (bidirectional)

**Where:** every encoder layer
**Inputs:**

* `X` = source embeddings + positional encodings, shape `(B, S, d)`
* Inside the layer: `Q = XW_Q`, `K = XW_K`, `V = XW_V` (all from the **same** `X`)
  **Mask:** **padding mask** over source positions (hide PAD tokens). Applied to the score matrix before softmax.
  **Goal (intuition):** For each source token, **pull in context from all other source tokens** to build a rich, contextual representation.
  **How:** weights = `softmax(QKᵀ/√d_k + mask)`; output = `weights @ V`.
  **Output “looks like”:** `(B, S, d)` tensor where each position is the **source token enriched with whole-sentence context**.
  **What it becomes:** After N encoder layers this becomes the **encoder memory** `H` that the decoder will read from.

---

### 2) Decoder masked self-attention (autoregressive)

**Where:** first attention sublayer in each decoder layer
**Inputs:**

* `Y` = **shifted-right** target embeddings + positional encodings, shape `(B, T, d)`
* Inside: `Q = YW_Q`, `K = YW_K`, `V = YW_V` (all from **Y**)
  **Mask:**
* **Look-ahead/causal mask** to forbid seeing future target positions.
* Plus target **padding mask** if needed.
  **Goal (intuition):** For each decoding step, summarize **what has been generated so far** (left context only).
  **How:** same attention as above, but causal mask zeros out future columns in `QKᵀ`.
  **Output “looks like”:** `(B, T, d)` tensor of **decoder hidden states so far**—each position knows only its **past**.
  **What it feeds:** This becomes the **queries** for the next sublayer (cross-attention).

---

### 3) Decoder cross-attention (encoder–decoder attention)

**Where:** second attention sublayer in each decoder layer
**Inputs:**

* **Queries** from the decoder’s masked self-attn output `Z` `(B, T, d)`
* **Keys/Values** from encoder memory `H` `(B, S, d)`

  * Inside: `Q = ZW_Q`, `K = HW_K`, `V = HW_V`
    **Mask:** **Source padding mask** (hide PADs in the encoder memory). No look-ahead here.
    **Goal (intuition):** For each target step, **retrieve the most relevant source information** to help decide the next token.
    **How:** weights over source = `softmax(QKᵀ/√d_k + mask)`; context = `weights @ V`.
    **Output “looks like”:** `(B, T, d)` **context vectors**—each target position now carries a **focused summary of the source** aligned to that position.
    **What it feeds:** Added back via residual + norm, then through the decoder **feed-forward**. After N layers, a **Linear → Softmax** produces next-token logits.

---

### Shapes: (B, S, d) vs (B, T, d)

* **B** = batch size.
* **S** = **source** sequence length (encoder side).
* **T** = **target** sequence length (decoder side).
* **d** = model (embedding) dimension, often called `d_model`.

**What each block outputs**

* In attention, the **output shape matches the length of the *queries*** and the last dim is `d` (after the heads are concatenated + output projection).

  * **Encoder self-attention:** queries over the source → **(B, S, d)**.
  * **Decoder masked self-attention:** queries over the target prefix → **(B, T, d)**.
  * **Decoder cross-attention:** queries from decoder (length **T**) attending to encoder memory (length **S**) → **(B, T, d)**.
* Internally, the **attention weights** are `(B, num_heads, T, S)` for cross-attn (or `(B, num_heads, S, S)` for encoder self-attn).

So: last dim `d` is consistent; the **second dimension equals the query length** (S for encoder self-attn, T for both decoder sublayers).


---

### Are $W_Q, W_K, W_V$ shared across blocks?

**No (in the vanilla Transformer).**

* Every **attention sublayer** (encoder self-attn, decoder masked self-attn, decoder cross-attn) has its **own** learnable projections $W_Q, W_K, W_V$ (and an output projection $W_O$).
* Within a sublayer, **each head** has its own slice of those projections.
* **Across layers** (stack depth $N$), parameters are also **not shared**.

---

### How the pieces fit (one sentence each)

* **Encoder self-attention**: builds **source memory** `H` = “every source token, with global source context.”
* **Decoder masked self-attention**: builds **decoder state** `Z` = “what I’ve generated so far, no peeking ahead.”
* **Decoder cross-attention**: forms **source-aware context** for each target step by **querying `H` with `Z`**.

---

### Quick mental model (Q/K/V roles)

* **Query (Q)** = “what I currently need” (decoder state at a step).
* **Key (K)** = “where in the other sequence is that information stored?”
* **Value (V)** = “the actual information to bring back.”
* **Self-attn** uses Q/K/V from the **same** sequence (source or target).
* **Cross-attn** uses Q from **decoder**, K/V from **encoder**.

---

# How does the decoder “feed what’s already generated”?

there are two (2) different cases, depending on the state the model is in, i.e. training or inferencing.

### Training (teacher forcing):

* During training we feed **the entire Y_in at once** for efficiency (parallelization) (prepend `<BOS>`, drop last token).

* This tensor `Y` (length **T**) goes into the decoder’s **first block (masked self-attention)**.

* Without a mask, self-attention at position t could “peek” at positions > t in Y_in and cheat (leak future tokens), inflating accuracy but breaking causality.

* The **causal mask** ensures position *t* can only attend to `<  t` positions, even though the whole sequence is present.

* The causal mask zeros out attention to future positions, enforcing the same constraint the model faces at inference.

* The causal mask is essential to enable parallel training without information leakage. At inference you usually feed only the prefix anyway, which already prevents seeing the future; the mask is still part of the layer but becomes effectively redundant if you truly pass one step at a time. (It matters again if you batch multiple steps or use caching/speculative decoding.)

* In training we feed the **entire** decoder input once:

  ```
  Y_in = [<BOS>, Jane, visits, Africa, in]
  Y_out = [Jane, visits, Africa, in, September, <EOS>]
  ```
* The decoder predicts **each next token** in parallel. Conceptually, each position t sees only the **prefix** and must predict **y\_t**:

| t | decoder sees (masked SA lets it attend only up to t-1) | target y\_t |
| - | ------------------------------------------------------ | ----------- |
| 1 | `<BOS>`                                                | `Jane`      |
| 2 | `<BOS> Jane`                                           | `visits`    |
| 3 | `<BOS> Jane visits`                                    | `Africa`    |
| 4 | `<BOS> Jane visits Africa`                             | `in`        |
| 5 | `<BOS> Jane visits Africa in`                          | `September` |
| 6 | `<BOS> Jane visits Africa in September`                | `<EOS>`     |

* **the label at step t is a single next token**, not the whole growing string (“Jane visits”). We compute a loss at **every position** (sum/mean of cross-entropy over t), but we get all logits in **one forward pass** thanks to the **causal mask**.

* The encoder output `H` (from the French source) is available to every position via **cross-attention** during that same pass.

### Inference (autoregressive decoding/generation) :

* Start with `Y = [<BOS>]`.
* Greedy/beam loop:

  1. Run decoder on the **prefix** `Y`
  2. Queries = current decoder states; Keys/Values = encoder memory H.
  3. Take logits → choose next token → **append to `Y`.**
  4. Repeat until `<EOS>` or max length.

* Key intuition: At each step, masked self-attention builds “what I’ve said so far. Cross-attention pulls “what in the source is relevant now,”. FFN + Linear/Softmax converts that into a probability over the vocabulary.

* Throughout, the **encoder output `H` is fixed**. In **cross-attention**, queries come from the decoder’s current states, keys/values come from `H`. So each step’s query **pulls source context** from `H` and combines it with the decoder’s own left-context (from masked self-attn).

---

Now we don’t have ground truth, so we generate **sequentially**:

1. Start with:

   ```
   Y = [<BOS>]
   ```

   Run decoder (masked SA over just `<BOS>`, cross-attn over encoder `H`) → logits → pick `Jane`.
   `Y = [<BOS>, Jane]`

2. Run again with the new prefix:

   ```
   Y = [<BOS>, Jane]
   ```

   → predict `visits`.
   `Y = [<BOS>, Jane, visits]`

3. Repeat:

   * `[<BOS>, Jane, visits]` → `Africa`
   * `[<BOS>, Jane, visits, Africa]` → `in`
   * `[<BOS>, Jane, visits, Africa, in]` → `September`
   * `[... , September]` → `<EOS>` → stop

Under the hood:

* **Masked self-attention** builds “what I’ve said so far.”
* **Cross-attention** queries the fixed encoder memory `H` to pull the relevant French context each step.
* For speed, implementations use **KV caching** so you don’t recompute attention over the whole prefix each time.

---

### Intuition flow**

1. **Encoder self-attn →** build **source memory** `H` (each source token enriched by all source tokens) fixed throughout.
2. **Decoder masked self-attn →** build **decoder state** for the current prefix (each target position sees only past).
3. **Decoder cross-attn →** use decoder state as **Q**, encoder memory as **K/V**, to fetch the **relevant source info** for predicting the next token.

---

### TL;DR

* **Training:** whole target sequence processed **in parallel**; causal mask prevents peeking ahead; loss at every step.
* **Inference:** generate **one token at a time**, appending to the prefix; same blocks, same masks, but now the prefix grows each step.
