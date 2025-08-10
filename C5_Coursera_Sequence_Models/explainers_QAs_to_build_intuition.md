# Attention as linear transformations - embedding weights change based on context
- [source](https://www.youtube.com/watch?v=UPtG_38Oq8o&list=PLCip3d1iHEMXcAZPhPSb6Br0dykmPKcji&t=2175s)

*   Attention **resolves word ambiguity** by **contextually modifying embeddings**, intuitively visualized as words **"pulling" each other** to new, precise locations in space.
*   These contextual modifications are achieved through **linear transformations (matrices)**, which **rotate, stretch, or shear** embeddings to create better-separated, context-aware representations.
*   The **Query (Q) and Key (K)** matrices transform embeddings to **optimize for similarity calculations** (via dot product), quantifying how words relate to each other.
*   While Q and K are for **finding relationships**, the **Value (V) matrix** creates embeddings **optimized for the Transformer's primary task of next-word prediction**.

# Self, Masked Self and Cross Attention
- [source](https://www.youtube.com/watch?v=uvEax6XwfJc&list=PLCip3d1iHEMXcAZPhPSb6Br0dykmPKcji&t=3s)

All of the 1) Encoder self-attention (bidirectional), 2) Decoder masked self-attention (autoregressive) and 3) Decoder cross-attention (encoder–decoder attention) are implementations of the **Multi Head Attention**, which in turn is based on **the Scaled dot-product attention**. Their main differences lies in their inputs, purpose, masks.

## 1) Encoder self-attention (bidirectional)

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

## 2) Decoder masked self-attention (autoregressive)

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

## 3) Decoder cross-attention (encoder–decoder attention)

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

## Shapes: (B, S, d) vs (B, T, d)

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

## Are $W_Q, W_K, W_V$ shared across blocks?

**No (in the vanilla Transformer).**

* Every **attention sublayer** (encoder self-attn, decoder masked self-attn, decoder cross-attn) has its **own** learnable projections $W_Q, W_K, W_V$ (and an output projection $W_O$).
* Within a sublayer, **each head** has its own slice of those projections.
* **Across layers** (stack depth $N$), parameters are also **not shared**.

---

## How the pieces fit (one sentence each)

* **Encoder self-attention**: builds **source memory** `H` = “every source token, with global source context.”
* **Decoder masked self-attention**: builds **decoder state** `Z` = “what I’ve generated so far, no peeking ahead.”
* **Decoder cross-attention**: forms **source-aware context** for each target step by **querying `H` with `Z`**.

---

## Quick mental model (Q/K/V roles)

* **Query (Q)** = “what I currently need” (decoder state at a step).
* **Key (K)** = “where in the other sequence is that information stored?”
* **Value (V)** = “the actual information to bring back.”
* **Self-attn** uses Q/K/V from the **same** sequence (source or target).
* **Cross-attn** uses Q from **decoder**, K/V from **encoder**.

---

# How does the decoder “feed what’s already generated”?

there are two (2) different cases, depending on whether the we are during training or inferencing.

**Training (teacher forcing):**

* During training we feed **the entire Y_in at once** for efficiency (parallelization) (prepend `<BOS>`, drop last token).
* This tensor `Y` (length **T**) goes into the decoder’s **first block (masked self-attention)**.
* Without a mask, self-attention at position t could “peek” at positions > t in Y_in and cheat (leak future tokens), inflating accuracy but breaking causality.
* The **causal mask** ensures position *t* can only attend to `<  t` positions, even though the whole sequence is present.
* The causal mask zeros out attention to future positions, enforcing the same constraint the model faces at inference.

**Inference (generation):**

* Start with `Y = [<BOS>]`.
* Greedy/beam loop:

  1. Run decoder on the **prefix** `Y`
  2. Queries = current decoder states; Keys/Values = encoder memory H.
  3. Take logits → choose next token → **append to `Y`.**
  4. Repeat until `<EOS>` or max length.

* Key intuition: At each step, masked self-attention builds “what I’ve said so far. Cross-attention pulls “what in the source is relevant now,”. FFN + Linear/Softmax converts that into a probability over the vocabulary.

* Throughout, the **encoder output `H` is fixed**. In **cross-attention**, queries come from the decoder’s current states, keys/values come from `H`. So each step’s query **pulls source context** from `H` and combines it with the decoder’s own left-context (from masked self-attn).

**Intuition flow**

1. **Encoder self-attn →** build **source memory** `H` (each source token enriched by all source tokens) fixed throughout.
2. **Decoder masked self-attn →** build **decoder state** for the current prefix (each target position sees only past).
3. **Decoder cross-attn →** use decoder state as **Q**, encoder memory as **K/V**, to fetch the **relevant source info** for predicting the next token.