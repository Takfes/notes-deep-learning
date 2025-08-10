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
