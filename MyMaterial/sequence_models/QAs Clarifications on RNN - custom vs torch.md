
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
