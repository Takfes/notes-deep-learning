
# FUTURE TASKS

**neural_networks**
- [Udemy - Implementing a Neural Network from scratch with Numpy](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/18823906#overview)
- start from `08_nns_universal_approximators` and explore the `prediction (extrapolation) vs approximation` - adjust learning task : multiple inputs to predict single future output - siren, fourier - recursive models [ChatGPT Conversation](https://chatgpt.com/c/6890586c-1474-8324-a174-e830e8d5ec02)
- for both `approximation` and `prediction` task - test network capacity to learn to approximate multiple (more than one) data generating processes - can a single network learn many functions?
- review training & dataset/dataloaders. review batch, stochastic, mini-batch gradient descent

---

**seq2seq**
- recap info on LSTM vs EncDec vs Transformers - resources : [Encoder Decoder Models for Dummies](https://medium.com/plain-simple-software/encoder-decoder-models-simply-explained-25a7fccf46d4), [A study on Attention mechanism](https://medium.com/perceptronai/a-study-on-attention-mechanism-7d199cf783b6), [Attention Seq2Seq with PyTorch: learning to invert a sequence](https://medium.com/data-science/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53)
- pytorch implement a recurrent (rnn/gru/lstm), encoder-decoder and encoder-decoder+attention models. 
[luke ditria](https://www.youtube.com/watch?v=iKZzXisK1-Q&list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa&index=23), [aladdin persson](https://www.youtube.com/watch?v=EoGUlvhRYpk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=38)
- pytorch implement transformer+attention [umar jamil](https://www.youtube.com/watch?v=ISNdQcPhsts&list=PLCip3d1iHEMXcAZPhPSb6Br0dykmPKcji&index=17)
- review transformers Coursera's assignment W4_1_Transformer_Subclass_v1
- [Udemy RNN Code Challenges 217,218](https://titancement.udemy.com/course/deeplearning_x/learn/lecture/29274746#overview)
- [Udemy - Practical Recurrent Networks in PyTorch](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/15615300#overview)
- [Udemy - Sequence Modelling + Practical Sequence Modelling](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/15615300#overview) 
- consider financial time series task. classification task: what's gonna hit first; increase by x%, 
decrease by x/2% or neither.. regression task is far more classic, i.e. 1-step, or x-step ahead delta.
consider including more time series, for instance you might include BTC for predicting altcoin-X.

---

**convolution**
- [Udemy - Practical Convolutional Networks](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/15615300#overview)
- consider similar task, this time relying on convolution, i.e. feed candle chart images on different 
timeframes. 

---

**graph_networks**
- enable gnn explainer excel, how does adjacency, normalized adjacencny etc
- test PyTorch Geometric dataset - perform the same prediction task once using the readily available layers and the other, creating the network from scratch (either numpy or torch)
