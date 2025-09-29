
# FUTURE TASKS

## QUESTIONS/TASKS
- refine and merge the code-commit prompt
- Notebooks :
    - Encoder-Decoder - w/ and w/o attention 
        - chat gpt discussion [store demand dataset](https://chatgpt.com/c/68d8809e-2eec-8323-907a-d0d8e78924a6)
    - sinusodial prediction
    - custom RNNCell would need to manual unroll the series, whilst nn.RNN is fed the entire series and unrolls that internally : "We let Pytorch handle the rollout behind-the-scenes, so just feed in the whole encoder sequence"
    - custom RNN, replicate the operations

- Time Series :
- what happens with trend in timeseries? [medium blog](https://medium.com/@maxbrenner-ai/implementing-seq2seq-models-for-efficient-time-series-forecasting-88dba1d66187) referencing this [https://www.uber.com/en-GR/blog/m4-forecasting-competition/] - how to improve air passengers model; model seems to ignore trend
- Prediction vs forecast task [ChatGPT Conversation](https://chatgpt.com/c/6890586c-1474-8324-a174-e830e8d5ec02)

- Functionalities : 
    - tensorboard logging to training loops
    - model checkpointing to training loops
    - early stopping to training loops
    - LR scheduling to training loops
    - GPU support to training loops


## NEURAL NETS
- [Udemy - Implementing a Neural Network from scratch with Numpy](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/18823906#overview)
- review training & dataset/dataloaders. review batch, stochastic, mini-batch gradient descent

---

## SEQUENCES
**recurrent**
- build a custom RNN; review C5_W1_1 assignment & [PyTorch RNN](https://medium.com/@VersuS_/coding-a-recurrent-neural-network-rnn-from-scratch-using-pytorch-a6c9fc8ed4a7)
- follow along [luke ditria](https://www.youtube.com/watch?v=iKZzXisK1-Q&list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa&index=23) courses, mlp & custom rnn-like architecture
- review [Udemy RNN Code Challenges 217,218](https://titancement.udemy.com/course/deeplearning_x/learn/lecture/29274746#overview) - focus : hidden state visualization
- start from `08_nns_universal_approximators` and explore the `prediction (extrapolation) vs approximation` - adjust learning task : multiple inputs to predict single future output - siren, fourier - recursive models [ChatGPT Conversation](https://chatgpt.com/c/6890586c-1474-8324-a174-e830e8d5ec02)
- for both `approximation` and `prediction` task - test network capacity to learn to approximate multiple (more than one) data generating processes - can a single network learn many functions?

**seq2seq**
- recap info on LSTM vs EncDec vs Transformers - resources
- [Encoder Decoder Models for Dummies](https://medium.com/plain-simple-software/encoder-decoder-models-simply-explained-25a7fccf46d4)
- [A study on Attention mechanism](https://medium.com/perceptronai/a-study-on-attention-mechanism-7d199cf783b6)
- [Attention Seq2Seq with PyTorch: learning to invert a sequence](https://medium.com/data-science/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53)
- [Store Item Demand Forecasting w/ Encoder-Decoder Model](https://medium.com/data-science/encoder-decoder-model-for-multistep-time-series-forecasting-using-pytorch-5d54c6af6e60)
- [What to know before transformers](https://medium.com/@infin94/understanding-the-seq2seq-model-what-you-should-know-before-understanding-transformers-e5891bcd57ec)
- review transformers Coursera's assignment W4_1_Transformer_Subclass_v1

**transformers**
- pytorch implement transformer+attention [umar jamil](https://www.youtube.com/watch?v=ISNdQcPhsts&list=PLCip3d1iHEMXcAZPhPSb6Br0dykmPKcji&index=17)

**mix&match**
- implement recurrent (rnn/gru/lstm), enc-dec, enc-dec+attention through [luke ditria](https://www.youtube.com/watch?v=iKZzXisK1-Q&list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa&index=23) & [aladdin persson](https://www.youtube.com/watch?v=EoGUlvhRYpk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=38)
- [Udemy - Sequence Modelling + Practical Sequence Modelling](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/15615300#overview) 
- consider financial time series task. classification task: what's gonna hit first; increase by x%, 
decrease by x/2% or neither.. regression task is far more classic, i.e. 1-step, or x-step ahead delta.
consider including more time series, for instance you might include BTC for predicting altcoin-X.

---

## CONVOLUTIONS
- [Udemy - Practical Convolutional Networks](https://titancement.udemy.com/course/the-complete-neural-networks-bootcamp-theory-applications/learn/lecture/15615300#overview)
- consider similar task, this time relying on convolution, i.e. feed candle chart images on different 
timeframes. 

---

## GRAPH NETS
- enable gnn explainer excel, how does adjacency, normalized adjacencny etc
- test PyTorch Geometric dataset - perform the same prediction task once using the readily available layers and the other, creating the network from scratch (either numpy or torch)
