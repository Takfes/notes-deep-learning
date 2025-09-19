from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (Dense, Embedding, LSTM, Input,
                                            Concatenate, Bidirectional)
from tensorflow.python.keras.datasets import imdb


def build_model(max_len: int,
                max_feats: int,
                emb_dimensions: int,
                n_outputs: int = 1):
    """

    :param max_len:
    :param max_feats:
    :param emb_dimensions:
    :param n_outputs:
    :return:
    """
    # this is the placeholder tensor for the input sequences
    seq = Input(shape=(max_len,), dtype='int32')

    # this embedding layer will transform the sequences of integers into
    # vectors of size 128
    emb_layer = Embedding(max_feats, emb_dimensions, input_length=max_len)

    embedded = emb_layer(seq)

    # apply forwards LSTM
    forwards = LSTM(64)(embedded)

    # apply backwards LSTM
    backwards = LSTM(64, go_backwards=True)(embedded)

    # concatenate the outputs of the 2 LSTMs
    merged = Concatenate([forwards, backwards], axis=-1)

    # after_dp = Dropout(0.5)(merged)
    # output = Dense(1, activation='sigmoid')(after_dp)

    if n_outputs == 1:
        output = Dense(n_outputs,
                       activation='sigmoid')(merged)

        model = Model(input=seq,
                      output=output)

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    else:
        output = Dense(n_outputs,
                       activation='softmax')(merged)

        model = Model(input=seq,
                      output=output)

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model


def build_model2(max_len: int,
                 max_feats: int,
                 emb_dimensions: int,
                 n_outputs: int = 1):
    """

    :param max_len:
    :param max_feats:
    :param emb_dimensions:
    :param n_outputs:
    :return:
    """
    # this is the placeholder tensor for the input sequences
    seq = Input(shape=(max_len,), dtype='int32')

    # this embedding layer will transform the sequences of integers into
    # vectors of size 128
    emb_layer = Embedding(max_feats, emb_dimensions, input_length=max_len)

    embedded = emb_layer(seq)

    # # apply forwards LSTM
    # forwards = LSTM(64)(embedded)
    #
    # # apply backwards LSTM
    # backwards = LSTM(64, go_backwards=True)(embedded)
    #
    # # concatenate the outputs of the 2 LSTMs
    # merged = Concatenate([forwards, backwards], axis=-1)

    # after_dp = Dropout(0.5)(merged)
    # output = Dense(1, activation='sigmoid')(after_dp)

    lstm = Bidirectional(LSTM(64))(embedded)

    if n_outputs == 1:
        output = Dense(n_outputs,
                       activation='sigmoid')(lstm)

        model = Model(input=seq,
                      output=output)

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    else:
        output = Dense(n_outputs, activation='softmax')(lstm)

        model = Model(input=seq, output=output)

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


if __name__ == "__main__":
    # Train a Bidirectional LSTM on the IMDB sentiment classification task.

    # Dataset of 25,000 movies reviews from IMDB, labeled by sentiment
    # (positive/negative). Reviews have been preprocessed, and each review is
    # encoded as a sequence of word indexes (integers).

    max_features = 20000
    maxlen = 100  # cut texts after this number of words (among top
    # max_features most common words)
    batch_size = 32
    emb_dim = 50

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print("Pad sequences (samples x time)")

    x_train = sequence.pad_sequences(x_train,
                                     maxlen=maxlen)

    x_test = sequence.pad_sequences(x_test,
                                    maxlen=maxlen)

    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    rnn_model = build_model(max_len=maxlen,
                            max_feats=max_features,
                            emb_dimensions=emb_dim)

    print('Train...')

    train_samples = 9_000

    rnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=5,
                  validation_split=0.2)

    score = rnn_model.evaluate(
        x_test,  # features
        y_test,  # labels
        batch_size=batch_size,  # batch size
        verbose=1  # the mostX_test extended verbose
    )

    print('\nTest categorical_crossentropy:', score[0])
    print('\nTest accuracy:', score[1])
