import os
from glob import glob
from typing import Tuple, List

import numpy as np
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


# Download Dataset from here
# https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset/downloads/imdb-movie-reviews-dataset.zip/1
# Unzip dataset in the same folder.


def load_imdb_data(dataset_type: str = 'train') -> Tuple[List, List]:
    """

    :param dataset_type:
    :return:
    """
    assert dataset_type in ['train', 'test']

    IMDB_DIR = os.path.join(os.getcwd(), 'aclImdb')
    TRAIN_DIR = os.path.join(IMDB_DIR, 'train')  # Get the path of the train set

    # Setup empty lists to fill
    targets, inputs = list(), list()

    # We will use this mapping for tranforming the label to integer
    label_map = {
        'neg': 0,
        'pos': 1
    }

    # First go through the negatives, then through the positives
    for label_type in ['neg', 'pos']:

        # Get the sub path
        dir_name = os.path.join(TRAIN_DIR, label_type)

        print('Loading ', label_type)

        # Loop over all files in path that

        txt_files = glob(os.path.join(dir_name, '*.txt'))

        for fname in tqdm(txt_files,
                          desc='Loading Text Files'):
            # Read the text file and put it in the list
            fname = os.path.join(dir_name, fname)

            with open(fname, 'r') as f:
                inputs.append(f.read())

            # Attach the corresponding label
            targets.append(label_map[label_type])

            # if label_type == 'neg':
            #     labels.append(0)
            # else:
            #     labels.append(1)

    return inputs, targets


def load_glove_embeddings(dim: int = 100) -> dict:
    """

    :param dim: The embeddings size (dimensions)
    :return:
    """
    glove_dir = os.path.join(os.getcwd(), 'glove.6B')  # This is the folder with the embeddings

    print('Loading word vectors')

    embed_index = dict()  # We create a dictionary of word -> embedding

    fname = os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(dim))

    f = open(fname)  # Open file

    # In the dataset, each line represents a new word embedding
    # The line starts with the word and the embedding values follow
    for line in tqdm(f, desc='Loading Embeddings', unit='word'):
        values = line.split()
        # The first value is the word, the rest are the values of the embedding
        word = values[0]
        # Load embedding
        embedding = np.asarray(values[1:], dtype='float32')

        # Add embedding to our embedding dictionary
        embed_index[word] = embedding
    f.close()

    print('Found %s word vectors.' % len(embed_index))

    return embed_index


def create_embeddings_matrix(emb_index: dict,
                             tokenizer: Tokenizer,
                             emb_dim: int = 100) -> np.ndarray:
    """

    :param emb_index: Embeddings Index
    :param tokenizer: Keras fitted tokenizer.
    :param emb_dim: Embeddings dimension.
    :return: A matrix of shape (nb_words, emb_dim) containing the globe embeddings.
    """
    assert emb_dim in [50, 100, 200, 300]

    # Create a matrix of all embeddings
    all_embs = np.stack(
        emb_index.values())  # .values() gets the all the arrays from the keys

    # Calculate mean
    emb_mean = all_embs.mean()
    # Calculate standard deviation
    emb_std = all_embs.std()

    print("Embeddings AVG: {} | STD: {}".format(emb_mean, emb_std))

    # We can now create an embedding matrix holding all word vectors.

    word_index = tokenizer.word_index

    # How many words are there actually. Because we may have requested X most common tokens
    # and the total tokens are X/2
    nb_words = min(max_words,
                   len(word_index))

    # Create a random matrix with the same mean and std as the embeddings

    embedding_matrix = np.random.normal(emb_mean,  # mean
                                        emb_std,  # std
                                        (nb_words, emb_dim)  # shape of the matrix
                                        )

    # The vectors need to be in the same position as their index.
    # Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

    # Loop over all words in the word index
    for word, i in word_index.items():  # .items() return a tuple with (word, word_index)

        # If we are above the amount of words we want to use we do nothing
        if i >= max_words:
            continue

        # Get the embedding vector for the word
        embedding_vector = emb_index.get(word)

        # If there is an embedding vector, put it in the embedding matrix
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def build_model_1(maximum_words,
                  max_seq_len,
                  emb_dim: int = 50,
                  ):
    """

    :param maximum_words: Total number of words to be used by the model
    :param max_seq_len: The sequence length for each text (total number of tokens)
    :param emb_dim: The size of the embeddings vector.
    :return: A sequential model.
    """

    seq_model = Sequential()
    seq_model.add(Embedding(input_dim=maximum_words,
                            output_dim=emb_dim,
                            embeddings_initializer='uniform',
                            input_length=max_seq_len))

    seq_model.add(Flatten())

    seq_model.add(Dense(32,
                        activation='relu'))

    seq_model.add(Dense(1,
                        activation='sigmoid'))

    print(seq_model.summary())

    seq_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])

    return seq_model


def build_model_with_glove_embeddings(maximum_words,
                                      emb_dim,
                                      max_seq_len,
                                      emb_matrix):
    """

    :param maximum_words: Total number of words to be used by the model
    :param emb_dim: The size of the embeddings vector.
    :param max_seq_len: The sequence length for each text (total number of tokens)
    :param emb_matrix: The pretrained glove embedding matrix to be used as weights.
    :return: a keras sequential model.
    """

    seq_model = Sequential()
    seq_model.add(Embedding(input_dim=maximum_words,
                            output_dim=emb_dim,
                            input_length=max_seq_len,
                            weights=[emb_matrix],
                            trainable=False))

    seq_model.add(Flatten())
    seq_model.add(Dense(32, activation='relu'))
    seq_model.add(Dense(1, activation='sigmoid'))

    print(seq_model.summary())

    # Notice that we now have far fewer trainable parameters.
    seq_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])

    return seq_model


def predict_new_comment(text: str,
                        tokenizer: Tokenizer,
                        seq_max_length: int,
                        model: Sequential):
    """

    :param text:
    :param tokenizer:
    :param seq_max_length:
    :param model:
    :return:
    """

    seq = tokenizer.texts_to_sequences([text])
    print('raw seq:', seq)

    seq = pad_sequences(seq,
                        maxlen=seq_max_length)

    print('padded seq:', seq)

    prediction = model.predict(seq)

    prob = 100 * prediction[0][0]

    print('Positivity probability: {} %'.format(round(prob, 3)))

    return int(prob > 50.0)


if __name__ == "__main__":
    texts, labels = load_imdb_data('train')

    print(len(labels), len(texts))

    print(np.mean(labels))

    # positive review
    print('Label: {}'.format(labels[24000]))
    print(texts[24000])

    # negative review
    print('Label: {}'.format(labels[1]))
    print(texts[1])

    max_words = 10_000  # We will only consider the 10K most used words in this dataset

    # Setting up Keras tokenizer
    reviews_tokenizer = Tokenizer(num_words=max_words,
                                  lower=True,
                                  oov_token='<OOV>')

    # this is like the .fit() that we call we using Scikit-learn and Count-Vectorizer
    reviews_tokenizer.fit_on_texts(texts)  # Generate tokens by counting frequency

    # this is like the .transform() that we call we using Scikit-learn and Count-Vectorizer.
    # The major difference is that it turns text into sequence of numbers. NOT one-hot-encoding
    sequences = reviews_tokenizer.texts_to_sequences(texts)

    # The tokenizers word index is a dictionary that maps each word to a number.
    # You can see that words that are frequently used in discussions about
    # movies have a lower token number.
    word_index = reviews_tokenizer.word_index

    print('Token for "the": {}'.format(word_index['the']))
    print('Token for "Movie": {}'.format(word_index['movie']))
    print('Token for "generator": {}'.format(word_index['generator']))

    # Display the first 10 words of the sequence tokenized
    print(sequences[24002][:10])

    # To proceed, we now have to make sure that all text sequences we feed into the model
    # have the same length.

    # We can do this with Keras pad sequences tool.
    # It cuts of sequences that are too long and adds zeros to sequences that are too short.

    # Make all sequences 100 words long
    maxlen = 100

    data = pad_sequences(sequences, maxlen=maxlen)

    # We have 25K, 100 word sequences now
    print('New data shape: {}'.format(data.shape))

    # Now we can turn all data into proper training and validation data.
    labels = np.asarray(labels)

    # Shuffling data

    # Using the length of the texts we create indexes
    indices = np.arange(data.shape[0])  # numpy's range() function. eg array([0, 1, 2, 3, 4, 5])

    # We shuffle the indices on the fly, eg: array([3, 0, 1, 4, 2, 5])
    np.random.shuffle(indices)

    data = data[indices]  # we get the shuffled texts
    labels = labels[indices]  # and the shuffled sentiments

    training_samples = 20000  # We will be training on 20K samples
    validation_samples = 5000  # We will be validating on 5k samples

    # Split data
    x_train = data[:training_samples]
    y_train = labels[:training_samples]

    x_val = data[training_samples:]
    y_val = labels[training_samples:]

    print('X train shape: {}'.format(x_train.shape))
    print('y train shape: {}'.format(y_train.shape))

    print('X val shape: {}'.format(x_val.shape))
    print('y val shape: {}'.format(y_val.shape))

    # Embeddings
    # Words and word tokens are categorical features.
    # As such, we can not directly feed them into the neural net.
    # Just because a word has a larger token value, it does not express a higher value in any way.
    # It is just a different category.
    # We have already dealt with categorical data by turning it into one-hot-encoded vectors.
    # (by using Scikit-Learn's CountVectorizer()
    # But for words, this is impractical.
    # Since our vocabulary is 10,000 words, each vector would contain 10,000 numbers which
    # are all zeros except for one. This is highly inefficient.
    # Instead we will use an embedding.

    # Embeddings also turn categorical data into vectors.
    # But instead of creating a one hot vector, we create a vector in which all elements are numbers

    # In practice, embeddings work like a look up table. For each token, they store a vector

    # In practice it looks like this:
    # * We have to specify how large we want the word vectors to be.
    # A 50 dimensional vector is able to capture good embeddings even for quite large vocabularies.
    # We also have to specify:
    # for how many words we want embeddings and how long our sequences are.

    embedding_dim = 50

    # I've already created a function that wraps up the model. Use this if you want
    # model = build_model_1(embedding_dim= 50)

    # Setting the model
    model_1 = Sequential()
    model_1.add(Embedding(input_dim=max_words,
                          output_dim=embedding_dim,
                          embeddings_initializer='uniform',
                          mask_zero=False,
                          input_length=maxlen))

    model_1.add(Flatten())

    model_1.add(Dense(32,
                      activation='relu'))

    model_1.add(Dense(1,
                      activation='sigmoid'))

    print(model_1.summary())

    # You can see that the Embedding layer has 500,000 trainable parameters,
    # that is 50 parameters for each of the 10K words.

    model_1.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['acc'])

    history = model_1.fit(x_train, y_train,
                          epochs=10,
                          batch_size=32,
                          validation_data=(x_val, y_val))

    # SOS
    # Note that training your own embeddings is prone to over fitting.

    # As you can see our model archives 100% accuracy
    # on the training set but only around  83-84% accuracy on the validation set.
    # A clear sign of over fitting

    #  In practice it is therefore quite rare to train new embeddings unless you have
    #  a massive dataset. More commonly, pre-trained embeddings are used.

    # # Glove Embeddings
    # # https://nlp.stanford.edu/projects/glove/
    # We will use the following small pretrained-embedding dataset
    # http://nlp.stanford.edu/data/glove.6B.zip

    # # After downloading the GloVe embeddings from the GloVe website
    # we can load them into our model

    embedding_dim = 100  # We now use larger embeddings

    embeddings_index = load_glove_embeddings(dim=embedding_dim)

    # Not all words that are in our IMDB vocabulary might be in the GloVe embedding though.
    #
    # For missing words it is wise to use random embeddings
    # with the ame mean and standard deviation as the GloVe embeddings

    embedding_matrix = create_embeddings_matrix(emb_index=embeddings_index,
                                                tokenizer=reviews_tokenizer,
                                                emb_dim=embedding_dim)
    #

    # This embedding matrix can be used as weights for the embedding layer.
    # This way, the embedding layer uses the pre-trained GloVe weights instead of random ones.

    # We can also set the embedding layer to NOT trainable. This means, Keras won't change the
    # weights of the embeddings while training which makes sense since our embeddings are already
    # trained.

    model_2 = Sequential()
    model_2.add(Embedding(max_words,
                          embedding_dim,
                          input_length=maxlen,
                          weights=[embedding_matrix],
                          trainable=False))

    model_2.add(Flatten())
    model_2.add(Dense(32, activation='relu'))
    model_2.add(Dense(1, activation='sigmoid'))
    print(model_2.summary())

    # Notice that we now have far fewer trainable parameters.
    model_2.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['acc'])

    history2 = model_2.fit(x_train, y_train,
                           epochs=10,
                           batch_size=32,
                           validation_data=(x_val, y_val))

    # Now our model over fits less but also does worse on the validation set.

    # Using our model
    # To determine the sentiment of a text, we can now use our trained model.
    # Demo on a positive text
    my_text = 'I love dogs. Dogs are the best.' \
              ' They are lovely, cuddly animals that only want the best for humans.'

    seq = reviews_tokenizer.texts_to_sequences([my_text])
    print('raw seq:', seq)

    seq = pad_sequences(seq, maxlen=maxlen)
    print('padded seq:', seq)

    prediction = model_2.predict(seq)
    print('positivity:', prediction)

    # Demo on a negative text
    my_text = 'The bleak economic outlook will force many small businesses into bankruptcy.'

    predict_new_comment(text=my_text,
                        tokenizer=reviews_tokenizer,
                        seq_max_length=maxlen,
                        model=model_2)
