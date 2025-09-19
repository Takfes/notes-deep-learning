#!/usr/bin/env python
from __future__ import print_function

import re
import tarfile
from functools import reduce
from pprint import pprint
from typing import List, Tuple

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import (recurrent, Input, Embedding,
                                            RepeatVector, Dropout, Dense, LSTM)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.data_utils import get_file
from tqdm import tqdm


def tokenize(sent: str) -> List[str]:
    """
    Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    >>> ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']


    :param sent: str. A sentence
    :return: A list of tokens
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines: List[bytes], only_supporting: bool = False,
                  verbose: int = 1) -> list:
    """
    Parse stories provided in the bAbi tasks format

    lines example:
    [b'1 Mary moved to the bathroom.\n',
     b'2 Sandra journeyed to the bedroom.\n',
     b'3 Mary got the football there.\n',
     b'4 John went to the kitchen.\n',
     b'5 Mary went back to the kitchen.\n',
     b'6 Mary went back to the garden.\n',
     b'7 Where is the football? \tgarden\t3 6\n',
     b'8 Sandra went back to the office.\n',
     b'9 John moved to the office.\n',
     b'10 Sandra journeyed to the hallway.\n',
     b'11 Daniel went back to the kitchen.\n',
     b'12 Mary dropped the football.\n',
     b'13 John got the milk there.\n',
     b'14 Where is the football? \tgarden\t12 6\n',
     b'15 Mary took the football there.\n',
     b'16 Sandra picked up the apple there.\n',
     b'17 Mary travelled to the hallway.\n',
     b'18 John journeyed to the kitchen.\n',
     b'19 Where is the football? \thallway\t15 17\n',
     b'20 John moved to the hallway.\n',
     b'21 Sandra left the apple.\n',
     b'22 Where is the apple? \thallway\t21 10\n',
     b'23 Mary got the apple there.\n',
     b'24 John travelled to the garden.\n',
     b'25 John went back to the hallway.\n',
     b'26 John went back to the bedroom.\n',
     b'27 Mary journeyed to the bedroom.\n',
     b'28 John journeyed to the kitchen.\n',
     b'29 John left the milk.\n',
     b'30 Mary left the apple.\n',
     b'31 Where is the milk? \tkitchen\t29 28\n',   --> here is the question
     and the response

    :param lines: A list of bytes string. Each row has an id and a text
    :param only_supporting: bool. If True, only the sentences that support the
     answer are kept.
    :param verbose: int. Verbosity level

    :return:
    """

    data, story = list(), list()

    for line in tqdm(lines, desc='Parsing Lines', unit='story_line'):

        # convert bytes to string
        line = line.decode('utf-8').strip()

        # getting the id and the rest of the text
        nid, line = line.split(' ', 1)  # splits only in the first space
        # (creates a 2 dimension list]

        # convert the id to integer
        nid = int(nid)

        if nid == 1:  # new story
            story = list()

        if '\t' in line:
            quest, ans, supporting = line.split('\t')

            # tokenizing the question
            quest = tokenize(quest)

            if only_supporting:
                # Only select the related sub_story
                supporting = map(int, supporting.split())
                sub_story = [story[i - 1] for i in supporting]

            else:
                # Provide all the sub_stories
                sub_story = [x for x in story if x]

            data.append((sub_story, quest, ans))

            story.append('')

        else:
            # split the sentence into tokens
            sent = tokenize(line)
            story.append(sent)

    if verbose > 0:
        print()
        pprint(data[0])

    return data


# In[4]:


def get_stories(
        f, only_supporting: bool = False,
        max_length: int = None) -> List[Tuple[List[str], List[str], str]]:
    """
    Given a file name:
    a) read the file
    b) retrieve the stories
    c) convert the sentences into a single story.

    If max_length is supplied, any stories longer than 'max_length' tokens will
     be discarded.

    :param f:
    :param only_supporting:
    :param max_length:
    :return:
    """

    data = parse_stories(f.readlines(), only_supporting=only_supporting)

    flatten = lambda info: reduce(lambda x, y: x + y, info)

    # output = [(flatten(story), q, answer) for story, q, answer in data
    # if not max_length or len(flatten(story)) < max_length]

    output = list()

    for story, q, answer in data:

        if not max_length or len(flatten(story)) < max_length:
            # creates a tuple with 3 inputs
            # the first inputs is a list of all the tokens for the stories
            # the second is a list of the tokens for the question
            # the thirds is the response
            output.append((flatten(story), q, answer))

    return output


# In[5]:


def vectorize_stories(data: List[Tuple[List[str], List[str], str]],
                      word_to_idx: dict,
                      story_max_len: int,
                      query_max_len: int) -> Tuple:
    """
    This function vectorizes the data
    The data consists of a Tuple containing the following
    1) A list of tokens depicting the story
    2) A list of tokens depicting the question
    3) A token depicting the response
    
    Vectorizes the stories, the questions, and the responses
    
    :param data:
    :param word_to_idx:
    :param story_max_len:
    :param query_max_len:
    :return:
    """

    xs = []  # list for the stories
    xqs = []  # list for the questions
    ys = []  # list for the outputs (answers)

    for story, query, answer in data:
        x = [word_to_idx[w.strip().lower()] for w in
             story]  # indexes of the tokens for each story
        xq = [word_to_idx[w.strip().lower()] for w in
              query]  # indexes of the tokens for each question

        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_to_idx) + 1)  # creates a list of zeros

        # put one 1 in where the token of the response is located (one-hot
        # encoding the answers since it's answer is a single word
        y[word_to_idx[answer.strip().lower()]] = 1

        xs.append(x)
        xqs.append(xq)
        ys.append(y)

    # padding the story sequences
    pad_xs = pad_sequences(xs,
                           maxlen=story_max_len)

    # padding the question sequences
    pad_xqs = pad_sequences(xqs,
                            maxlen=query_max_len)

    # converting the list of np.arrays to a single np.ndarray
    np_ys = np.array(ys)

    # returns a tuple of three numpy arrays.
    return pad_xs, pad_xqs, np_ys


# In[6]:


def fetch_file():
    """
    fetches the file if not existent.
    :return:
    """
    try:
        fpath = get_file(
            fname='babi-tasks-v1-2.tar.gz',
            origin='https://s3.amazonaws.com/text-datasets/'
                   'babi_tasks_1-20_v1-2.tar.gz')
    except:
        url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
        print(
            f'Error downloading dataset, please download it manually:\n'
            f'$ wget {url}\n'
            f'$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise

    return fpath


class Config:
    """
    Configuration variables used for the model build and model fit.
    """
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 64
    EPOCHS = 20


def build_model(story_max_len: int,
                query_max_len: int,
                embed_hidden_size: Config.EMBED_HIDDEN_SIZE) -> Model:
    """

    :param story_max_len: int. Needed for the sentence (story) input
    :param query_max_len: int. Needed for the question input.
    :param embed_hidden_size: int. The number of hidden layers to use for the
                                   RNN layers.
    :return: A keras Model (not sequential)
    """
    assert story_max_len > 0
    assert query_max_len > 0
    assert embed_hidden_size > 0

    print('Build model...')

    sentence = Input(shape=(story_max_len,),
                     dtype='int32',
                     name='sentence_input')

    encoded_sentence = Embedding(vocab_size,
                                 embed_hidden_size,
                                 name='sentence_embedding')(sentence)

    question = Input(shape=(query_max_len,),
                     dtype='int32',
                     name='question_input')

    encoded_question = Embedding(vocab_size,
                                 embed_hidden_size,
                                 name='question_embedding')(question)

    encoded_question = LSTM(embed_hidden_size,
                            name='lstm_question')(encoded_question)

    encoded_question = RepeatVector(story_max_len,
                                    name='lstm_question_3d')(encoded_question)

    merged = layers.add([encoded_sentence,
                         encoded_question])

    merged = LSTM(embed_hidden_size)(merged)

    merged = Dropout(0.3)(merged)

    predictions = Dense(vocab_size,
                        activation='softmax')(merged)

    model = Model([sentence, question], predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Done building model...')
    print(model.summary())

    return model


if __name__ == "__main__":
    # ### ETL Pipeline
    # fetch data if not present
    path = fetch_file()
    print(path)

    # Default QA1 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # QA1 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # QA2 with 1000 samples

    # #### Select spesific challenge
    challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    # QA2 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'

    with tarfile.open(path) as tar:
        train_extract = tar.extractfile(challenge.format('train'))
        text_extract = tar.extractfile(challenge.format('test'))

        train = get_stories(train_extract)
        print()
        test = get_stories(text_extract)

    # #### Build Vocabulary
    vocab = set()
    # unifying train and test to get all the tokens for story, question, answer
    for s, q, a in train + test:
        sample_tokens = [token.strip().lower() for token in s + q + [a]]
        vocab |= set(sample_tokens)

    vocab = sorted(vocab)
    print(len(vocab))

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1

    # creating a dictionary with words (tokens) as keys and index as value
    # (starting from 1)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    print(word_idx)

    # #### Calculate Maximum Length for the stories and the questions
    # calculates the max length of all the lengths of the tokens for each story
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))

    # calculates the max length of all the lengths of the tokens for each query
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    print(story_maxlen)
    print(query_maxlen)

    print(f'RNN: {Config.RNN} \n Embed: {Config.EMBED_HIDDEN_SIZE} '
          f'\n Sent: {Config.SENT_HIDDEN_SIZE} '
          f'\n Query: {Config.QUERY_HIDDEN_SIZE}')

    # #### Vectorize Data

    # In[14]:

    x, xq, y = vectorize_stories(data=train,
                                 word_to_idx=word_idx,
                                 story_max_len=story_maxlen,
                                 query_max_len=query_maxlen)

    # In[15]:

    tx, txq, ty = vectorize_stories(data=test,
                                    word_to_idx=word_idx,
                                    story_max_len=story_maxlen,
                                    query_max_len=query_maxlen)

    # In[16]:

    print(f'vocab = {vocab}', end='\n\n')

    print(f'Story_maxlen= {story_maxlen} \n'
          f'Query_maxlen = {query_maxlen}')

    print(f'X-Story.shape = {x.shape}')
    print(f'X-Question.shape = {xq.shape}')
    print(f'y-response.shape = {y.shape}')

    # ### Build Model

    # In[17]:

    qa_model = build_model(story_max_len=story_maxlen,
                           query_max_len=query_maxlen,
                           embed_hidden_size=Config.EMBED_HIDDEN_SIZE)

    # ### Train Model

    # In[18]:

    print('Training')
    qa_model.fit(
        [x, xq], y,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_split=0.05,
        verbose=2
    )

    # ### Evaluate Model

    # In[19]:

    loss, acc = qa_model.evaluate([tx, txq], ty,
                                  batch_size=Config.BATCH_SIZE)

    # In[20]:

    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
