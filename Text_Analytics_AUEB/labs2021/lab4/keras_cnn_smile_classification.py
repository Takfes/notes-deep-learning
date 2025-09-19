# ! pip install -U scikit-image

import shutil
from io import BytesIO
from pathlib import Path
from typing import Tuple, List

import IPython.display
import PIL.Image
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import block_reduce
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.utils import np_utils


def load_files() -> Tuple[List[Path], List[Path]]:
    """
    This function discovers the positive and negative file paths for our task.
    :return: Returns a tuple with the positive and negative paths
    """

    # neg_path = 'SMILEsmileD-master/SMILEs/negatives/negatives7/'  # does not work on all OS

    # works on every OS. Old way using os.path.join()

    # neg_path = os.path.join('SMILEsmileD-master',
    #                         'SMILEs',
    #                         'negatives',
    #                         'negatives7')

    # new way by using pathlib module.
    neg_path = Path.cwd().joinpath('SMILEsmileD-master',
                                   'SMILEs',
                                   'negatives',
                                   'negatives7')

    print("Negative path: {}".format(neg_path))

    # pos_path = 'SMILEsmileD-master/SMILEs/positives/positives7/'

    pos_path = Path.cwd().joinpath('SMILEsmileD-master',
                                   'SMILEs',
                                   'positives',
                                   'positives7')

    print("Positive path: {}".format(pos_path), end='\n\n')

    print('Loading Negative image paths')

    # negative_paths = glob.glob(os.path.join(neg_path, '*.jpg'))
    neg_paths = list(neg_path.glob('*.jpg'))

    print('Loaded {} Negative image examples'.format(len(neg_paths)), end='\n\n')

    print('Loading Positive image paths')
    # positive_paths = glob.glob(os.path.join(neg_path, '*.jpg'))
    pos_paths = list(pos_path.glob('*.jpg'))

    print('Loaded {} Positive image examples'.format(len(pos_paths)))

    return neg_paths, pos_paths


def examples_to_dataset(img_paths: List[Path],
                        labels: List[int],
                        block_size: int = 2,
                        as_gray: bool = True):
    """
    This function, given the img_paths loads the images from disk.
    Also, it reduces the images size by under-sampling the pixels


    block_size:
    1: same size.
    2: undersample by 2

    :param img_paths: A list of Paths that define the locations of our images
    :param labels: The labels of our images
    :param block_size: Int. 1 stay unchanged. 2 subsample by 2 etc
    :param as_gray:
    :return:
    """
    assert len(img_paths) == len(labels)

    X = []
    y = []

    for path, label in zip(img_paths, labels):
        # reads the image from the filepath
        img = imread(str(path),
                     as_gray=as_gray)

        # reduces the image size by x times by taking the mean of the pixels.
        img = block_reduce(img,
                           block_size=(block_size,
                                       block_size),
                           func=np.mean)

        X.append(img)

        y.append(label)

    return np.asarray(X), np.asarray(y)


def find_rectangle(n,
                   max_ratio=2):
    """

    :param n:
    :param max_ratio:
    :return:
    """

    sides = []
    square = int(math.sqrt(n))

    for w in range(square, max_ratio * square):
        h = n / w
        used = w * h
        leftover = n - used
        sides.append((leftover, (w, h)))

    return sorted(sides)[0][1]


def make_mosaic(images: np.ndarray,
                n=None,
                nx=None,
                ny=None,
                w=None,
                h=None):
    """
    Creates a mosaic of images for demonstration purposes.

    Should work for 1d and 2d images,
    assumes images are square but can be overwritten

    :param images:
    :param n:
    :param nx:
    :param ny:
    :param w:
    :param h:
    :return:
    """

    if n is None and nx is None and ny is None:

        nx, ny = find_rectangle(len(images))

    else:
        nx = n if nx is None else nx
        ny = n if ny is None else ny

    images = np.array(images)

    if images.ndim == 2:  # grey scale. Only one channel

        side = int(np.sqrt(len(images[0])))

        h = side if h is None else h
        w = side if w is None else w

        images = images.reshape(-1, h, w)

    else:
        h = images.shape[1]
        w = images.shape[2]

    image_gen = iter(images)

    mosaic = np.empty((h * ny, w * nx))

    for i in range(ny):

        ia = (i) * h
        ib = (i + 1) * h

        for j in range(nx):
            ja = j * w
            jb = (j + 1) * w

            mosaic[ia:ib, ja:jb] = next(image_gen)

    return mosaic


def show_array(a,
               fmt='png',
               filename=None):
    """

    :param a:
    :param fmt:
    :param filename:
    :return:
    """

    a = np.squeeze(a)
    a = np.uint8(np.clip(a, 0, 255))

    image_data = BytesIO()

    PIL.Image.fromarray(a).save(image_data, fmt)

    if filename is None:
        IPython.display.display(IPython.display.Image(data=image_data.getvalue()))
        plt.show()
    else:

        with open(filename, 'w') as f:
            image_data.seek(0)
            shutil.copyfileobj(image_data, f)


def print_indicator(data,
                    model,
                    class_names,
                    bar_width=50):
    """

    :param data:
    :param model:
    :param class_names:
    :param bar_width:
    :return:
    """

    probabilities = model.predict(np.array([data]), verbose=0)[0]

    print(probabilities)

    left_count = int(probabilities[1] * bar_width)

    right_count = bar_width - left_count

    left_side = '-' * left_count

    right_side = '-' * right_count

    print(class_names[0], left_side + '###' + right_side, class_names[1])

    print("{} {}###{} {}".format(class_names[0], left_side, right_side, class_names[1]))


def build_model(
        x,
        n_classes,
        n_filters: int = 32,
        n_pool: int = 2,
        n_conv: int = 3,
        dr: float = 0.25, ):
    """

    :param x:
    :param n_classes:
    :param n_filters:
    :param n_pool:
    :param n_conv:
    :param dr:
    :return:
    """
    assert n_classes == 2

    seq_model = Sequential()

    seq_model.add(Conv2D(n_filters,
                         (n_conv, n_conv),
                         activation='relu',
                         input_shape=x.shape[1:]))

    seq_model.add(Conv2D(n_filters,
                         (n_conv, n_conv),
                         activation='relu'))

    seq_model.add(MaxPooling2D(pool_size=(n_pool,
                                          n_pool)))

    seq_model.add(Dropout(dr))
    seq_model.add(Flatten())

    seq_model.add(Dense(128,
                        activation='relu'))

    seq_model.add(Dropout(dr))

    seq_model.add(Dense(n_classes,
                        activation='softmax'))

    seq_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    print(seq_model.summary())

    return seq_model


if __name__ == "__main__":
    negative_paths, positive_paths = load_files()

    # if not os.path.exists('SMILEsmileD-master'):
    #     !curl - L - O
    #     https: // github.com / hromi / SMILEsmileD / archive / master.zip
    #     !unzip - q
    #     master.zip
    #     !rm
    #     master.zip
    #
    #     print('Done')

    image_paths = negative_paths + positive_paths
    image_labels = [0] * len(negative_paths) + [1] * len(positive_paths)

    print(pd.DataFrame({'img_paths': image_paths,
                        'img_labels': image_labels}).sample(5))

    X, y = examples_to_dataset(img_paths=image_paths,
                               labels=image_labels,
                               block_size=2,
                               as_gray=True)

    # Converting to floats and normalizing the images
    X = X.astype(np.float32) / 255.

    # Converting the labels to integers
    y = y.astype(np.int32)

    print(X.dtype, X.min(), X.max(), X.shape)

    print(y.dtype, y.min(), y.max(), y.shape)

    show_array(255 * make_mosaic(X[:len(negative_paths)], n=15), fmt='jpeg')  # negative at the beginning

    show_array(255 * make_mosaic(X[-len(positive_paths):], n=15), fmt='jpeg')  # positive at the end

    print(X.shape)

    # expanding the dimensions in order to be able to fit it in a CNN model.
    X = np.expand_dims(X, axis=-1)

    print(X.shape)

    np.save('X.npy', X)
    np.save('y.npy', y)

    # load the data
    X = np.load('X.npy')
    y = np.load('y.npy')

    # convert classes to vector
    nb_classes = 2

    y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

    # shuffle all the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    print('Getting shuffled indices: {}'.format(indices))

    # shuffling X and Y
    X = X[indices]
    y = y[indices]

    # prepare weighting for classes since they're unbalanced
    class_totals = y.sum(axis=0)
    class_weight = class_totals.max() / class_totals

    print('X | Type: {}, Min: {}, Max: {}, Shape{}'.format(X.dtype, X.min(), X.max(), X.shape))
    print('y | Type: {}, Min: {}, Max: {}, Shape{}'.format(y.dtype, y.min(), y.max(), y.shape))

    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    smile_model = build_model(x=X,
                              n_classes=nb_classes,
                              n_filters=nb_filters,
                              n_pool=nb_pool,
                              n_conv=nb_conv,
                              dr=0.25)

    validation_split = 0.10

    smile_model.fit(X,
                    y,
                    batch_size=128,
                    class_weight=class_weight,
                    epochs=15,
                    verbose=1,
                    validation_split=validation_split)

    open('model.json', 'w').write(smile_model.to_json())
    smile_model.save_weights('weights.h5')

    plt.plot(smile_model.history.history['loss'])
    plt.plot(smile_model.history.history['val_loss'])
    plt.show()

    plt.plot(smile_model.history.history['accuracy'])
    plt.plot(smile_model.history.history['val_accuracy'])
    plt.show()

    n_validation = int(len(X) * validation_split)

    y_predicted = smile_model.predict(X[-n_validation:])

    print(roc_auc_score(y[-n_validation:],
                        y_predicted))

    smile_model = model_from_json(open('model.json').read())
    smile_model.load_weights('weights.h5')

    X = np.load('X.npy')
    class_names = ['Neutral', 'Smiling']

    img = X[-5]

    show_array(255 * img)

    print_indicator(img, smile_model, class_names)

    img = X[10]

    show_array(255 * img)

    print_indicator(img, smile_model, class_names)
