# pip install tf-nightly-2.0-preview
from itertools import cycle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

# https://towardsdatascience.com/model-evaluation-techniques-for-classification-models-eac30092c38b

def build_binary_model(n_features: int = 20):
    """

    :param n_features:
    :return:
    """

    model = Sequential()
    model.add(Dense(100,
                    input_dim=n_features,
                    activation='relu'))
    model.add(Dense(100,
                    activation='relu'))

    model.add(Dense(1,
                    activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def build_multi_class_model(n_features: int = 20,
                            nb_classes: int = 3) -> Sequential:
    """

    :param n_features:
    :param nb_classes:
    :return:
    """
    model = Sequential()

    model.add(Dense(20,
                    input_dim=n_features,
                    activation='relu'))

    model.add(Dense(40,
                    activation='relu'))

    model.add(Dense(nb_classes,
                    activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def plot_multi_class_roc_auc_curves(nb_classes,
                                    y_true,
                                    y_pred_score,
                                    lw: int = 2):
    """
    ROC, AUC for a categorical classifier
    ROC curve extends to problems with three or more classes with what is known as the one-vs-all approach.
    For instance, if we have three classes, we will create three ROC curves,

    For each class, we take it as the positive class and group the rest classes jointly as the negative class.

    Class 1 vs classes 2&3
    Class 2 vs classes 1&3
    Class 3 vs classes 1&2

    :param nb_classes:
    :param y_true:
    :param y_pred_score:
    :param lw:
    :return:
    """

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"],
             tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink',
             linestyle=':',
             linewidth=4)

    plt.plot(fpr["macro"],
             tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy',
             linestyle=':',
             linewidth=4)

    colors = cycle(['aqua',
                    'darkorange',
                    'cornflowerblue'])

    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i],
                 tpr[i],
                 color=color,
                 lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1],
             [0, 1],
             'k--',
             lw=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.7, 1)

    plt.plot(fpr["micro"],
             tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink',
             linestyle=':',
             linewidth=4)

    plt.plot(fpr["macro"],
             tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy',
             linestyle=':',
             linewidth=4)

    colors = cycle(['aqua',
                    'darkorange',
                    'cornflowerblue'])

    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i],
                 tpr[i],
                 color=color,
                 lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1],
             [0, 1],
             'k--', lw=lw)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_binary_class_row_auc(y_true,
                              clf_names: list,
                              clfs_preds: list):
    """

    :param y_true: The true labels in one hot encoding
    :param clf_names: The names of the classifiers in order to plot
    :param clfs_preds: A list of numpy arrays, that contain predictions from various classifiers
    :return:
    """
    assert len(clf_names) == len(clfs_preds)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    for clf_name, preds in zip(clf_names, clfs_preds):
        fpr, tpr, thresholds = roc_curve(y_true,
                                         preds)

        # AUC value can also be calculated like this.

        auc_score = auc(fpr, tpr)

        plt.plot(fpr,
                 tpr,
                 label='{} (area = {:.3f})'.format(clf_name, auc_score))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.7, 1)
    plt.plot([0, 1],
             [0, 1],
             'k--')

    for clf_name, preds in zip(clf_names, clfs_preds):
        fpr, tpr, thresholds = roc_curve(y_true,
                                         preds)

        # AUC value can also be calculated like this.

        auc_score = auc(fpr, tpr)

        plt.plot(fpr,
                 tpr,
                 label='{} (area = {:.3f})'.format(clf_name, auc_score))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()


def print_confusion_matrix(y_true,
                           y_pred,
                           class_names: List[str],
                           figsize: Tuple[int, int] = (10, 7),
                           fontsize: int = 14) -> pd.DataFrame:
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heat-map.

    For something more extraordinary check this repo:
    https://github.com/wcipriano/pretty-print-confusion-matrix


    :param class_names:  An ordered list of class names
    :param figsize: A 2-long tuple, the first value determining the horizontal size of the outputted
                    figure, the second determining the vertical size. Defaults to (10,7).
    :param fontsize: Font size for axes labels. Defaults to 14.
    :return: The confusion matrix as a dataset
    """
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    df_cm = pd.DataFrame(conf_matrix,
                         index=class_names,
                         columns=class_names)

    fig = plt.figure(figsize=figsize)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0,
                                 ha='right',
                                 fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45,
                                 ha='right',
                                 fontsize=fontsize)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return df_cm


def run_binary_example():
    """

    :return:
    """

    X, y = make_classification(n_samples=100_000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    keras_model = build_binary_model()

    print('Fitting Keras Binary Model')
    keras_model.fit(X_train,
                    y_train,
                    epochs=15,
                    batch_size=128,
                    verbose=2)

    y_pred_keras = keras_model.predict(X_test).ravel()

    print('Fitting Random Forests CLF')
    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3,
                                n_estimators=50)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict_proba(X_test)[:, 1]

    print('Fitting Logistic Regression CLF')
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict_proba(X_test)[:, 1]

    plot_binary_class_row_auc(y_true=y_test,
                              clf_names=['Keras',
                                         'RandomF',
                                         'LogisticR'],
                              clfs_preds=[y_pred_keras,
                                          y_pred_rf,
                                          y_pred_lr])


def run_multi_class_example():
    """

    :return:
    """
    # 3 classes to classify
    n_classes = 3

    X, y = make_classification(n_samples=100_000,
                               n_features=20,
                               n_informative=3,
                               n_redundant=0,
                               n_classes=n_classes,
                               n_clusters_per_class=2)

    # Binarize the output
    lb = LabelBinarizer()

    y_hot = lb.fit_transform(y)

    n_classes = y_hot.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=0.25)

    keras_model2 = build_multi_class_model()

    keras_model2.fit(X_train,
                     y_train,
                     epochs=20,
                     batch_size=128,
                     verbose=2)

    y_score = keras_model2.predict(X_test)

    plot_multi_class_roc_auc_curves(nb_classes=n_classes,
                                    y_true=y_test,
                                    y_pred_score=y_score)

    y_pred_class = keras_model2.predict_classes(X_test)

    y_test_normal = lb.inverse_transform(y_test)

    print_confusion_matrix(y_true=y_test_normal,
                           y_pred=y_pred_class,
                           class_names=['0', '1', '2'])


if __name__ == "__main__":
    # run_binary_example()
    run_multi_class_example()
