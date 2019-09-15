"""
This file contains the different loss functions
"""
from __future__ import division, print_function, absolute_import
import numpy as np


def loss_logistic(learner, X, Y):
    """
    calculate the logistic loss
    :param learner: object, the learner
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the target values
    :return float, the loss
    """
    size = len(Y)
    assert(len(X) == size)
    weight_vector = learner.coef_[0]
    intercept = learner.intercept_

    loss = 0

    for i in range(size):
        loss += np.log((1 + np.exp(-Y[i] *
                                   (np.dot(weight_vector, X[i]) + intercept))))

    if isinstance(loss, np.ndarray):
        return loss[0] / size
    else:
        return loss / size


def loss_01(learner, X, Y):
    """
    calculate the 01 loss
    :param learner: object, the learner
    :param learner: object, the learner
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the target values
    :return float, the loss
    """
    size = len(Y)
    assert(len(X) == size)
    pY = learner.predict(X)
    mistake = 0

    for corr, pred in zip(Y, pY):
        if corr != pred:
            mistake += 1
    return mistake / size


def loss_lr_mse(learner, X, Y):
    """
    calculate the mean squared error for the logistic regression learner
    :param learner: sklearn.linear_model.LogisticRegression, the learner
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the target values
    :return float, the loss
    """
    size = len(Y)
    assert(len(X) == size)
    pY = learner.predict_proba(X)
    diff = pY - Y
    return np.sum(diff ** 2) / size
