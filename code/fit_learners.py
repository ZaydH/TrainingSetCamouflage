"""
utility script containing the fit functions for the learners
"""
from __future__ import division, print_function, absolute_import


def fit_lr_learner(learner, X, Y, learner_params):
    """
    call the fit function for a logistic regression learner
    :param learner: LogisticRegression, the learner
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the target values
    :param learner_params: dict(string, object), the parameters for fit
    :return LogisticRegression
    """
    learner.fit(X, Y)
    return learner
