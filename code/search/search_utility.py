"""
search utility functions for the camouflage experiments
"""
from __future__ import division, print_function, absolute_import
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import math


def calc_MMD(X, indices):
    """
    calculate the MMD_b for the cover instances selected for training
    :param X: numpy 2D array of floats, the kernel matrix for all cover data
    :param indices: list(int), the indices for the training set
    :return float, the MMD_b
    """
    m, n = len(X), len(indices)
    Y = X[np.ix_(indices, indices)]  # the kernel matrix for the training data
    # k(x_i, y_j), where x is the cover data and y is the training data
    XY = X[indices]
    return math.sqrt(np.sum(X) / (m**2) - 2 * np.sum(XY) / (m * n) +
                     np.sum(Y) / (n**2))


def calc_acceptance_region(m, n, K, alpha):
    """
    :param m: int, the size of the cover dataset
    :param n: int, the size of the training dataset
    :param K: float, maximum value of k(x, x')
    :param alpha: float, the level of the test
    :return float, the size of acceptance region
    """
    return 2 * (math.sqrt(K / m) + math.sqrt(K / n)) + \
        math.sqrt(2 * K * math.log(1 / alpha) * (1 / m + 1 / n))


def get_best_result(results):
    """
    finds the best result returned from a list of results returned by
    find_score_for_next_seq
    :param results: list(float, list(int)), the input score and indices tuples
    :return the best result
    """
    score_index = 0

    best_index = -1
    best_error = float('inf')

    for index, result in enumerate(results):
        if result[score_index] < best_error:
            best_error = result[score_index]
            best_index = index

    return results[best_index]


def get_best_result_index(results):
    """
    find the index of the best result returned from a list of results returned
    by find_score_for_next_seq
    :param results: list(float, list(int)), the input score and indices tuples
    :return int, index of the best result
    """
    score_index = 0

    best_index = -1
    best_error = float('inf')

    for index, result in enumerate(results):
        if result[score_index] < best_error:
            best_error = result[score_index]
            best_index = index

    return best_index


def check_mult_class(Y, pretrained):
    """
    Check if the selected instances for training have instances from multiple
    classes or not.
    Some learners require this. if the learner is alrady pretrained, then we
    don't need to worry.
    :param Y: numpy 1D array of floats, the target values for the instances
    :param pretrained: boolean, True if network is already pretrained false
    otherwise
    :return True, if the learner is pretrained or there are multiple classes
            in the selected indices for the candidate pool. False otherwise
    """
    if pretrained:
        # if pretrained then learner already knows about the different classes
        return True
    else:
        mult_class = False
        temp_class = Y[0]
        for y in Y[1:]:
            if y != temp_class:
                mult_class = True
                break

        return mult_class


def calc_MMD_original(X1, X2):
    """
    the orginal formulation for MMD
    :param X1: numpy 2D array of floats, the feature vectors of X1
    :param X2: numpy 2D array of floats, the feature vectors of X2
    :return float, the MMD_b
    """
    k_X1 = rbf_kernel(X1, gamma=1)
    k_X2 = rbf_kernel(X2, gamma=1)
    k_X1X2 = rbf_kernel(X1, X2, gamma=1)

    m, n = len(X1), len(X2)
    return math.sqrt(np.sum(k_X1) / m**2 + np.sum(k_X2) / n**2 -
                     2 * np.sum(k_X1X2) / (m * n))


def get_error(load_learner, fit_learner, learner_params, select_instances,
              indices, cover_X, cover_Y, target_X, target_Y, loss_function):
    """
    method to calculate the error of the learner on a set of neighbors. this
    method is outside the class to facilitate parallel processing
    :param load_learner: function pointer, loads a learner
    :param fit_learner: function pointer, fits a learner
    :param learner_params: dict(string, object), the parameters for
    load_learner
    :param select_instances: function pointer, used to select instances from
    the cover data given a list of indices
    :param indices: numpy 1D array of ints, indices to evaluate
    :param cover_X: the feature vectors for the cover data
    :param cover_Y: numpy 1D array of floats, the target values of the cover
    data
    :param target_X: the feature vectors for the target data
    :param target_Y: numpy 1D array of floats, the target values for the
    target data
    :param loss_function: funciton pointer, a function that takes the learner
    and target data as input to calculate a loss
    """
    learner = load_learner(learner_params)
    training_X, training_Y = select_instances(cover_X, cover_Y, indices)

    # check if multiclass or not
    mult_class = check_mult_class(training_Y, pretrained=False)

    if mult_class:
        learner = fit_learner(learner, training_X, training_Y, learner_params)
        error = loss_function(learner, target_X, target_Y)
    else:
        error = 1.0

    return [error, indices]


def get_all_errors(load_learner, fit_learner, learner_params, select_instances,
                   indices, cover_X, cover_Y, target_X, target_Y,
                   test_X, test_Y, loss_function, loss_01_function):
    """
    method to calculate the error of the learner on a set of neighbors.
    this method is outside the class to facilitate parallel processing
    :param load_learner: function pointer, loads a learner
    :param fit_learner: function pointer, fits a learner
    :param learner_params: dict(string, object), the parameters for
    load_learner
    :param select_instances: function pointer, used to select instances from
    the cover data given a list of indices
    :param indices: numpy 1D array of ints, indices to evaluate
    :param cover_X: the feature vectors for the cover data
    :param cover_Y: numpy 1D array of floats, the target values of the cover
    data
    :param target_X: the feature vectors for the target data
    :param target_Y: numpy 1D array of floats, the target values for the
    target data
    :param test_X: the feature vectors for the test data
    :param test_Y: numpy 1D array of floats, the target values for the test
    data
    :param loss_function: funciton pointer, a function that takes the learner
    and target data as input to calculate a loss
    :param loss_01_function: function pointer, gives 01 error on data
    """
    learner = load_learner(learner_params)
    training_X, training_Y = select_instances(cover_X, cover_Y, indices)
    # count positive and negative instances
    pos_count = 0
    for y in training_Y:
        if y == 1:
            pos_count += 1

    neg_count = len(training_Y) - pos_count

    # check if multiclass or not
    mult_class = check_mult_class(training_Y, pretrained=False)

    if mult_class:
        learner = fit_learner(learner, training_X, training_Y, learner_params)
        error = loss_function(learner, target_X, target_Y)
        target_01_error = loss_01_function(learner, target_X, target_Y)
        test_error = loss_function(learner, test_X, test_Y)
        test_01_error = loss_01_function(learner, test_X, test_Y)
        cover_error = loss_function(learner, cover_X, cover_Y)
        cover_01_error = loss_01_function(learner, cover_X, cover_Y)
    else:
        error = 1.0

    return [error, indices, pos_count, neg_count, target_01_error,
            test_error, test_01_error, cover_error, cover_01_error]
