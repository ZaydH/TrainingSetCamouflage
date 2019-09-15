"""
the base search class for the camouflage experiments
"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta
from abc import abstractmethod
import numpy as np


class SearchBase:
    """Abstract base class for all the search classes"""

    __metaclass__ = ABCMeta

    think_key = 'thinking_indices'
    opt_key = 'opt_indices'
    error_key = 'opt_error'
    mmd_key = 'mmd'
    finish_key = 'search_finished'
    acceptance_region_key = 'acceptance_region'
    rejected_key = 'rejected'
    time_key = 'time'

    def __init__(self, load_learner, fit_learner, learner_params,
                 select_instances, cover_X, cover_Y, target_X, target_Y,
                 kernel_matrix, max_K, loss_function, thinking_budget=100000,
                 alpha=0.05):
        """
        :param load_learner: function pointer, loads a learner
        :param fit_learner: function pointer, fits a learner
        :param learner_params: dict(string, object), the parameters for
        load_learner
        :param select_instances: function pointer, used to select instances
        from the cover data given a list of indices
        :param cover_X: the feature vectors for the cover data
        :param cover_Y: numpy 1D array of floats, the target values of the
        cover data
        :param target_X: the feature vectors for the target data
        :param target_Y: numpy 1D array of floats, the target values for the
        target data
        :param kernel_matrix: numpy 2D array of floats, the kernel matrix for
        the cover feature values
        :param max_K: float, the maximum possible value of the kernel
        :param loss_function: funciton pointer, a function that takes the
        learner and target data as input to calculate a loss
        :param thinking_budget: int, the maximum number of times the learner
        will be trained
        :param alpha: float,  between 0 and 1, the level of the hypothesis test
        """
        assert(thinking_budget > 0)
        assert(0 < alpha < 1)

        self.load_learner = load_learner
        self.fit_learner = fit_learner
        self.learner_params = learner_params
        self.select_instances = select_instances
        self.cover_X = cover_X
        self.cover_Y = cover_Y
        self.target_X = target_X
        self.target_Y = target_Y
        self.kernel_matrix = kernel_matrix
        self.loss_function = loss_function
        self.thinking_budget = thinking_budget
        self.alpha = alpha

    @abstractmethod
    def search_optimum_teaching_seq(self):
        """
        :return dictionary(string, object), a dictionary containing best
                teaching set (as list of indices) for each
                thinking budget, the optimum teaching set (as list indices)
                and if search finished or not
        """
        return

    @staticmethod
    def get_pos_neg_indices(Y):
        """
        get the index of positive and negative instances given the target
        values
        :param Y: numpy 1D array of floats, the target values
        :return numpy 1D array of ints, the indices for positive instances
                numpy 1D array of ints, the indices for negative instances
        """
        positive_indices = []
        negative_indices = []
        for index, y in enumerate(Y):
            if y == 1:
                positive_indices.append(index)
            else:
                negative_indices.append(index)

        return np.array(positive_indices).astype(int), \
            np.array(negative_indices).astype(int)
