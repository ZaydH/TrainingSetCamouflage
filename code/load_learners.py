"""
utility script containing the load functions for the learners
"""
from __future__ import division, print_function, absolute_import


def load_lr_learner(learner_params):
    """
    create a logistic regression learner
    :param learner_params: dict(string, object), the parameters for
    load_learner
    :return LogisticRegression
    """
    from sklearn.linear_model import LogisticRegression
    reg_param_key = 'reg_param'

    if reg_param_key not in learner_params:
        C = 1  # default value
    else:
        C = 1 / learner_params[reg_param_key]
    return LogisticRegression(random_state=learner_params['random_state'],
                              fit_intercept=learner_params['fit_intercept'],
                              C=C)
