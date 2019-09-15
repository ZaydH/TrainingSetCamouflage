"""
Code to search for a teaching sequence using BeamSearch for logistic regression
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from search.BeamSearch import BeamSearch
from search.loss_functions import loss_01, loss_logistic
from argparse import ArgumentParser
from load_learners import load_lr_learner
from fit_learners import fit_lr_learner
from utility import load_dataset, select_instances
import os


def experiment_BeamSearch(input_dir, output_fname, train_size, thinking_budget,
                          beam_width, children_count, loss_type,
                          seed, homogeneous=False, print_vector=False,
                          num_cpus=1):
    """
    perform beam search on a given dataset using logistic
    regression learner
    :param input_dir: string, path to the input directory
    :param output_fname: string, path to the output file
    :param train_size: int, size of the train set
    :param thinking_budget: int, the thinking budget
    :param beam_width: int, the width of the beam
    :param children_count: int, number of children to consider for each
    sequence
    :param loss_type: string, type of loss to use
    :param seed: int, the random seed to use
    :param homogeneous: boolean, if the learner is homogeneous or not,
    default False
    :param print_vector: boolean, if True then print the final vector,
    default False
    :param num_cpus: int, the number of CPUs to use
    """
    # load the data
    cover_X, cover_Y, val_X, val_Y, test_X, test_Y, kernel_matrix = \
        load_dataset(input_dir)

    loss_function = loss_01 if loss_type == '01' else loss_logistic

    # parameters
    max_K, alpha = 1, 0.05
    learner_params = {'random_state': seed, 'fit_intercept': not homogeneous}

    search_alg = BeamSearch(load_lr_learner, fit_lr_learner, learner_params,
                            select_instances, cover_X, cover_Y,
                            val_X, val_Y, kernel_matrix, max_K, loss_function,
                            train_size, beam_width, children_count,
                            thinking_budget=thinking_budget, alpha=alpha,
                            output_fname=output_fname,
                            seed=seed, num_cpus=num_cpus)
    results = search_alg.search_optimum_teaching_seq()

    print('\n\nInput Arguments')
    print('Input Dir: {}'.format(input_dir))
    print('Output File: {}'.format(output_fname))
    print('Size of train set: {}'.format(train_size))
    print('Thinking budget: {}'.format(thinking_budget))
    print('Beam width: {}'.format(beam_width))
    print('Children count: {}'.format(children_count))
    print('Loss Function: {}'.format(loss_type))
    print('Seed: {}'.format(seed))
    print('Homogeneous: {}'.format(homogeneous))
    print('Print Vector: {}'.format(print_vector))

    print('\n\nOutput')
    print('Length of optimal sequence: {}'.format(len(results['opt_indices'])))

    # get the error on after training
    learner = load_lr_learner(learner_params)
    opt_indices = results['opt_indices']
    train_X, train_Y = select_instances(cover_X, cover_Y, opt_indices)
    learner.fit(train_X, train_Y)
    cover_error, val_error, test_error = \
        1.0 - learner.score(cover_X, cover_Y), \
        1.0 - learner.score(val_X, val_Y), \
        1.0 - learner.score(test_X, test_Y)
    print('01 loss: Cover error: {}, Validation error: {}, Test error: {}'.format(cover_error, val_error, test_error))
    cover_error, val_error, test_error = \
        loss_logistic(learner, cover_X, cover_Y), \
        loss_logistic(learner, val_X, val_Y), \
        loss_logistic(learner, test_X, test_Y)
    print('logistic loss: Cover error: {}, Validation error: {}, Test error: {}'.format(cover_error, val_error,
                                                                                        test_error))

    # count positive and negative instances
    pos_count = 0
    for y in train_Y:
        if y == 1:
            pos_count += 1
    print('Pos count: {}, Neg count: {}'.format(pos_count,
                                                len(train_Y) - pos_count))

    # print vectors
    if print_vector:
        if not homogeneous:
            weights = np.zeros(len(learner.coef_[0]) + 1)
            weights[0] = learner.intercept_[0]
            weights[1:] = learner.coef_[0]
            print('Weights: {}'.format(weights))
        else:
            print('Weights: {}'.format(learner.coef_[0]))


def main():
    """the main method"""
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='The directory containing the input data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path prefix to the output file')
    parser.add_argument('-n', '--train_size', type=int, default=100,
                        help='Size of the training set to use')
    parser.add_argument('-b', '--thinking_budget', type=int, default=100000,
                        help='Number of learners to train for search')
    parser.add_argument('-w', '--beam_width', type=int, default=10,
                        help='Width of the beam')
    parser.add_argument('--children', type=int, default=50,
                        help='Number of children for each sequence')
    parser.add_argument('-l', '--loss', type=str, choices=['01', 'logistic'],
                        default='01', help='The loss function')
    parser.add_argument('-s', '--seed', type=int, default=1234,
                        help='The random seed')
    parser.add_argument('--homogeneous', action='store_true', default=False,
                        help='Flag for homogeneous logistic regression')
    parser.add_argument('-c', '--cpus', type=int, default=1,
                        help='The number of cpus to use, default 1')
    parser.add_argument('-v', '--print_vector', action='store_true',
                        default=False, help='Flag to print the final vector')
    args = parser.parse_args()

    # fix the name of the output file
    homogen = 'inhomogen'
    if args.homogeneous:
        homogen = 'homogen'

    output = '{}_{}_{}_{}_{}_{}_{}_{}_lr.pkl'.format(args.output, args.seed,
                                                     homogen, args.loss,
                                                     args.train_size,
                                                     args.thinking_budget,
                                                     args.beam_width,
                                                     args.children)

    print('Arguments')
    print('Input Dir: {}'.format(args.input_dir))
    print('Output File: {}'.format(output))
    print('Size of train set: {}'.format(args.train_size))
    print('Thinking budget: {}'.format(args.thinking_budget))
    print('Beam width: {}'.format(args.beam_width))
    print('Children count: {}'.format(args.children))
    print('Loss Function: {}'.format(args.loss))
    print('Seed: {}'.format(args.seed))
    print('Homogeneous: {}'.format(args.homogeneous))
    print('Print Vector: {}'.format(args.print_vector))
    print('CPUs: {}'.format(args.cpus))

    experiment_BeamSearch(args.input_dir, output, args.train_size,
                          args.thinking_budget, args.beam_width, args.children,
                          args.loss, args.seed, homogeneous=args.homogeneous,
                          print_vector=args.print_vector, num_cpus=args.cpus)


if __name__ == '__main__':
    main()
