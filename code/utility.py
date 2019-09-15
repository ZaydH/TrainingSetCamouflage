"""
utility functions for the camouflage experiments
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import os
import math


def select_instances(X, Y, indices):
    """
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of floats, the target values
    :param inidces: numpy 1D array of ints, the indices to extract
    """
    return X[indices], Y[indices]


def unpickle(file):
    """
    load the file
    :param file: string, path to the file
    :return dict(string, object): the dictionary loaded from file
    """
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
    return data_dict


def scale_features(cover_X, val_X, test_X):
    """
    scale the cover data to 0 and 1, apply the same transformation to the
    validation and test data
    :param cover_X: numpy 2D array of floats, the feature vectors for the
    cover data
    :param val_X: numpy 2D array of floats, the feature vectors for the
    validation data
    :param test_X: numpy 2D array of floats, the feature vectors for the
    test data
    :return numpy 2D array of floats, the scaled features of the cover data
            numpy 2D array of floats, the scaled features of the val data
            numpy 2D array of floats, the scaled features of the test data
    """
    '''
    dim_mins, dim_maxs = np.amin(cover_X, axis=0), np.amax(cover_X, axis=0)
    diffs = dim_maxs - dim_mins

    apply_transformation = lambda X, dim_mins, diffs: (X - dim_mins) / diffs

    return
        apply_transformation(cover_X, dim_mins, diffs),
        apply_transformation(val_X, dim_mins, diffs),
        apply_transformation(test_X, dim_mins, diffs)
    '''
    norms = np.sum(cover_X**2, axis=0)
    norms = np.sqrt(norms)
    apply_transformation = lambda X, square_sums: X / square_sums

    return apply_transformation(cover_X, norms), \
        apply_transformation(val_X, norms), \
        apply_transformation(test_X, norms)


def load_dataset(input_dir):
    """
    load dataset from a given directory
    :param input_dir: string, path to the input dir
    :return numpy 2D array of floats, the feature vectors for the cover data
            numpy 1D array of floats, the target values for the cover data
            numpy 2D array of floats, the feature vectors for the val data
            numpy 1D array of floats, the target values for the val data
            numpy 2D array of floats, the feature vectors for the test data
            numpy 1D array of floats, the target values for the test data
            numpy 2D array of floats, the kernel matrix for the cover data
    """
    # load the features
    cover_fname = os.path.join(input_dir, 'cover.npz')
    val_fname = os.path.join(input_dir, 'val.npz')
    test_fname = os.path.join(input_dir, 'test.npz')
    cover_data, val_data, test_data = \
        np.load(cover_fname), np.load(val_fname), np.load(test_fname)
    cover_X, cover_Y, val_X, val_Y, test_X, test_Y = \
        cover_data['X'], cover_data['Y'], val_data['X'], val_data['Y'], \
        test_data['X'], test_data['Y']

    # load the kernel matrix
    kernel_fname = os.path.join(input_dir, 'kernel.npy')
    kernel_matrix = np.load(kernel_fname)
    return cover_X, cover_Y, val_X, val_Y, test_X, test_Y, kernel_matrix


def load_data(input_fname):
    """
    load the dataset
    :param input_fname: string, path to npz file containing the dataset
    :return numpy 2D array of floats, the feautre vectors
            numpy 1D array of flotas, the corresponding target values
    """
    data = np.load(input_fname)
    return data['X'], data['Y']


def get_median_distance(X):
    """
    get the median distance between all pairs of points
    :param X: numpy 2D array of floats, the feature vectors
    :return float, the median distance
    """
    num_instances = len(X)
    num_distances = num_instances * (num_instances - 1) // 2
    distance_array = np.zeros(num_distances)
    distance_index = 0

    for i in range(num_instances):
        for j in range(num_instances):
            if i < j:
                distance_array[distance_index] = np.linalg.norm(X[i] - X[j])
                distance_index += 1

    distance_array = np.sort(distance_array)
    return distance_array[num_distances // 2]  # return the median


def get_max_distance(X):
    """
    get the maximum distance between point pairs
    :param X: numpy 2D array of floats, the feature vectors
    :return float, the maximum distnace
    """
    max_dist = 0
    num_instances = len(X)

    for i in range(num_instances):
        for j in range(num_instances):
            if i < j:
                dist = np.linalg.norm(X[i] - X[j])
                if dist > max_dist:
                    max_dist = dist

    return max_dist


def max_class_distance(X, Y):
    """
    get the maximum distance between point pairs within the same class
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the target values
    :return float, the maximum distance
    """
    pos_X, neg_X = [], []

    for index, y in enumerate(Y):
        if y == 1:
            pos_X.append(X[index])
        else:
            neg_X.append(X[index])

    pos_dist = get_max_distance(X)
    neg_dist = get_max_distance(X)

    if pos_dist > neg_dist:
        return pos_dist
    else:
        return neg_dist


def generate_kernel_matrix(X, Y, include_Y=True):
    """
    generate kernel matrix, chooses the bandwidth dynamically
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the target values
    :param include_Y: boolean, if True then add a scaled version of the target
    in features for kernel calculation, default True
    :return numpy 2D array of floats, the kernel matrix
    """
    sigma = get_median_distance(X)
    gamma = 1 / (2 * sigma**2)

    # include Y in kernel matrix
    if include_Y:
        max_class_dist = max_class_distance(X, Y)

        # set labels to either 0 or max_class_dist
        tempY = np.zeros(len(Y))
        for index, y in enumerate(Y):
            if y == 1:
                tempY[index] = max_class_dist
        kernel_X = np.hstack((X, tempY.reshape(-1, 1)))

        return rbf_kernel(kernel_X, gamma=gamma)

    # ignore Y in kernel matrix


def create_folds(X, Y, num_folds):
    """
    create folds for cross validation
    :param X: numpy 2D array of floats, the feature vectors
    :param Y: numpy 1D array of ints, the corresponding labels
    :param num_folds: int, number of folds
    :return list(numpy 2D array of floats), the folds for feature vectors
            list(numpy 1D array of ints), the corresponding labels
    """
    size = len(X)
    assert(size == len(Y))
    fold_size = math.ceil(size / num_folds)

    indices = np.array(range(size))
    np.random.shuffle(indices)

    fold_X, fold_Y = [], []

    for i in range(num_folds):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size

        if end_index > size:
            end_index = size

        fold_indices = indices[start_index: end_index]
        fold_X.append(X[fold_indices])
        fold_Y.append(Y[fold_indices])

    return fold_X, fold_Y


def create_train_test_set_from_folds(fold_X, fold_Y):
    """
    :param fold_X: list(numpy 2D array of floats), the folds for feature
    vectors
    :param fold_Y: list(numpy 1D array of ints), the corresponding labels
    :return list(numpy 2D array of floats), the list of training sets
            list(numpy 1D array of ints), corresponding list of labels
            list(numpy 2D array of floats), the list of test sets
            list(numpy 1D array of ints), corresponding list of labels
    """
    num_folds = len(fold_X)
    assert(num_folds == len(fold_Y))

    train_Xs, train_Ys, test_Xs, test_Ys = [], [], [], []

    for i in range(num_folds):
        test_X, test_Y = fold_X[i], fold_Y[i]

        train_X, train_Y = None, None
        for j in range(num_folds):
            if i == j:
                continue

            if train_X is None:
                train_X = fold_X[j]
                train_Y = fold_Y[j]
            else:
                train_X = np.vstack((train_X, fold_X[j]))
                train_Y = np.concatenate((train_Y, fold_Y[j]))

        train_Xs.append(train_X)
        train_Ys.append(train_Y)
        test_Xs.append(test_X)
        test_Ys.append(test_Y)

    return train_Xs, train_Ys, test_Xs, test_Ys
