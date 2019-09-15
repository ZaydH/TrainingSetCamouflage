"""
UniformSampling search class
"""
from .SearchBase import SearchBase
from .search_utility import calc_acceptance_region, calc_MMD, \
    get_best_result_index, get_error
from joblib import Parallel, delayed
import numpy as np
import time
import pickle


class UniformSampling(SearchBase):
    """Random search which outputs the best found result as output"""

    def __init__(self, load_learner, fit_learner, learner_params,
                 select_instances, cover_X, cover_Y, target_X,
                 target_Y, kernel_matrix, max_K, loss_function, size,
                 thinking_budget=100000, num_iterations=1, alpha=0.5,
                 output_fname=None, seed=1234, num_cpus=1):
        """
        :param load_learner: function pointer, loads a learner
        :param fit_learner: function pointer, fits a learner
        :param learner_params: dict(string, object), the parameters for the
        learner (both load and fit)
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
        :param size: int, size of the training sets to train
        :param thinking_budget: int, the maximum number of times the learner
        will be trained
        :param num_iterations: int, the number of times to divide the
        thinking budget
        :param alpha: float,  between 0 and 1, the level of the hypothesis test
        :param output_fname: string, path to file where the results will be
        saved
        :param seed: int, the random seed
        :param num_cpus: int, number of cpus to use for parallel processing
        """
        SearchBase.__init__(self, load_learner, fit_learner, learner_params,
                            select_instances, cover_X, cover_Y,
                            target_X, target_Y, kernel_matrix, max_K,
                            loss_function, thinking_budget=thinking_budget,
                            alpha=alpha)
        self.seed = seed
        np.random.seed(seed)
        self.num_cpus = num_cpus

        self.size = size
        self.num_iterations = num_iterations

        self.acceptance_region = calc_acceptance_region(len(cover_Y), size,
                                                        max_K, alpha)

        assert(output_fname is not None)
        self.output_fname = output_fname
        self.rejected = 0  # variable to store how many instances were rejected

    def generate_random_seqs(self, num):
        """
        generate a random training sequence
        :param num: int, number of random sequences to generate
        :return
            numpy 2D array of ints, the random sequences
            numpy 1D array of floats, the correspoinding mmds
        """
        random_seqs = np.zeros((num, self.size)).astype(int)
        random_mmds = np.zeros(num)
        cand_size = len(self.cover_Y)

        for i in range(num):
            random_seqs[i] = np.random.choice(list(range(cand_size)),
                                              size=self.size, replace=False)
            random_mmds[i] = calc_MMD(self.kernel_matrix, random_seqs[i])

            # repeat until mmd is in range
            while random_mmds[i] > self.acceptance_region:
                self.rejected += 1
                random_seqs[i] = np.random.choice(list(range(cand_size)),
                                                  size=self.size,
                                                  replace=False)
                random_mmds[i] = calc_MMD(self.kernel_matrix, random_seqs[i])

        return random_seqs, random_mmds

    def search_optimum_teaching_seq(self):
        """
        use random argmin search to find the optimum teaching set
        :return
            dictionary(string, object), a dictionary containing best
            teaching set (as list of indices) for each
            thinking budget, the optimum teaching set (as list indices)
            and if search finished or not
        """
        # assuming the division returns an integer
        iteration_budget = self.thinking_budget // self.num_iterations
        thinking = 0

        return_dict = {self.think_key: {}, self.finish_key: False,
                       self.acceptance_region_key: self.acceptance_region,
                       self.time_key: None}
        pool = Parallel(n_jobs=self.num_cpus)
        msg = 'Contains the results for uniform sampling search, saved ' + \
            'on {}.'.format(time.strftime('%m/%d/%Y %H:%M:%S'))

        start_time = time.time()
        opt_error, opt_seq = float('inf'), None

        for it in range(self.num_iterations):
            seqs, mmds = self.generate_random_seqs(iteration_budget)
            results = pool(delayed(get_error)(self.load_learner,
                           self.fit_learner, self.learner_params,
                           self.select_instances, seq,
                           self.cover_X, self.cover_Y,
                           self.target_X, self.target_Y,
                           self.loss_function) for seq in seqs)

            best_index = get_best_result_index(results)
            curr_opt_error, curr_opt_seq = results[best_index]
            thinking += iteration_budget

            if curr_opt_error < opt_error:
                opt_error, opt_seq, opt_mmd = \
                    curr_opt_error, curr_opt_seq, mmds[best_index]

                return_dict[self.think_key][thinking] = \
                    [opt_error, opt_mmd, opt_seq, time.time() - start_time]
                return_dict[self.error_key], return_dict[self.mmd_key], \
                    return_dict[self.opt_key] = opt_error, opt_mmd, opt_seq
                return_dict[self.rejected_key] = self.rejected
                return_dict[self.time_key] = time.time() - start_time

                msg = 'Contains the results for uniform sampling search, ' + \
                    'saved on {}.'.format(time.strftime('%m/%d/%Y %H:%M:%S'))
                pickle.dump([msg, return_dict], open(self.output_fname, 'wb'))

            print('Time to finish iteration {}: {}'.format(it + 1,
                                                           time.time()
                                                           - start_time))

        return_dict[self.finish_key] = True
        return_dict[self.time_key] = time.time() - start_time
        msg = 'Contains the results for uniform sampling search, saved ' + \
            'on {}.'.format(time.strftime('%m/%d/%Y %H:%M:%S'))
        pickle.dump([msg, return_dict], open(self.output_fname, 'wb'))

        return return_dict
