"""
BeamSearch search class
"""
from .SearchBase import SearchBase
from .search_utility import calc_acceptance_region, calc_MMD, \
    get_best_result_index, get_error
from joblib import Parallel, delayed
import numpy as np
import time
import pickle


class BeamSearch(SearchBase):
    """Class for BeamSearch: Rich and Knight (1991)"""

    def __init__(self, load_learner, fit_learner, learner_params,
                 select_instances, cover_X, cover_Y, target_X, target_Y,
                 kernel_matrix, max_K, loss_function, size, beam_width,
                 children_count, thinking_budget=100000, alpha=0.05,
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
        :param beam_width: int, width of the beam
        :param children_count: int, number of children to consider for each
        node
        :param thinking_budget: int, the maximum number of times the learner
        will be trained
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

        assert(size > 0 and beam_width > 0 and children_count > 0)
        self.size = size
        self.beam_width = beam_width
        self.children_count = children_count
        self.acceptance_region = calc_acceptance_region(len(cover_Y), size,
                                                        max_K, alpha)

        assert(output_fname is not None)
        self.output_fname = output_fname
        self.rejected = 0  # variable to store how many instances were rejected

    def generate_init_seqs(self):
        """
        generate a set of initial training sequences randomly
        :return numpy 2D array of ints, the inital sequences
                numpy 1D array of floats, the corresponding mmds
        """
        init_seqs = np.zeros((self.beam_width, self.size)).astype(int)
        init_mmds = np.zeros(self.beam_width)
        cand_size = len(self.cover_Y)

        # considering sets only for now
        for i in range(self.beam_width):
            init_seqs[i] = np.random.choice(list(range(cand_size)),
                                            size=self.size, replace=False)
            init_mmds[i] = calc_MMD(self.kernel_matrix, init_seqs[i])

            # repeat until mmd is in range
            while init_mmds[i] > self.acceptance_region:
                self.rejected += 1
                init_seqs[i] = np.random.choice(list(range(cand_size)),
                                                size=self.size, replace=False)
                init_mmds[i] = calc_MMD(self.kernel_matrix, init_seqs[i])

        return init_seqs, init_mmds

    def generate_child(self, sequence):
        """
        generate a single random child for one sequence
        :param sequence: numpy 1D array of int, the parent sequence
        :return numpy 1D array of ints, the randomly generated child
                float, mmd of generated child
        """
        random_pos = np.random.randint(self.size)

        # considering set
        cand_pool = list(range(len(self.cover_Y)))
        cand_pool = list(set(cand_pool).difference(sequence))
        cand_index = np.random.choice(cand_pool)

        child = np.copy(sequence)
        child[random_pos] = cand_index

        child_mmd = calc_MMD(self.kernel_matrix, child)

        return child, child_mmd

    def generate_seq_children(self, sequence):
        """
        generate children for one sequence
        :param sequence: numpy 1D array of int, the parent sequence
        :return numpy 2D array of ints, the children
                numpy 1D array of floats, the corresponding mmds
        """
        children = np.zeros((self.children_count, self.size)).astype(int)
        children_mmds = np.zeros(self.children_count)

        for i in range(self.children_count):
            children[i], children_mmds[i] = self.generate_child(sequence)

        return children, children_mmds

    def generate_children(self, sequences):
        """
        generate children for a list of sequences
        :param sequences: numpy 2D array of ints, the sequences
        :return numpy 2D array of ints, the children
                numpy 1D array of floats, the corresponding mmds
        """
        total_count = len(sequences) * self.children_count
        children = np.zeros((total_count, self.size)).astype(int)
        children_mmds = np.zeros(total_count)

        for index, sequence in enumerate(sequences):
            children[index * self.children_count: (index + 1) *
                     self.children_count], \
                children_mmds[index * self.children_count: (index + 1) *
                              self.children_count] \
                = self.generate_seq_children(sequence)

        return children, children_mmds

    def filter_children(self, children, children_mmds):
        """
        filter chilredn according to mmd values
        :param children: numpy 2D array of ints, the children
        :param children_mmds: numpy 1D array of floats, the corresponding mmds
        :return numpy 2D array of ints, the filtered children
                numpy 1D array of floats, the corresponding mmds
        """
        filtered_children, filtered_children_mmds = [], []

        for i in range(len(children_mmds)):
            if children_mmds[i] < self.acceptance_region:
                self.rejected += 1
                filtered_children.append(children[i])
                filtered_children_mmds.append(children_mmds[i])

        return np.array(filtered_children), np.array(filtered_children_mmds)

    def prune(self, parent_results, children_results):
        """
        select a random subset of parents and children based on their error
        :param parent_results: list([error, numpy 1D array of ints]), the
        results for parents
        :param children_results: list([error, numpy 1D array of ints]), the
        results for children
        :return numpy 2D array of ints, the sequences kept for next round
                list([error, numpy 1D array of ints]), the corresponding
                results
        """
        error_index, seq_index = 0, 1
        all_results = parent_results + children_results

        # remove repeated sequences
        seq_dict, temp = {}, []

        for result in all_results:
            if tuple(result[seq_index]) in seq_dict:
                continue
            else:
                seq_dict[tuple(result[seq_index])] = 1
                temp.append(result)
        all_results = temp

        pruned_seqs = np.zeros((self.beam_width, self.size)).astype(int)
        pruned_results = []

        errors = np.array([result[error_index] for result in all_results])
        indices = np.argsort(errors)[:self.beam_width]

        for i in range(self.beam_width):
            pruned_seqs[i] = all_results[indices[i]][seq_index]
            pruned_results.append(all_results[indices[i]])

        return pruned_seqs, pruned_results

    def search_optimum_teaching_seq(self):
        """
        use beam search to find the optimum teaching set
        :return dictionary(string, object), a dictionary containing best
                teaching set (as list of indices) for each
                thinking budget, the optimum teaching set (as list indices)
                and if search finished or not
        """
        start_time = time.time()
        # calculate how many times we have trained the learner
        thinking = 0

        return_dict = {self.think_key: {}, self.finish_key: True,
                       self.acceptance_region_key: self.acceptance_region}
        pool = Parallel(n_jobs=self.num_cpus)
        print(self.num_cpus)

        # generate the first set of training sets
        print('Generating initial sequences')
        seqs, mmds = self.generate_init_seqs()

        # evaluate the initial seqs
        results = pool(delayed(get_error)(self.load_learner, self.fit_learner,
                       self.learner_params, self.select_instances, seq,
                       self.cover_X, self.cover_Y, self.target_X, self.target_Y,
                       self.loss_function) for seq in seqs)
        best_index = get_best_result_index(results)
        opt_error, opt_seq = results[best_index]
        opt_mmd = mmds[best_index]
        thinking += len(results)

        return_dict[self.think_key][0] = [opt_error, opt_mmd, opt_seq,
                                          time.time()-start_time]
        return_dict[self.error_key], return_dict[self.mmd_key], \
            return_dict[self.opt_key] = opt_error, opt_mmd, opt_seq
        return_dict[self.rejected_key] = self.rejected
        msg = 'This file contains the return dict for the Beam Search ' + \
            'saved on {}.'.format(time.strftime('%m/%d/%Y %H:%M:%S'))
        pickle.dump([msg, return_dict], open(self.output_fname, 'wb'))

        print('Starting optimum error: {}'.format(opt_error))

        rnd = 0
        # run search until thinking budget criterion met
        while True:
            children, children_mmds = self.generate_children(seqs)
            children, children_mmds = self.filter_children(children,
                                                           children_mmds)

            # evaluate performance of all children
            children_results = pool(delayed(get_error)(self.load_learner,
                                    self.fit_learner, self.learner_params,
                                    self.select_instances, child, self.cover_X,
                                    self.cover_Y, self.target_X,
                                    self.target_Y, self.loss_function)
                                    for child in children)
            best_index = get_best_result_index(children_results)
            curr_opt_error, curr_seq = children_results[best_index]
            thinking += len(children_results)

            if curr_opt_error < opt_error:
                opt_error, opt_seq, opt_mmd = \
                    curr_opt_error, curr_seq, children_mmds[best_index]

                return_dict[self.think_key][thinking] = \
                    [opt_error, opt_mmd, opt_seq, time.time() - start_time]
                return_dict[self.error_key], return_dict[self.mmd_key], \
                    return_dict[self.opt_key] = opt_error, opt_mmd, opt_seq
                return_dict[self.rejected_key] = self.rejected
                return_dict[self.time_key] = time.time() - start_time
                # write the return dict to file
                msg = 'This file contains the return dict for the ' + \
                    'Beam Search saved on ' + \
                    '{}.'.format(time.strftime('%m/%d/%Y %H:%M:%S'))
                pickle.dump([msg, return_dict], open(self.output_fname, 'wb'))

            seqs, results = self.prune(results, children_results)

            rnd += 1

            # break when used all of thinking budget
            if thinking > self.thinking_budget:
                break

            # if error is minimized then break
            if opt_error == 0.0:
                break
        return_dict[self.time_key] = time.time() - start_time
        return_dict[self.rejected_key] = self.rejected
        pickle.dump([msg, return_dict], open(self.output_fname, 'wb'))
        return return_dict
