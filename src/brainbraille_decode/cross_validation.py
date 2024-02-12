import itertools
import math
import random
import numpy as np

class BrainBrailleCVGen():
    def __init__(self, subs, runs, sess):
        sub_len = len(subs)
        run_len = len(runs)
        ses_len = len(sess)
        if sub_len != run_len or sub_len != ses_len or run_len != ses_len:
            raise ValueError('subs, runs, and sess should all have the same length!')
        self.subs = np.array(subs, dtype=int)
        self.runs = np.array(runs, dtype=int)
        self.sess = np.array(sess, dtype=int)
        self.index = np.arange(len(subs))

    def _sub_select_n(self, sub=1, n=1):
        selected_index = self.index[self.subs == sub]
        selected_index_len = len(selected_index)
        if selected_index_len == 0:
            raise ValueError(f'sub ({sub}) does not exist in subs ({self.subs}) used when instantiating')
        if n >= selected_index_len:
            raise ValueError(f'n({n}) needs to be smaller than the total number of runs ({selected_index_len}) in sub {sub}')
        test_index = itertools.combinations(selected_index, n)
        test_index_len = math.comb(selected_index_len, n)
        return selected_index, test_index, test_index_len

    def sub_dependent_leave_n_run_out(self, sub=1, n=1, max_fold_num=200):
        selected_index, test_index, test_index_len = self._sub_select_n(sub, n)
        selected_index_set = set(selected_index)
        test_index_list = []
        train_index_list = []
        if (max_fold_num is None) or (max_fold_num > test_index_len):
            for t_ind in test_index:
                test_index_list.append(np.array(t_ind))
                train_index_list.append(np.array(tuple(selected_index_set - set(t_ind))))
        else:
            test_index = list(test_index)
            random.shuffle(test_index)
            for i in range(max_fold_num):
                test_index_list.append(np.array(test_index[i]))
                train_index_list.append(np.array(tuple(selected_index_set - set(test_index[i]))))
        return test_index_list, train_index_list

    def sub_adaptive_leave_one_sub_out_test_sub(self, sub=1, n_adapt=1, max_fold_num=200):
        test_list, train_list, adapt_list = self.sub_independent_leave_one_sub_out_with_calib_test_sub(sub, n_adapt, max_fold_num)
        for i, (train_list_i, adapt_list_i) in enumerate(zip(train_list, adapt_list)):
            train_list[i] = np.concatenate((train_list_i, adapt_list_i))
        return test_list, train_list

    def sub_adaptive_leave_one_sub_out(self, n_adapt=1, max_fold_num=200):
        unique_subs = np.unique(self.subs)
        test_index_list = []
        train_index_list = []
        for sub_i in unique_subs:
            test_i, train_i = self.sub_adaptive_leave_one_sub_out_test_sub(sub_i, n_adapt, max_fold_num)
            test_index_list.extend(test_i)
            train_index_list.extend(train_i)
        return test_index_list, train_index_list

    def sub_independent_leave_one_sub_out_with_calib_test_sub(self, sub=1, n_calib=1, max_fold_num=200):
        calib_list, test_list = self.sub_dependent_leave_n_run_out(sub, n_calib, max_fold_num)
        _, train_list_i = self.sub_independent_leave_one_sub_out_test_sub(sub=1)
        train_list = train_list_i * len(test_list)
        return test_list, train_list, calib_list

    def sub_independent_leave_one_sub_out_with_calib(self, sub=1, n_calib=1, max_fold_num_each=200):
        unique_subs = np.unique(self.subs)
        test_index_list = []
        train_index_list = []
        calib_index_list = []
        for sub_i in unique_subs:
            test_list_i,  train_list_i, calib_list_i = self.sub_independent_leave_one_sub_out_with_calib_test_sub(sub_i)
            test_index_list.extend(test_list_i)
            train_index_list.extend(train_list_i)
            calib_index_list.extend(calib_list_i)
        return test_index_list, train_index_list, calib_index_list

    def sub_independent_leave_one_sub_out_test_sub(self, sub=1):
        return [self.index[self.subs == sub]], [self.index[self.subs != sub]]

    def sub_independent_leave_one_sub_out(self):
        unique_subs = np.unique(self.subs)
        test_index_list = []
        train_index_list = []
        for sub_i in unique_subs:
            test_i, train_i = self.sub_independent_leave_one_sub_out_test_sub(sub_i)
            test_index_list.extend(test_i)
            train_index_list.extend(train_i)
        return test_index_list, train_index_list