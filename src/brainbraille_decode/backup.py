import numpy as np
from joblib import Parallel, delayed
from brainbraille_decode.viterbi_decoder_helpers import *
from brainbraille_decode.metrics import confusion_matrix
from sklearn.model_selection import KFold
import copy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from lipo import GlobalOptimizer
from brainbraille_decode.lm import add_k_gen, counts_to_proba
from brainbraille_decode.preprocessing import ZNormalizeByGroup

def letter_label_to_state_label(y, LETTERS_TO_DOT, region_order):
    dot_label = np.array([
        [[LETTERS_TO_DOT[l_i][region] for region in region_order] for l_i in run_i]
        for run_i in y
    ])
    return dot_label

class StateProbToOnStateProb(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        return [x_i[:, :, 1] for x_i in X]

    def fit(self, X, y=None):
        return self

class BrainBrailleLetterProbToPseudoEmissionProb(BaseEstimator, TransformerMixin):
    def __init__(self, prior):
        self.prior = prior

    def transform(self, X):
        return [x_i/self.prior for x_i in X]

    def fit(self, X, y=None):
        return self


class BrainBrailleSegmentedDataToProb(BaseEstimator, TransformerMixin):
    def __init__(self, LETTERS_TO_DOT, region_order, clf_per_r, flatten_feature=True, label_type='state'):
        self.LETTERS_TO_DOT = LETTERS_TO_DOT
        self.region_order = region_order
        self.clf_per_r = clf_per_r
        self.flatten_feature = flatten_feature
        self.label_type = label_type

    def fit(self, X, y):
        if self.clf_per_r is None:
            raise Exception("No classifier per region")
        self.X = X
        if self.label_type == 'state':
            y_label = np.array(
                letter_label_to_state_label(y, self.LETTERS_TO_DOT, self.region_order),
                dtype=object,
            )
        elif self.label_type == 'trans':
            y_label = np.array(
                letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order),
                dtype=object,
            )

        y_label = np.array(flatten_fold(y_label), dtype=np.int_)
        self.y_label = y_label

        for clf_r_i in self.clf_per_r:
            if "probability" in clf_r_i.get_params():
                clf_r_i.set_params(probability=True)

        X = flatten_fold(X)
        if self.flatten_feature:
            X = flatten_feature(X)

        self.preprocessed_X = X
        self.clf_per_r = Parallel(n_jobs=-1)(
            delayed(clf_fit)(clf_r_i, X, np.ascontiguousarray(y_label[:, r_i]))
            for clf_r_i, r_i in zip(self.clf_per_r, range(len(self.region_order)))
        )
        return self

    def transform(self, X):
        if self.clf_per_r is None:
            raise Exception("No classifier per region")
        X_each_run_start_end = get_run_start_end_index(X)
        if self.X is not X:
            self.X = X
            X = flatten_fold(X)
            if self.flatten_feature:
                X = flatten_feature(X)

            self.preprocessed_X = X
        preprocessed_X = self.preprocessed_X

        trans_proba = np.ascontiguousarray(
            Parallel(n_jobs=-1)(
                delayed(clf_pred_proba)(clf_r_i, preprocessed_X)
                for clf_r_i in self.clf_per_r
            )
        )
        trans_proba = trans_proba.transpose(1, 0, 2)
        self.trans_proba = [
            np.ascontiguousarray(trans_proba[s:e]) for s, e in X_each_run_start_end
        ]
        return self.trans_proba

    def get_trans_class(self, trans_proba=None):
        if (trans_proba is None) and (self.trans_proba is not None):
            trans_proba = self.trans_proba
        if trans_proba is None:
            raise Exception(
                "No input, trans_proba is None, or there is no previous cached self.letter_prob"
            )
        trans_class = [
            trans_proba_run_i.argmax(axis=-1) for trans_proba_run_i in trans_proba
        ]
        self.trans_class = trans_class
        return trans_class

class BrainBrailleDataToProbCV(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        LETTERS_TO_DOT,
        region_order,
        clf,
        data_slicer,
        param_grid,
        param_category_keys=[],
        log_args_keys=[],
        param_transform_dict={},
        label_type='state',
        train_group=None,
        test_group=None,
        KFold_n_split=None,
        z_normalize=True,
        flatten_feature=True,
        n_calls=64,
        random_state=42,
        verbose=True,
        inner_n_jobs=-1,
    ):
        self.region_order = region_order
        self.train_group = train_group
        self.test_group = test_group
        self.clf = clf
        self.param_grid = param_grid
        self.param_category_keys = param_category_keys
        if len(param_category_keys) == 0:
            for key, val in self.param_grid.items():
                if (len(val) != 2) or type(val[0]) == str:
                    self.param_category_keys.append(key)
        self.label_type = label_type
        self.log_args_keys = log_args_keys
        self.data_slicer = data_slicer
        self.LETTERS_TO_DOT = LETTERS_TO_DOT
        self.KFold_n_split = KFold_n_split
        self.z_normalize = z_normalize
        self.flatten_feature = flatten_feature
        total_param = set(param_grid.keys())
        range_param = total_param - set(self.param_category_keys)
        self.param_transform_dict = {}
        self.param_transform_dict.update(param_transform_dict)
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.inner_n_jobs = inner_n_jobs

    @staticmethod
    def clf_cv_param_search_score_gen(
        clf,
        param_transform_dict,
        cv_train_X,
        cv_train_y,
        cv_test_X,
        cv_test_y,
        n_jobs=-1,
    ):
        param_transform_keys = param_transform_dict.keys()

        def clf_score(**kwargs):
            for key in param_transform_keys:
                if key in kwargs:
                    kwargs[key] = param_transform_dict[key](kwargs[key])

            scores = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(clf_fit_pred_score)(
                        copy.deepcopy(clf).set_params(**kwargs),
                        train_X_i,
                        train_y_i,
                        test_X_i,
                        test_y_i,
                    )
                    for train_X_i, train_y_i, test_X_i, test_y_i in zip(
                        cv_train_X, cv_train_y, cv_test_X, cv_test_y
                    )
                )
            )

            avg_acc = np.mean(scores)
            # print(kwargs, avg_acc)
            return avg_acc

        return clf_score

    @staticmethod
    def tune_clf(
        objective_function,
        param_grid,
        param_category_keys,
        log_args_keys,
        n_calls,
        random_state,
    ):
        all_keys_set = set(param_grid.keys())
        category_keys_set = set(param_category_keys)
        range_keys_set = all_keys_set - category_keys_set
        lower_bounds = {}
        upper_bounds = {}
        categories = {}
        for key in range_keys_set:
            lower_bounds[key] = param_grid[key][0]
            upper_bounds[key] = param_grid[key][1]
        for key in category_keys_set:
            categories[key] = param_grid[key]
        optimizer = GlobalOptimizer(
            function=objective_function,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            categories=categories,
            log_args=log_args_keys,
            maximize=True,
            flexible_bound_threshold=0.1,
            random_state=random_state,
        )
        for _ in range(n_calls):
            candidate = optimizer.get_candidate()
            candidate.set(optimizer.function(**candidate.x))
            if np.isclose(optimizer.optimum[1], 1.0):
                break

        return optimizer.optimum, optimizer.running_optimum

    def fit(self, X, y, train_group=None, test_group=None):
        self.X = X
        if train_group is None:
            if self.train_group is None:
                train_group = np.zeros(len(X), dtype=int)
            else:
                train_group = np.array(self.train_group)
        if test_group is None:
            if self.test_group is None:
                test_group = np.zeros(len(y), dtype=int)
            else:
                test_group = np.array(self.test_group)
        n_split_max = len(train_group)
        KFold_n_split = self.KFold_n_split
        if KFold_n_split is None:
            KFold_n_split = n_split_max
        elif KFold_n_split > n_split_max:
            KFold_n_split = n_split_max
        shuffle = False if (KFold_n_split == n_split_max) else True
        random_state = self.random_state if shuffle else None
        cv = KFold(KFold_n_split, random_state=random_state, shuffle=shuffle)
        cv_train_ndx, cv_test_ndx = zip(*list(cv.split(X)))
        if self.label_type == 'state':
            y_label = np.array(
                letter_label_to_state_label(y, self.LETTERS_TO_DOT, self.region_order),
                dtype=object,
            )
        elif self.label_type == 'trans':
            y_label = np.array(
                letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order),
                dtype=object,
            )
        cv_train_X = [X[cv_ndx_i] for cv_ndx_i in cv_train_ndx]
        cv_train_y = [y_label[cv_ndx_i] for cv_ndx_i in cv_train_ndx]

        cv_test_X = [X[cv_ndx_i] for cv_ndx_i in cv_test_ndx]
        cv_test_y = [y_label[cv_ndx_i] for cv_ndx_i in cv_test_ndx]

        cv_train_X, cv_test_X = zip(
            *Parallel(n_jobs=-1)(
                delayed(preprocess_each_fold)(
                    cv_train_X_i=cv_train_X_i,
                    num_trans=len(cv_train_y_i[0]),
                    cv_data_slicer_i=self.data_slicer,
                    cv_test_X_i=cv_test_X_i,
                    z_norm_by_group=ZNormalizeByGroup(
                        train_group[cv_train_ndx_i], train_group[cv_test_ndx_i]
                    ),
                    flatten_feat_enable=self.flatten_feature,
                    flatten_fold_enable=True,
                )
                for cv_train_X_i, cv_test_X_i, cv_train_y_i, cv_train_ndx_i, cv_test_ndx_i in zip(
                    cv_train_X, cv_test_X, cv_train_y, cv_train_ndx, cv_test_ndx
                )
            )
        )

        cv_train_X = np.ascontiguousarray(cv_train_X)
        cv_test_X = np.ascontiguousarray(cv_test_X)

        cv_train_y = np.array([flatten_fold(y_i) for y_i in cv_train_y], dtype=np.int_)
        cv_test_y = np.array([flatten_fold(y_i) for y_i in cv_test_y], dtype=np.int_)

        tune_clf_results_and_history = Parallel(n_jobs=-1)(
            delayed(self.tune_clf)(
                self.clf_cv_param_search_score_gen(
                    self.clf,
                    self.param_transform_dict,
                    cv_train_X,
                    cv_train_y[:, :, r_i],
                    cv_test_X,
                    cv_test_y[:, :, r_i],
                ),
                self.param_grid,
                self.param_category_keys,
                self.log_args_keys,
                self.n_calls,
                self.random_state,
            )
            for r_i in range(len(self.region_order))
        )
        if tune_clf_results_and_history is None:
            raise Exception("tune clf failed")
        tune_clf_results, tune_clf_history = zip(*tune_clf_results_and_history)
        self.tune_clf_history = tune_clf_history
        if self.verbose:
            tune_clf_results_pretty_print(tune_clf_results, self.region_order)
        best_params, _, _ = zip(*tune_clf_results)

        self.clf_per_r = [
            copy.deepcopy(self.clf).set_params(**best_p) for best_p in best_params
        ]
        self.z_norm_by_group = (
            ZNormalizeByGroup(train_group, test_group) if self.z_normalize else None
        )

        self.sliced_data_to_trans_prob = BrainBrailleSegmentedDataToProb(
            self.LETTERS_TO_DOT, self.region_order, self.clf_per_r, self.flatten_feature, self.label_type
        )
        if self.z_norm_by_group is not None:
            X = self.z_norm_by_group.fit_transform(X)
        X = self.data_slicer.fit_transform(X, num_trans=len(y_label[0]))
        self.preprocessed_X = self.sliced_data_to_trans_prob.fit_transform(X, y)

        return self

    def transform(self, X):
        if self.X is X:
            return self.preprocessed_X
        if self.z_norm_by_group is not None:
            X = self.z_norm_by_group.transform(X)
        X = self.data_slicer.transform(X)
        return self.sliced_data_to_trans_prob.transform(X)

    def get_trans_class(self, trans_proba=None):
        return self.sliced_data_to_trans_prob.get_trans_class(trans_proba)

## ============================================================================
# stim_text_cont = sent_separation_tok.join(stimulus_text_content.split("\n"))
# one_gram = get_one_gram_feat_vector(stim_text_cont)
# two_gram = get_two_gram_feat_vector(stim_text_cont)