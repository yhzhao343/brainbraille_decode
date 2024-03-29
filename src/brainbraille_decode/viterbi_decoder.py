import numpy as np
from joblib import Parallel, delayed
from .viterbi_decoder_helpers import *
from .metrics import confusion_matrix
from sklearn.model_selection import KFold
import copy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from lipo import GlobalOptimizer
from .lm import add_k_gen, counts_to_proba

class BrainBrailleSegmentedDataToTransProb(BaseEstimator, TransformerMixin):
    def __init__(self, LETTERS_TO_DOT, region_order, clf_per_r, flatten_feature=True):
        self.LETTERS_TO_DOT = LETTERS_TO_DOT
        self.region_order = region_order
        self.clf_per_r = clf_per_r
        self.flatten_feature = flatten_feature

    def fit(self, X, y):
        if self.clf_per_r is None:
            raise Exception("No classifier per region")
        self.X = X
        y_trans = np.array(
            letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order),
            dtype=object,
        )
        y_trans = np.array(flatten_fold(y_trans), dtype=np.int_)
        self.y_trans = y_trans

        for clf_r_i in self.clf_per_r:
            if "probability" in clf_r_i.get_params():
                clf_r_i.set_params(probability=True)

        X = flatten_fold(X)
        if self.flatten_feature:
            X = flatten_feature(X)

        self.preprocessed_X = X
        self.clf_per_r = Parallel(n_jobs=-1)(
            delayed(clf_fit)(clf_r_i, X, np.ascontiguousarray(y_trans[:, r_i]))
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


class TransProbToStateProbClf:
    def __init__(self, method="arithmetic"):
        self.weight = np.array([[0.5], [0.5]])
        self.weight_sum = 1
        self.method = method

    def fit(self, X, y):
        self.weight[0] = 1 - np.mean(np.abs(X[:, 0] - y), axis=0)
        self.weight[1] = 1 - np.mean(np.abs(X[:, 1] - y), axis=0)
        self.weight_sum = np.sum(self.weight)
        return self

    def predict_proba(self, X):
        if self.method == "arithmetic":
            state_prob = np.squeeze(X @ self.weight) / self.weight_sum
        elif self.method == "geometric":
            state_prob = np.power(
                np.power(X[:, 0], self.weight[0]) * np.power(X[:, 1], self.weight[1]),
                1 / self.weight_sum,
            )
        else:
            state_prob = np.mean(X, axis=1)
        return state_prob


class TransProbToStateProb(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        LETTERS_TO_DOT,
        region_order,
        clf_per_r=None,
        clf_input_take_all_region=False,
        clf_input_flatten=True,
    ):
        self.LETTERS_TO_DOT = LETTERS_TO_DOT
        self.region_order = region_order
        if clf_per_r is None:
            self.clf_per_r = [TransProbToStateProbClf() for _ in self.region_order]
        else:
            self.clf_per_r = clf_per_r
        self.clf_input_take_all_region = clf_input_take_all_region
        self.clf_input_flatten = clf_input_flatten

    @staticmethod
    def prep_X(X, r_i=None, clf_input_flatten=False):
        if r_i is None:
            if clf_input_flatten:
                X_out = flatten_feature(X)
            else:
                X_out = X
        else:
            X_out = X[:, :, r_i]
        return X_out

    def fit(self, X, y):
        if self.clf_per_r is None:
            raise Exception("No classifier per region")
        y_trans = np.array(
            letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order),
            dtype=object,
        )

        state_prob_1_all_runs = [run_i[:, :, 2] + run_i[:, :, 3] for run_i in X]
        state_prob_2_all_runs = [run_i[:, :, 1] + run_i[:, :, 3] for run_i in X]
        state_prob_1_all_runs_padded = []
        state_prob_2_all_runs_padded = []
        for state_prob_1_run_i, state_prob_2_run_i in zip(
            state_prob_1_all_runs, state_prob_2_all_runs
        ):
            state_prob_1_all_runs_padded.append(
                np.concatenate(
                    (state_prob_1_run_i, state_prob_2_run_i[-1, :][np.newaxis, :])
                )
            )
            state_prob_2_all_runs_padded.append(
                np.concatenate(
                    (state_prob_1_run_i[0, :][np.newaxis, :], state_prob_2_run_i)
                )
            )

        state_prob_1 = np.array(flatten_fold(state_prob_1_all_runs_padded))
        state_prob_2 = np.array(flatten_fold(state_prob_2_all_runs_padded))
        state_prob = np.stack((state_prob_1, state_prob_2), axis=1)

        y_state = np.array(
            flatten_fold(
                [
                    np.concatenate((run_i // 2, (run_i[-1] % 2)[np.newaxis, :]))
                    for run_i in y_trans
                ]
            ),
            dtype=np.double,
        )

        self.clf_per_r = Parallel(n_jobs=-1)(
            delayed(clf_fit)(
                clf_r_i,
                np.ascontiguousarray(
                    self.prep_X(
                        state_prob,
                        r_i if not self.clf_input_take_all_region else None,
                        self.clf_input_flatten,
                    )
                ),
                np.ascontiguousarray(y_state[:, r_i]),
            )
            for clf_r_i, r_i in zip(self.clf_per_r, range(len(self.region_order)))
        )

        return self

    def transform(self, X):
        if self.clf_per_r is None:
            raise Exception("No classifier per region")

        state_prob_1_all_runs = [run_i[:, :, 2] + run_i[:, :, 3] for run_i in X]
        state_prob_2_all_runs = [run_i[:, :, 1] + run_i[:, :, 3] for run_i in X]
        state_prob_1_all_runs_padded = []
        state_prob_2_all_runs_padded = []
        for state_prob_1_run_i, state_prob_2_run_i in zip(
            state_prob_1_all_runs, state_prob_2_all_runs
        ):
            state_prob_1_all_runs_padded.append(
                np.concatenate(
                    (state_prob_1_run_i, state_prob_2_run_i[-1, :][np.newaxis, :])
                )
            )
            state_prob_2_all_runs_padded.append(
                np.concatenate(
                    (state_prob_1_run_i[0, :][np.newaxis, :], state_prob_2_run_i)
                )
            )

        X_each_run_start_end = get_run_start_end_index(state_prob_1_all_runs_padded)

        state_prob_1 = np.array(flatten_fold(state_prob_1_all_runs_padded))
        state_prob_2 = np.array(flatten_fold(state_prob_2_all_runs_padded))
        state_prob = np.stack((state_prob_1, state_prob_2), axis=1)

        state_prob = Parallel(n_jobs=-1)(
            delayed(clf_pred_proba)(
                clf_r_i,
                self.prep_X(
                    state_prob,
                    r_i if not self.clf_input_take_all_region else None,
                    self.clf_input_flatten,
                ),
            )
            for clf_r_i, r_i in zip(self.clf_per_r, range(len(self.region_order)))
        )

        state_prob = np.array(state_prob).T

        self.state_prob = [
            np.ascontiguousarray(state_prob[s:e]) for s, e in X_each_run_start_end
        ]
        return self.state_prob


class StateProbaToLetterProb(BaseEstimator, TransformerMixin):
    def __init__(self, LETTERS_TO_DOT, region_order, clf=None):
        self.letters = np.array(list(LETTERS_TO_DOT.keys()))
        self.letters_to_ind = {letter: ind for ind, letter in enumerate(self.letters)}
        self.region_order = region_order
        self.region_to_index = {r: ind for ind, r in enumerate(region_order)}
        region_by_letter_arr = np.zeros(
            (len(self.letters), len(self.region_order)), dtype=np.bool_
        )
        for letter, region_onoff in LETTERS_TO_DOT.items():
            for r, onoff in region_onoff.items():
                region_by_letter_arr[self.letters_to_ind[letter]][
                    self.region_to_index[r]
                ] = onoff
        self.region_by_letter_arr = region_by_letter_arr
        self.clf = (
            StateProbaToLetterProbClf(self.letters.size, region_by_letter_arr)
            if clf is None
            else clf
        )
        self.letter_prob = None
        self.naive_letter = None

    def fit(self, X, y):
        X = np.ascontiguousarray(flatten_fold(X), dtype=np.double)
        y = np.ascontiguousarray(
            [self.letters_to_ind[l] for l in flatten_fold(y)], dtype=np.int_
        )
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        X_each_run_start_end = get_run_start_end_index(X)
        X = np.ascontiguousarray(flatten_fold(X), dtype=np.double)
        letter_prob = self.clf.predict_proba(X)
        self.letter_prob = [
            np.ascontiguousarray(letter_prob[s:e]) for s, e in X_each_run_start_end
        ]
        return self.letter_prob

    def get_naive_letter(self, letter_prob=None):
        if (letter_prob is None) and (self.letter_prob is not None):
            letter_prob = self.letter_prob
        if letter_prob is None:
            raise Exception(
                "No input, letter_prob is None, or there is no previous cached self.letter_prob"
            )
        naive_letter = [
            self.letters[letter_prob_run_i.argmax(axis=-1)]
            for letter_prob_run_i in letter_prob
        ]
        self.naive_letter = naive_letter
        return naive_letter

    def get_naive_letter_cm(self, y_letter, letter_prob=None):
        pred_letter = self.get_naive_letter(letter_prob)
        pred_letter_flatten = [l_i for run_i in pred_letter for l_i in run_i]
        y_letter_flatten = [l_i for run_i in y_letter for l_i in run_i]
        y_unique_labels = np.sort(np.unique(y_letter_flatten))
        return confusion_matrix(y_letter_flatten, pred_letter_flatten, y_unique_labels)


class StateProbaToLetterProbClf:
    def __init__(self, num_letters, region_by_letter_arr):
        self.num_letters = num_letters
        self.region_by_letter_arr = region_by_letter_arr

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return get_naive_letter_prob(
            X, self.region_by_letter_arr, np.empty((len(X), self.num_letters))
        )


class LetterProbaToLetterDecode(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        LETTERS_TO_DOT,
        region_order,
        bigram_dict=None,
        unigram_counts=None,
        unigram_smoothing_k=0,
        bigram_counts=None,
        bigram_smoothing_k=0,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
        insertion_penalty=0,
        insertion_penalty_lower=-10.0,
        insertion_penalty_higher=10.0,
        softmax_on_bigram_matrix=True,
        CV_tune_insertion_penalty=True,
        skip_naive=False,
        skip_letter_viterbi=False,
        skip_grammar_viterbi=False,
        random_state=42,
        n_calls=30,
    ):
        self.bigram_counts = None
        self.bigram_matrix = None
        self.bigram_log_matrix = None
        self.unigram_counts = None
        self.unigram_prior = None
        self.unigram_log_matrix = None
        self.unigram_smoothing_k = unigram_smoothing_k
        self.bigram_smoothing_k = bigram_smoothing_k
        self.region_order = region_order
        self.regressor_to_ind = {r: ind for ind, r in enumerate(region_order)}
        if unigram_counts is not None:
            self.
        self.add_LETTERS_TO_DOT(LETTERS_TO_DOT)
        if bigram_counts is not None:
            self.add_bigram_counts_matrix(bigram_counts, self.bigram_smoothing_k)
        elif bigram_dict is not None:
            self.add_bigram_dict(bigram_dict, softmax_on_bigram_matrix)
        self.words_node_symbols = words_node_symbols
        self.words_link_start_end = words_link_start_end
        self.words_dictionary = words_dictionary
        self.insertion_penalty = insertion_penalty
        self.insertion_penalty_lower = insertion_penalty_lower
        self.insertion_penalty_higher = insertion_penalty_higher
        self.CV_tune_insertion_penalty = CV_tune_insertion_penalty
        self.naive_prob_letter_label = None
        self.bigram_weighted_letter_label = None
        self.letter_viterbi_decode_letter_label = None
        self.skip_naive = skip_naive
        self.skip_letter_viterbi = skip_letter_viterbi
        self.skip_grammar_viterbi = skip_grammar_viterbi
        self.random_state = random_state
        self.n_calls = n_calls
        self.X = None
        self.y = None

    def add_LETTERS_TO_DOT(self, LETTERS_TO_DOT):
        self.LETTERS_TO_DOT = LETTERS_TO_DOT
        letters = np.array(list(LETTERS_TO_DOT.keys()))
        self.letters = letters
        self.DOT_TO_LETTERS = {
            "".join([str(val[r]) for r in self.region_order]): key
            for key, val in self.LETTERS_TO_DOT.items()
        }
        self.letters_to_ind = {letter: ind for ind, letter in enumerate(letters)}
        region_by_letter_arr = np.zeros(
            (len(letters), len(self.region_order)), dtype=np.bool_
        )
        for letter, region_onoff in LETTERS_TO_DOT.items():
            for r, onoff in region_onoff.items():
                region_by_letter_arr[self.letters_to_ind[letter]][
                    self.regressor_to_ind[r]
                ] = onoff
        self.region_by_letter_arr = region_by_letter_arr

    def add_unigram_counts_matrix(self, unigram_counts, k):
        if unigram_counts is not None:
            self.unigram_counts = unigram_counts
            self.unigram_prior = counts_to_proba(unigram_counts, add_k_gen(k))
            self.unigram_log_matrix = np.log10(self.unigram_prior)

    def add_bigram_counts_matrix(self, bigram_counts, k):
        if bigram_counts is not None:
            self.bigram_counts = bigram_counts
            self.bigram_matrix = counts_to_proba(bigram_counts, add_k_gen(k))
            self.bigram_log_matrix = np.log10(self.bigram_matrix)

    def add_bigram_dict(self, bigram_dict, softmax_on_bigram_matrix=False):
        # Legacy code! Do not use
        if bigram_dict is not None:
            self.bigram_matrix = np.zeros((self.letters.size, self.letters.size))
            for prev, next_prob in bigram_dict.items():
                for next, prob_val in next_prob.items():
                    self.bigram_matrix[self.letters_to_ind[prev]][
                        self.letters_to_ind[next]
                    ] = prob_val
            self.bigram_matrix[self.bigram_matrix == 0] = np.min(
                self.bigram_matrix[self.bigram_matrix > 0]
            )
            if softmax_on_bigram_matrix:
                self.bigram_matrix = softmax(self.bigram_matrix)
            self.bigram_log_matrix = np.log10(self.bigram_matrix)

    @staticmethod
    def tune_insertion_penalty(
        objective_function, lower_bounds, upper_bounds, random_state, n_calls
    ):
        optimizer = GlobalOptimizer(
            function=objective_function,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
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

    @staticmethod
    def decode_acc_with_ins_pen_gen(
        X,
        y,
        decode_grammar_viterbi,
        letters,
        bigram_log_matrix,
        words_node_symbols,
        words_link_start_end,
        words_dictionary,
    ):
        def decode_acc_with_ins_pen(insertion_penalty):
            y_pred = decode_grammar_viterbi(
                X,
                letters,
                bigram_log_matrix,
                words_node_symbols,
                words_link_start_end,
                words_dictionary,
                insertion_penalty,
            )
            y_pred = [l for run in y_pred for l in run]
            y_true = [l for run in y for l in run]
            return accuracy_score(y_true, y_pred)

        return decode_acc_with_ins_pen

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.CV_tune_insertion_penalty:
            self.re_tune()
        return self

    def re_tune(
        self,
        decode_grammar_viterbi=None,
        letters=None,
        # bigram_log_matrix=None,
        bigram_dict=None,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
        insertion_penalty_lower=None,
        insertion_penalty_higher=None,
        n_calls=None,
        random_state=None,
        softmax_on_bigram_matrix=True,
        CV_tune_insertion_penalty=None,
    ):
        if decode_grammar_viterbi is None:
            decode_grammar_viterbi = self.decode_grammar_viterbi

        if letters is None:
            letters = self.letters

        if bigram_dict is not None:
            self.add_bigram_dict(bigram_dict, softmax_on_bigram_matrix)
        bigram_log_matrix = self.bigram_log_matrix

        if words_node_symbols is None:
            words_node_symbols = self.words_node_symbols

        if words_link_start_end is None:
            words_link_start_end = self.words_link_start_end

        if words_dictionary is None:
            words_dictionary = self.words_dictionary

        if insertion_penalty_lower is None:
            insertion_penalty_lower = self.insertion_penalty_lower

        if insertion_penalty_higher is None:
            insertion_penalty_higher = self.insertion_penalty_higher

        if n_calls is None:
            n_calls = self.n_calls

        if random_state is None:
            random_state = self.random_state

        if CV_tune_insertion_penalty is None:
            CV_tune_insertion_penalty = self.CV_tune_insertion_penalty

        if CV_tune_insertion_penalty:
            objective_function = self.decode_acc_with_ins_pen_gen(
                self.X,
                self.y,
                decode_grammar_viterbi,
                letters,
                bigram_log_matrix,
                words_node_symbols,
                words_link_start_end,
                words_dictionary,
            )
            tune_clf_results, tune_clf_history = self.tune_insertion_penalty(
                objective_function,
                {"insertion_penalty": insertion_penalty_lower},
                {"insertion_penalty": insertion_penalty_higher},
                random_state,
                n_calls,
            )
            self.insertion_penalty = tune_clf_results[0]["insertion_penalty"]

    @staticmethod
    def decode_naive_prob_letter(naive_letter_prob, letters):
        naive_prob_letter_label = [
            letters[np.argmax(run_i, axis=1)] for run_i in naive_letter_prob
        ]
        return naive_prob_letter_label

    @staticmethod
    def decode_letter_viterbi(
        naive_letter_prob, letters, bigram_matrix, bigram_log_matrix
    ):
        bigram_weighted_prob = [
            add_bigram_probabilities(run_i, bigram_matrix, np.empty_like(run_i))
            for run_i in naive_letter_prob
        ]
        bigram_weighted_ind = [
            np.argmax(run_i, axis=-1) for run_i in bigram_weighted_prob
        ]
        bigram_weighted_letter_label = [letters[run_i] for run_i in bigram_weighted_ind]
        # latest_results = self.bigram_weighted_letter_label
        letter_viterbi_decode_letter_label = [
            letter_level_bigram_viterbi_decode(run_i, bigram_log_matrix, letters)
            for run_i in naive_letter_prob
        ]

        return (
            bigram_weighted_prob,
            bigram_weighted_letter_label,
            letter_viterbi_decode_letter_label,
        )

    @staticmethod
    def decode_grammar_viterbi(
        letter_probs_array_all_runs,
        letter_list,
        transition_log_prob_matrix,
        grammar_node_symbols,
        grammar_link_start_end,
        dictionary,
        insert_panelty=0.0,
    ):
        return Parallel(n_jobs=-1)(
            delayed(letter_bigram_viterbi_with_grammar_decode)(
                run_i,
                letter_list,
                transition_log_prob_matrix,
                grammar_node_symbols,
                grammar_link_start_end,
                dictionary,
                insert_panelty,
            )
            for run_i in letter_probs_array_all_runs
        )

    def predict(
        self,
        X=None,
        skip_naive=None,
        skip_letter_viterbi=None,
        skip_grammar_viterbi=None,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
        insertion_penalty=None,
    ):
        if X is None:
            if self.naive_letter_prob is not None:
                X = self.naive_letter_prob
            else:
                raise Exception("No X input")
        self.naive_letter_prob = X
        if skip_naive is None:
            skip_naive = self.skip_naive

        if not skip_naive:
            self.naive_prob_letter_label = self.decode_naive_prob_letter(
                self.naive_letter_prob, self.letters
            )

            latest_results = self.naive_prob_letter_label

        if skip_letter_viterbi is None:
            skip_letter_viterbi = self.skip_letter_viterbi

        if (
            (self.bigram_matrix is not None)
            and (not skip_letter_viterbi)
            and (self.naive_letter_prob is not None)
        ):
            (
                self.bigram_weighted_prob,
                self.bigram_weighted_letter_label,
                self.letter_viterbi_decode_letter_label,
            ) = self.decode_letter_viterbi(
                self.naive_letter_prob,
                self.letters,
                self.bigram_matrix,
                self.bigram_log_matrix,
            )
            latest_results = self.letter_viterbi_decode_letter_label

        if words_node_symbols is None:
            words_node_symbols = self.words_node_symbols
        if words_link_start_end is None:
            words_link_start_end = self.words_link_start_end
        if words_dictionary is None:
            words_dictionary = self.words_dictionary
        if insertion_penalty is None:
            insertion_penalty = self.insertion_penalty

        if skip_grammar_viterbi is None:
            skip_grammar_viterbi = self.skip_grammar_viterbi
        if (
            (not skip_grammar_viterbi)
            and (words_node_symbols is not None)
            and (words_link_start_end is not None)
            and (words_dictionary is not None)
            and (insertion_penalty is not None)
            and (self.naive_letter_prob is not None)
        ):
            self.grammar_viterbi_decode_letter_label = self.decode_grammar_viterbi(
                self.naive_letter_prob,
                self.letters,
                self.bigram_log_matrix,
                words_node_symbols,
                words_link_start_end,
                words_dictionary,
                insertion_penalty,
            )
            latest_results = self.grammar_viterbi_decode_letter_label

        return latest_results

    def obtain_letter_viterbi_cm(self, y_letter):
        if self.letter_viterbi_decode_letter_label is not None:
            pred_letter_flatten = [
                l_i
                for run_i in self.letter_viterbi_decode_letter_label
                for l_i in run_i
            ]
            y_letter_flatten = [l_i for run_i in y_letter for l_i in run_i]
            y_unique_labels = np.sort(np.unique(y_letter_flatten))
            return confusion_matrix(
                y_letter_flatten, pred_letter_flatten, y_unique_labels
            )


class BrainBrailleDataToTransProbCV(BaseEstimator, TransformerMixin):
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
        train_group=None,
        test_group=None,
        KFold_n_split=None,
        z_normalize=True,
        flatten_feature=True,
        n_calls=32,
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
        y_trans = np.array(
            letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order),
            dtype=object,
        )
        cv_train_X = [X[cv_ndx_i] for cv_ndx_i in cv_train_ndx]
        cv_train_y = [y_trans[cv_ndx_i] for cv_ndx_i in cv_train_ndx]

        cv_test_X = [X[cv_ndx_i] for cv_ndx_i in cv_test_ndx]
        cv_test_y = [y_trans[cv_ndx_i] for cv_ndx_i in cv_test_ndx]

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

        self.sliced_data_to_trans_prob = BrainBrailleSegmentedDataToTransProb(
            self.LETTERS_TO_DOT, self.region_order, self.clf_per_r, self.flatten_feature
        )
        if self.z_norm_by_group is not None:
            X = self.z_norm_by_group.fit_transform(X)
        X = self.data_slicer.fit_transform(X, num_trans=len(y_trans[0]))
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

class BrainBrialleTransProbToPseudoEmissionProb(BaseEstimator, TransformerMixin):
    def __init__(self, LETTERS_TO_DOT, region_order):
        self.LETTERS_TO_DOT = LETTERS_TO_DOT
        self.region_order = region_order

    def fit(self, X, y):
        y_trans = np.array(
            letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order),
            dtype=object,
        )
        num_run, num_sample, num_reg = y_trans.shape
        y_trans_concat = y_trans.reshape((num_run * num_sample, num_reg))
        y_trans_prior = np.zeros((num_reg, 4))
        for i in range(4):
            y_trans_prior[:, i] = np.sum(y_trans_concat == i, axis=0) / y_trans_concat.shape[0]
        self.y_trans_prior = y_trans_prior
        self.y_trans_prior_scaled = self.y_trans_prior / np.min(self.y_trans_prior, axis=1)[:, np.newaxis]
        return self

    def transform(self, X):
        transformed_X = [x_i / self.y_trans_prior_scaled for x_i in X]
        return transformed_X