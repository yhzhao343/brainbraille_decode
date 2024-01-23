import numpy as np
from numba import jit, prange, f8, i4, i8, b1
import numba as nb
from sklearn.svm import SVC
from joblib import Parallel, delayed
from .metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import copy
import dlib
import sys


def clf_pred_proba(clf, X):
    return clf.predict_proba(X)


def clf_pred(clf, X):
    return clf.predict(X)


def letter_label_to_transition_label(y, LETTERS_TO_DOT, region_order):
    dot_label = [
        [[LETTERS_TO_DOT[l_i][region] for region in region_order] for l_i in run_i]
        for run_i in y
    ]
    transition_label = [
        [
            [(prev[i] * 2 + curr[i]) for i in range(len(region_order))]
            for prev, curr in zip(run_i[0:-1], run_i[1:])
        ]
        for run_i in dot_label
    ]
    return transition_label


@jit(
    f8[:, ::1](f8[:, :], b1[:, ::1], f8[:, ::1]),
    parallel=True,
    nopython=True,
    fastmath=True,
    cache=True,
)
def get_naive_letter_prob(
    state_prob_by_type_run_i, region_by_letter_arr, naive_letter_prob
):
    t, r = state_prob_by_type_run_i.shape
    l, _ = region_by_letter_arr.shape
    naive_letter_prob[:, :] = 1
    for l_i in prange(l):
        r_by_l_i = region_by_letter_arr[l_i]
        not_r_by_l_i = np.logical_not(r_by_l_i)
        for t_i in range(t):
            naive_letter_prob[t_i][l_i] *= np.prod(
                state_prob_by_type_run_i[t_i][r_by_l_i]
            )
            naive_letter_prob[t_i][l_i] *= np.prod(
                1 - state_prob_by_type_run_i[t_i][not_r_by_l_i]
            )
    return naive_letter_prob


@jit(
    f8[:, ::1](f8[:, ::1], f8[:, ::1], f8[:, ::1]),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def add_bigram_probabilities(letter_probs, bigram_prob_matrix, letter_probs_updated):
    letter_probs_updated[0, :] = letter_probs[0, :]
    for i in prange(1, letter_probs.shape[0]):
        prev_letter_prob = letter_probs[i - 1]
        transition_correction_prob = np.zeros_like(prev_letter_prob)
        # for l_i, prob_l in enumerate(prev_letter_prob):
        for l_i in range(prev_letter_prob.size):
            prob_l = prev_letter_prob[l_i]
            transition_correction_prob += bigram_prob_matrix[l_i] * prob_l
        letter_probs_updated[i] = letter_probs[i] * transition_correction_prob
    return letter_probs_updated


def letter_level_bigram_viterbi_decode(letter_probs, bigram_log_prob_matrix, letters):
    emission_log_prob_table = np.zeros_like(letter_probs)
    prev_table = np.zeros_like(letter_probs, dtype=np.int32)
    letter_ind = np.zeros(letter_probs.shape[0], dtype=np.int32)
    return letters[
        letter_level_bigram_viterbi_decode_numba(
            letter_probs,
            bigram_log_prob_matrix,
            emission_log_prob_table,
            prev_table,
            letter_ind,
        )
    ]


MIN_FLOAT = -sys.float_info.max


@jit(
    i4[::1](f8[:, ::1], f8[:, ::1], f8[:, ::1], i4[:, ::1], i4[::1]),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def letter_level_bigram_viterbi_decode_numba(
    letter_probs,
    bigram_log_prob_matrix,
    emission_log_prob_table,
    prev_table,
    letter_ind,
):
    num_entry, dict_size = letter_probs.shape
    letter_log_probs = np.log10(letter_probs)
    emission_log_prob_table[0, :] = letter_log_probs[0, :]
    emission_log_prob_table[1:, :] = MIN_FLOAT
    prev_table[:] = -1
    for i in range(1, num_entry):
        for l_i in prange(dict_size):
            new_log_prob_from_l_i = (
                emission_log_prob_table[i - 1][l_i]
                + bigram_log_prob_matrix[l_i]
                + letter_log_probs[i]
            )
            new_log_prob_larger = new_log_prob_from_l_i > emission_log_prob_table[i, :]
            emission_log_prob_table[i][new_log_prob_larger] = new_log_prob_from_l_i[
                new_log_prob_larger
            ]
            prev_table[i][new_log_prob_larger] = l_i
    letter_ind[-1] = np.argmax(emission_log_prob_table[-1, :])
    for i in range(num_entry - 2, -1, -1):
        letter_ind[i] = prev_table[i + 1, letter_ind[i + 1]]
    return letter_ind


def softmax(scores):
    scores = np.array(scores)
    input_is_1d = len(scores.shape) == 1
    if input_is_1d:
        scores = scores[np.newaxis, :]
    exp_scores = np.exp(scores - np.max(scores, axis=-1)[:, np.newaxis])
    res = exp_scores / np.sum(exp_scores, axis=-1)[:, np.newaxis]
    if input_is_1d:
        res = np.squeeze(res)
    return res


DEFAULT_SVM_PARAM = {
    "kernel": "rbf",
    "probability": True,
    "break_ties": True,
    "max_iter": -1,
}


def fit_svm(X, y_label, SVM_params=None):
    clf = SVC(**DEFAULT_SVM_PARAM)
    if SVM_params is not None:
        clf = clf.set_params(**SVM_params)
    clf.fit(X, y_label)
    return clf


class SVMProbDecoder:
    def __init__(
        self,
        LETTERS_TO_DOT,
        region_order,
        bigram_dict=None,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
        SVM_params=DEFAULT_SVM_PARAM,
        insertion_penalty=None,
        softmax_on_bigram_matrix=False,
    ):
        self.region_order = region_order
        self.regressor_to_ind = {r: ind for ind, r in enumerate(region_order)}
        self.add_LETTERS_TO_DOT(LETTERS_TO_DOT)
        self.SVM_params = [SVM_params] * len(region_order)
        self.clfs = [None] * len(region_order)
        # for i in range(len(region_order)):
        # self.clfs.append(SVC())
        self.add_bigram_dict(bigram_dict, softmax_on_bigram_matrix)
        self.state_prob_1_w = np.ones(len(region_order))
        self.state_prob_2_w = np.ones(len(region_order))
        self.words_node_symbols = words_node_symbols
        self.words_link_start_end = words_link_start_end
        self.words_dictionary = words_dictionary
        self.insertion_penalty = insertion_penalty
        self.trans_prob_by_type = None
        self.trans_class = None
        self.trans_class_by_type = None
        self.tuple_label_by_type = None
        self.label_1 = None
        self.label_2 = None
        self.string_label_1 = None
        self.string_label_2 = None
        self.direct_decode_letter_label_1 = None
        self.direct_decode_letter_label_2 = None
        self.state_prob_by_type_1 = None
        self.state_prob_by_type_2 = None
        self.state_prob_by_type = None
        self.naive_letter_prob = None
        self.naive_prob_letter_label = None
        self.bigram_weighted_prob = None
        self.bigram_weighted_letter_label = None
        self.letter_viterbi_decode_letter_label = None
        self.X_cache = []

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

    def add_bigram_dict(self, bigram_dict, softmax_on_bigram_matrix=False):
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
        else:
            self.bigram_matrix = None
            self.bigram_log_matrix = None

    def fit(self, X, y=None, probability=True, r_i=None):
        self.X_cache = []

        transition_label = letter_label_to_transition_label(
            y, self.LETTERS_TO_DOT, self.region_order
        )
        transition_label = np.array(
            [entry for run in transition_label for entry in run]
        )
        X = np.array([entry_i for run_i in X for entry_i in run_i])
        num_entry, num_timeframe, num_region = X.shape
        X_expanded = X.reshape((num_entry, num_timeframe * num_region))

        if r_i is None:
            for param in self.SVM_params:
                param["probability"] = probability
            self.clfs = Parallel(n_jobs=len(self.SVM_params))(
                delayed(fit_svm)(X_expanded, transition_label[:, i], SVM_params_i)
                for i, SVM_params_i in enumerate(self.SVM_params)
            )
            prob_flatten = np.array(
                Parallel(n_jobs=len(self.clfs))(
                    delayed(clf_pred_proba)(clf_i, X_expanded) for clf_i in self.clfs
                )
            )
            prob_flatten = prob_flatten.transpose(1, 0, 2)
            state_prob_1 = prob_flatten[:, :, 2] + prob_flatten[:, :, 3]
            state_prob_2 = prob_flatten[:, :, 1] + prob_flatten[:, :, 3]
            state_prob_1_label = transition_label // 2
            state_prob_2_label = transition_label % 2
            state_prob_1_e = np.mean(np.abs(state_prob_1 - state_prob_1_label), axis=0)
            state_prob_2_e = np.mean(np.abs(state_prob_2 - state_prob_2_label), axis=0)
            self.state_prob_1_w = 1 - state_prob_1_e
            self.state_prob_2_w = 1 - state_prob_2_e
        else:
            self.SVM_params[r_i]["probability"] = probability
            self.clfs[r_i] = fit_svm(
                X_expanded, transition_label[:, r_i], self.SVM_params[r_i]
            )

        return self

    def predict(
        self,
        X,
        r_i=None,
        bigram_dict=None,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
        insertion_penalty=None,
        token_label=True,
        skip_letter_viterbi=False,
        skip_grammar_viterbi=False,
        state_prob_combine_method="arithmetic",
        use_cache=True,
        softmax_on_naive_letter_prob=False,
        softmax_on_bigram_matrix=False,
    ):
        if self.clfs is None:
            raise Exception("No fitted clfs found")
        # X = np.array(X)
        has_cache = len(self.X_cache) > 0
        cache_is_valid = True
        if has_cache:
            if len(X) == len(self.X_cache):
                for i in range(len(X)):
                    if not np.allclose(X[i], self.X_cache[i]):
                        cache_is_valid = False
                        break
            else:
                cache_is_valid = False

        else:
            cache_is_valid = False
            self.X_cache = X.copy()
        if bigram_dict is not None:
            self.add_bigram_dict(bigram_dict, softmax_on_bigram_matrix)
        if not cache_is_valid:
            use_cache = False
        latest_results = None
        finished_naive_prob_letter_label = False
        finished_letter_viterbi_decode_letter_label = False
        finished_grammar_viterbi_decode_letter_label = False
        if not use_cache:
            X_each_run_len = [len(x_i) for x_i in X]
            X_each_run_start_end = [
                (end - X_each_run_len[j], end)
                for j, end in enumerate(
                    [
                        np.sum(X_each_run_len[: (i + 1)])
                        for i in range(len(X_each_run_len))
                    ]
                )
            ]
            X = np.array([e_i for x_i in X for e_i in x_i])
            num_entry, num_timeframe, num_region = X.shape
            X_expanded = X.reshape((num_entry, num_timeframe * num_region))
            if r_i is None:
                prob_flatten = np.array(
                    Parallel(n_jobs=len(self.clfs))(
                        delayed(clf_pred_proba)(clf_i, X_expanded)
                        for clf_i in self.clfs
                    )
                )
            else:
                if self.clfs[r_i].probability:
                    y_prob = clf_pred_proba(self.clfs[r_i], X_expanded)
                    return [y_prob[s:e] for s, e in X_each_run_start_end]
                else:
                    y = clf_pred(self.clfs[r_i], X_expanded)
                    return [y[s:e] for s, e in X_each_run_start_end]

            prob_flatten = prob_flatten.transpose(1, 0, 2)
            prob = [prob_flatten[s:e] for s, e in X_each_run_start_end]
            self.prob_flatten = prob_flatten
            self.prob = prob
            # NOTE: Each run may have different length. Don't put run as a dimension for the np array
            trans_class = [np.argmax(run_i, axis=-1).astype(np.int32) for run_i in prob]
            self.trans_class = trans_class
            # state_1 = [run_i // 2 for run_i in trans_class]
            # state_2 = [run_i % 2 for run_i in trans_class]
            # label_1 = [np.vstack((s1_run_i, s2_run_i[-1, :])) for s1_run_i, s2_run_i in zip(state_1, state_2)]
            # label_2 = [np.vstack((s1_run_i[ 0, :], s2_run_i)) for s1_run_i, s2_run_i in zip(state_1, state_2)]
            # string_label_1 = [[''.join([str(r) for r in e_i]) for e_i in run_i] for run_i in label_1]
            # string_label_2 = [[''.join([str(r) for r in e_i]) for e_i in run_i] for run_i in label_2]
            # direct_decode_letter_label_1 = [[self.DOT_TO_LETTERS.setdefault(e_i, '?') for e_i in run_i] for run_i in string_label_1]
            # direct_decode_letter_label_2 = [[self.DOT_TO_LETTERS.setdefault(e_i, '?') for e_i in run_i] for run_i in string_label_2]
            state_prob_1 = [run_i[:, :, 2] + run_i[:, :, 3] for run_i in prob]
            state_prob_2 = [run_i[:, :, 1] + run_i[:, :, 3] for run_i in prob]
            state_prob_1 = [
                np.vstack((sp1_run_i, sp2_run_i[-1, :]))
                for sp1_run_i, sp2_run_i in zip(state_prob_1, state_prob_2)
            ]
            state_prob_2 = [
                np.vstack((sp1_run_i[0, :], sp2_run_i))
                for sp1_run_i, sp2_run_i in zip(state_prob_1, state_prob_2)
            ]
            self.state_prob_by_type_1 = state_prob_1
            self.state_prob_by_type_2 = state_prob_2
            state_prob_by_type = []
            if state_prob_combine_method == "arithmetic":
                state_prob_by_type = [
                    (sp1_run_i * self.state_prob_1_w + sp2_run_i * self.state_prob_2_w)
                    / (self.state_prob_1_w + self.state_prob_2_w)
                    for sp1_run_i, sp2_run_i in zip(state_prob_1, state_prob_2)
                ]
            elif state_prob_combine_method == "geometric":
                state_prob_by_type = [
                    np.power(
                        np.power(sp1_run_i, self.state_prob_1_w)
                        * np.power(sp2_run_i, self.state_prob_2_w),
                        1 / (self.state_prob_1_w + self.state_prob_2_w),
                    )
                    for sp1_run_i, sp2_run_i in zip(state_prob_1, state_prob_2)
                ]
            elif state_prob_combine_method == "naive":
                state_prob_by_type = [
                    (sp1_run_i + sp2_run_i) / 2
                    for sp1_run_i, sp2_run_i in zip(state_prob_1, state_prob_2)
                ]
            self.state_prob_by_type = state_prob_by_type
            self.naive_letter_prob = [
                get_naive_letter_prob(
                    run_i,
                    self.region_by_letter_arr,
                    np.empty((run_i.shape[0], self.letters.size)),
                )
                for run_i in state_prob_by_type
            ]
            if softmax_on_naive_letter_prob:
                self.naive_letter_prob = [
                    softmax(run_i) for run_i in self.naive_letter_prob
                ]
            letter_ind = [run_i.argmax(axis=-1) for run_i in self.naive_letter_prob]
            self.naive_prob_letter_label = [self.letters[run_i] for run_i in letter_ind]
            finished_naive_prob_letter_label = True

        if finished_naive_prob_letter_label:
            latest_results = self.naive_prob_letter_label

        if (
            (self.bigram_matrix is not None)
            and (not skip_letter_viterbi)
            and (self.naive_letter_prob is not None)
        ):
            self.bigram_weighted_prob = [
                add_bigram_probabilities(
                    run_i, self.bigram_matrix, np.empty_like(run_i)
                )
                for run_i in self.naive_letter_prob
            ]
            self.bigram_weighted_ind = [
                np.argmax(run_i, axis=-1) for run_i in self.bigram_weighted_prob
            ]
            self.bigram_weighted_letter_label = [
                self.letters[run_i] for run_i in self.bigram_weighted_ind
            ]
            self.letter_viterbi_decode_letter_label = [
                letter_level_bigram_viterbi_decode(
                    run_i, self.bigram_log_matrix, self.letters
                )
                for run_i in self.naive_letter_prob
            ]
            finished_letter_viterbi_decode_letter_label = True
        if finished_letter_viterbi_decode_letter_label:
            latest_results = self.letter_viterbi_decode_letter_label

        if words_node_symbols is None:
            words_node_symbols = self.words_node_symbols
        if words_link_start_end is None:
            words_link_start_end = self.words_link_start_end
        if words_dictionary is None:
            words_dictionary = self.words_dictionary
        if insertion_penalty is None:
            insertion_penalty = self.insertion_penalty

        if (
            (not skip_grammar_viterbi)
            and (words_node_symbols is not None)
            and (words_link_start_end is not None)
            and (words_dictionary is not None)
            and (insertion_penalty is not None)
            and (self.naive_letter_prob is not None)
        ):
            # print('letter_bigram_viterbi_with_grammar_decode...')
            if len(self.naive_letter_prob) == 1:
                # print('letter_bigram_viterbi_with_grammar_decode')
                self.grammar_viterbi_decode_letter_label = [
                    letter_bigram_viterbi_with_grammar_decode(
                        self.naive_letter_prob[0],
                        self.letters,
                        self.bigram_log_matrix,
                        words_node_symbols,
                        words_link_start_end,
                        words_dictionary,
                        insertion_penalty,
                    )
                ]
            else:
                self.grammar_viterbi_decode_letter_label = Parallel(n_jobs=-1)(
                    delayed(letter_bigram_viterbi_with_grammar_decode)(
                        run_i,
                        self.letters,
                        self.bigram_log_matrix,
                        words_node_symbols,
                        words_link_start_end,
                        words_dictionary,
                        insertion_penalty,
                    )
                    for run_i in self.naive_letter_prob
                )

            finished_grammar_viterbi_decode_letter_label = True

        if finished_grammar_viterbi_decode_letter_label:
            latest_results = self.grammar_viterbi_decode_letter_label

        return latest_results


@jit(
    nb.types.Tuple((b1, b1[:]))(b1[:], b1[:], b1[:, :]),
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def resolve_null_helper(
    curr_nodes_mask, non_end_null_node_mask, grammar_node_transition_table
):
    next_nodes_null_mask = np.logical_and(curr_nodes_mask, non_end_null_node_mask)
    has_null = np.any(next_nodes_null_mask)
    if has_null:
        # node_symbol_index_arr[next_nodes_null_mask]
        null_ind = np.arange(len(next_nodes_null_mask))[next_nodes_null_mask]
        curr_nodes_mask = np.logical_xor(curr_nodes_mask, next_nodes_null_mask)
        null_next_nodes = (
            np.count_nonzero(grammar_node_transition_table[null_ind], axis=0) > 0
        )
        curr_nodes_mask = np.logical_or(curr_nodes_mask, null_next_nodes)
    return (has_null, curr_nodes_mask)


@jit(
    b1[:](b1[:], b1[:], b1[:, :], i8),
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def resolve_null(
    curr_nodes_mask,
    non_end_null_node_mask,
    grammar_node_transition_table,
    max_null_resolve_count=3,
):
    next_nodes_null_mask = np.logical_and(curr_nodes_mask, non_end_null_node_mask)
    has_null = np.any(next_nodes_null_mask)
    num_resolve_counter = max_null_resolve_count
    while has_null and (num_resolve_counter > 0):
        has_null, curr_nodes_mask = resolve_null_helper(
            curr_nodes_mask, non_end_null_node_mask, grammar_node_transition_table
        )
        num_resolve_counter -= 1
    if num_resolve_counter == 0:
        raise ValueError(
            "Invalid grammar! Consecutive !Null node chain with length "
            + " larger than max_null_resolve_count"
        )
    return curr_nodes_mask


@jit(
    i4[:](
        f8[:, ::1],
        i4[:, ::1],
        i4[::1],
        i4[::1],
        f8[:, ::1],
        i4[:, ::1],
        f8[:, ::1],
        i4[:, ::1],
        i4[::1],
        i4[::1],
        b1[::1],
        f8[::1],
        b1[:, ::1],
        b1[:, ::1],
        i4[::1],
        f8,
    ),
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def letter_bigram_viterbi_with_grammar_decode_numba_helper(
    log_prob_table,
    node_letters_ind_spelling_mat,
    node_symbol_index_arr,
    node_letters_len,
    transition_log_prob_matrix,
    grammar_link_start_end,
    emission_word_log_prob_table,
    prev_word_table,
    node_letters_ind_first_letter,
    node_letters_ind_last_letter,
    node_len_is_zero_mask,
    node_letters_ind_spelling_transition_modifier,
    grammar_node_transition_table,
    resolve_next_null_cahce,
    reverse_nodes,
    insert_panelty=0.0,
):
    num_entry, num_letter = log_prob_table.shape
    num_node, max_spelling_len = node_letters_ind_spelling_mat.shape
    node_symbol_index_arr = np.arange(num_node)
    emission_word_log_prob_table[:][:] = MIN_FLOAT
    prev_word_table[:][:] = num_node
    node_len_is_zero_mask = node_letters_len == 0
    node_letters_ind_first_letter = node_letters_ind_spelling_mat[:, 0]
    node_letters_ind_first_letter[node_len_is_zero_mask] = num_letter
    node_letters_ind_last_letter[:] = 0
    for i, l in enumerate(node_letters_len):
        node_letters_ind_last_letter[i] = node_letters_ind_spelling_mat[i, l - 1]
    node_letters_ind_last_letter[node_len_is_zero_mask] = num_letter
    node_letters_ind_spelling_transition_modifier[:] = 0
    for i, l in enumerate(node_letters_len):
        if (not (node_len_is_zero_mask[i])) and (l > 0):
            for j in range(l - 1):
                prev_l = node_letters_ind_spelling_mat[i, j]
                next_l = node_letters_ind_spelling_mat[i, j + 1]
                node_letters_ind_spelling_transition_modifier[
                    i
                ] += transition_log_prob_matrix[prev_l, next_l]
    grammar_node_transition_table[:, :] = False
    for indices in grammar_link_start_end:
        grammar_node_transition_table[indices[0], indices[1]] = True
    non_end_null_node_mask = node_len_is_zero_mask.copy()
    dummy_start_node_ind = num_node - 1
    end_node_index = num_node - 2
    non_end_null_node_mask[end_node_index] = False
    resolve_next_null_cahce[:][:] = False
    for i, next_node_mask in enumerate(grammar_node_transition_table):
        resolve_next_null_cahce[i] = resolve_null(
            next_node_mask, non_end_null_node_mask, grammar_node_transition_table, 3
        )

    next_node_indices = node_symbol_index_arr[
        resolve_next_null_cahce[dummy_start_node_ind]
    ]
    curr_node_index = -1
    curr_node_end_i = -1
    bad_start_counter = 0
    for next_node_index in next_node_indices:
        next_node_len = node_letters_len[next_node_index]
        if (next_node_len + curr_node_end_i) >= num_entry:
            bad_start_counter += 1
            continue
        next_nodes_spelling = node_letters_ind_spelling_mat[next_node_index][
            : node_letters_len[next_node_index]
        ]
        next_nodes_spelling_transition_modifier = (
            node_letters_ind_spelling_transition_modifier[next_node_index]
        )
        next_nodes_emission_prob = next_nodes_spelling_transition_modifier
        for letter_i, letter in enumerate(next_nodes_spelling):
            next_nodes_emission_prob += log_prob_table[letter][
                curr_node_end_i + 1 + letter_i
            ]
        if (
            next_nodes_emission_prob
            > emission_word_log_prob_table[
                next_node_index, curr_node_end_i + next_node_len
            ]
        ):
            emission_word_log_prob_table[
                next_node_index, curr_node_end_i + next_node_len
            ] = next_nodes_emission_prob
            prev_word_table[
                next_node_index, curr_node_end_i + next_node_len
            ] = curr_node_index

    if bad_start_counter == len(next_node_indices):
        raise ValueError(
            "Invalid grammar!, Start node spelling length is longer "
            + "than the entire sequence"
        )

    for curr_node_end_i in range(0, num_entry):
        curr_node_mask = emission_word_log_prob_table[:, curr_node_end_i] > MIN_FLOAT
        if not np.any(curr_node_mask):
            continue
        curr_node_indice = node_symbol_index_arr[curr_node_mask]
        all_nodes_end_i = node_letters_len + curr_node_end_i
        all_nodes_end_i_no_overflow = all_nodes_end_i < num_entry
        if not np.any(all_nodes_end_i_no_overflow):
            continue

        for curr_node_index in curr_node_indice:
            cur_log_prob = emission_word_log_prob_table[
                curr_node_index, curr_node_end_i
            ]
            next_nodes_mask = resolve_next_null_cahce[curr_node_index]
            next_nodes_mask = np.logical_and(
                next_nodes_mask, all_nodes_end_i_no_overflow
            )
            has_end = next_nodes_mask[end_node_index]
            next_nodes_mask[end_node_index] = False
            next_nodes_indices = node_symbol_index_arr[next_nodes_mask]

            if next_nodes_indices.size:
                next_nodes_end_i = all_nodes_end_i[next_nodes_mask]
                next_nodes_spelling_mat = node_letters_ind_spelling_mat[next_nodes_mask]
                next_nodes_spelling_len = node_letters_len[next_nodes_mask]
                next_nodes_spelling_transition_modifier = (
                    node_letters_ind_spelling_transition_modifier[next_nodes_mask]
                )
                next_nodes_spelling_probs = np.zeros(
                    len(next_nodes_spelling_len), dtype=np.float64
                )
                for s_i in range(len(next_nodes_spelling_len)):
                    for letter_i in range(next_nodes_spelling_len[s_i]):
                        next_nodes_spelling_probs[s_i] += log_prob_table[
                            curr_node_end_i + 1 + letter_i,
                            next_nodes_spelling_mat[s_i, letter_i],
                        ]
                curr_node_log_prob_trans = transition_log_prob_matrix[
                    node_letters_ind_last_letter[curr_node_index]
                ]
                next_nodes_transition_log_prob_from_current_node = (
                    curr_node_log_prob_trans[
                        node_letters_ind_first_letter[next_nodes_mask]
                    ]
                )
                next_nodes_new_emission_log_probs = (
                    insert_panelty
                    + cur_log_prob
                    + next_nodes_transition_log_prob_from_current_node
                    + next_nodes_spelling_transition_modifier
                    + next_nodes_spelling_probs
                )

                for i, end_i in enumerate(next_nodes_end_i):
                    end_i = next_nodes_end_i[i]
                    next_node_i = next_nodes_indices[i]
                    next_nodes_emission_log_probs_i = emission_word_log_prob_table[
                        next_node_i, end_i
                    ]
                    if (
                        next_nodes_new_emission_log_probs[i]
                        > next_nodes_emission_log_probs_i
                    ):
                        next_nodes_emission_log_probs_i = (
                            next_nodes_new_emission_log_probs[i]
                        )
                        emission_word_log_prob_table[
                            next_node_i, end_i
                        ] = next_nodes_emission_log_probs_i
                        prev_word_table[next_node_i, end_i] = curr_node_index

            if has_end:
                if (
                    cur_log_prob
                    > emission_word_log_prob_table[end_node_index, curr_node_end_i]
                ):
                    emission_word_log_prob_table[
                        end_node_index, curr_node_end_i
                    ] = cur_log_prob
                    prev_word_table[end_node_index, curr_node_end_i] = curr_node_index

    curr_node_entry_index = num_entry - 1
    node_count = 0
    curr_node_index = prev_word_table[end_node_index, curr_node_entry_index]
    reverse_nodes[:] = 0

    while (curr_node_index != -1) and (node_count < num_entry):
        node_len = node_letters_len[curr_node_index]
        reverse_nodes[node_count] = node_len
        reverse_nodes[node_count] = curr_node_index
        node_count = node_count + 1
        curr_node_index = prev_word_table[curr_node_index, curr_node_entry_index]
        curr_node_entry_index -= node_len
    return reverse_nodes[:node_count]


def letter_bigram_viterbi_with_grammar_decode(
    letter_probs_array,
    letter_list,
    transition_log_prob_matrix,
    grammar_node_symbols,
    grammar_link_start_end,
    dictionary,
    insert_panelty=0.0,
):
    num_entry = len(letter_probs_array)
    letter_to_ind = {letter: i for i, letter in enumerate(letter_list)}
    # number_of_symbols = len(grammar_node_symbols)
    node_letters_spelling = [dictionary[symbol] for symbol in grammar_node_symbols]

    log_prob_table = np.log10(letter_probs_array)

    node_letters_len = np.array(
        [len(spell) for spell in node_letters_spelling], dtype=np.int32
    )
    max_node_letters_len = np.max(node_letters_len)
    num_node = len(node_letters_spelling)
    node_letters_ind_spelling_mat = np.zeros((num_node, max_node_letters_len), np.int32)
    for node_i, node in enumerate(node_letters_spelling):
        for l_i, l in enumerate(node):
            node_letters_ind_spelling_mat[node_i, l_i] = letter_to_ind[l]

    # node_symbol_to_ind = {
    #     symbol: i for i, symbol in enumerate(grammar_node_symbols)
    # }
    grammar_link_start_end = np.array(grammar_link_start_end, dtype=np.int32)
    emission_word_log_prob_table = np.empty((num_node, num_entry), dtype=np.float64)
    prev_word_table = np.empty((num_node, num_entry), dtype=np.int32)
    num_node, max_spelling_len = node_letters_ind_spelling_mat.shape
    node_letters_ind_last_letter = np.empty(num_node, dtype=np.int32)
    node_letters_ind_first_letter = np.empty(num_node, dtype=np.int32)
    node_len_is_zero_mask = np.empty_like(node_letters_len, dtype=np.bool_)
    node_symbol_index_arr = np.empty(num_node, dtype=np.int32)
    node_letters_ind_spelling_transition_modifier = np.empty(num_node, dtype=np.float64)
    grammar_node_transition_table = np.empty((num_node, num_node), dtype=np.bool_)
    resolve_next_null_cahce = np.empty((num_node, num_node), dtype=np.bool_)
    reverse_nodes = np.empty(num_entry, dtype=np.int32)
    # print('letter_bigram_viterbi_with_grammar_decode_numba_helper')
    reverse_nodes = letter_bigram_viterbi_with_grammar_decode_numba_helper(
        log_prob_table,
        node_letters_ind_spelling_mat,
        node_symbol_index_arr,
        node_letters_len,
        transition_log_prob_matrix,
        grammar_link_start_end,
        emission_word_log_prob_table,
        prev_word_table,
        node_letters_ind_first_letter,
        node_letters_ind_last_letter,
        node_len_is_zero_mask,
        node_letters_ind_spelling_transition_modifier,
        grammar_node_transition_table,
        resolve_next_null_cahce,
        reverse_nodes,
        insert_panelty,
    )

    letters = [
        j
        for i in range(len(reverse_nodes) - 1, -1, -1)
        for j in node_letters_spelling[reverse_nodes[i]]
    ]
    return letters


def _calculate_SVM_run_i_accuracy(
    r_i, x_train, y_train, x_test, y_test, decoder, c, gamma="scale", use_proba=True
):
    decoder.steps[-1][1].SVM_params[r_i] = {"C": c, "gamma": gamma}
    region_order = decoder.steps[-1][1].region_order
    LETTERS_TO_DOT = decoder.steps[-1][1].LETTERS_TO_DOT
    decoder.fit(
        x_train, y_train, SVMProbDecoder__r_i=r_i, SVMProbDecoder__probability=use_proba
    )
    test_trans_label = letter_label_to_transition_label(
        y_test, LETTERS_TO_DOT, region_order
    )
    test_trans_label_flatten = np.array(
        [e[r_i] for run_i in test_trans_label for e in run_i]
    )
    y_trans_class = decoder.predict(x_test, r_i=r_i)
    y_trans_class_flatten = np.array([e for run_i in y_trans_class for e in run_i])
    if use_proba:
        accuracy = np.array(
            [
                probs[correct_i]
                for probs, correct_i in zip(
                    y_trans_class_flatten, test_trans_label_flatten
                )
            ]
        ).mean()
    else:
        accuracy = accuracy_score(test_trans_label_flatten, y_trans_class_flatten)

    return accuracy


def _SVM_c_gamma_cost(
    c_power,
    gamma_power,
    r_i,
    use_proba,
    cv_decoders,
    x_train_all,
    y_train_all,
    x_test_all,
    y_test_all,
):
    c = 10**c_power
    gamma = 0.1**gamma_power
    acc_all_run = Parallel(n_jobs=len(cv_decoders))(
        delayed(_calculate_SVM_run_i_accuracy)(
            r_i, x_train, y_train, x_test, y_test, decoder, c, gamma, use_proba
        )
        for x_train, y_train, x_test, y_test, decoder in zip(
            x_train_all, y_train_all, x_test_all, y_test_all, cv_decoders
        )
    )
    return -np.mean(acc_all_run)


def _SVM_c_cost(
    c_power,
    r_i,
    use_proba,
    cv_decoders,
    x_train_all,
    y_train_all,
    x_test_all,
    y_test_all,
):
    c = 10**c_power
    acc_all_run = Parallel(n_jobs=len(cv_decoders))(
        delayed(_calculate_SVM_run_i_accuracy)(
            r_i, x_train, y_train, x_test, y_test, decoder, c, use_proba
        )
        for x_train, y_train, x_test, y_test, decoder in zip(
            x_train_all, y_train_all, x_test_all, y_test_all, cv_decoders
        )
    )
    return -np.mean(acc_all_run)


def _tune_SVM_for_r_i(
    r_i,
    use_proba,
    cv_decoders,
    x_train_all,
    y_train_all,
    x_test_all,
    y_test_all,
    C_power_range,
    gamma_power_range,
    tune_gamma=True,
    SVM_n_calls=32,
):
    if tune_gamma:

        def _SVM_c_gamma_cost_partial(c_power, gamma_power):
            return _SVM_c_gamma_cost(
                c_power,
                gamma_power,
                r_i,
                use_proba,
                cv_decoders,
                x_train_all,
                y_train_all,
                x_test_all,
                y_test_all,
            )

        res = dlib.find_min_global(
            _SVM_c_gamma_cost_partial,
            [C_power_range[0], gamma_power_range[0]],
            [C_power_range[1], gamma_power_range[1]],
            SVM_n_calls,
        )
    else:

        def _SVM_c_cost_partial(c_power):
            return _SVM_c_cost(
                c_power,
                r_i,
                use_proba,
                cv_decoders,
                x_train_all,
                y_train_all,
                x_test_all,
                y_test_all,
            )

        res = dlib.find_min_global(
            _SVM_c_cost_partial, [C_power_range[0]], [C_power_range[1]], SVM_n_calls
        )
    return res


def _calculate_run_i_insertion_accuracy(
    x_test, y_test, decoder, insertion_penalty=None
):
    decoder.insertion_penalty = insertion_penalty
    y_pred = decoder.predict(
        x_test, insertion_penalty=insertion_penalty, skip_letter_viterbi=True
    )
    acc = accuracy_score(
        [e for run_i in y_test for e in run_i], [e for run_i in y_pred for e in run_i]
    )
    return acc


def _calculate_all_run_insertion_accuracy(
    x_test_all, y_test_all, cv_decoders, insertion_penalty=None
):
    return Parallel(n_jobs=len(cv_decoders))(
        delayed(_calculate_run_i_insertion_accuracy)(
            x_test, y_test, decoder, insertion_penalty
        )
        for x_test, y_test, decoder in zip(x_test_all, y_test_all, cv_decoders)
    )


class SVMandInsertionPenaltyTunedSVMProbDecoder:
    def __init__(
        self,
        decoder,
        each_fold_n_jobs=5,
        C_power_range=(0, 3),
        gamma_power_range=(0, 3),
        random_state=42,
        insertion_penalty_range=(-20, 0),
        n_splits=None,
        SVM_n_calls=16,
        insertion_n_calls=16,
        tune_gamma=True,
        tune_without_probability=True,
        verbose=True,
    ):
        self.decoder = decoder
        self.C_power_range = C_power_range
        self.gamma_power_range = gamma_power_range
        self.random_state = random_state
        self.insertion_penalty_range = insertion_penalty_range
        self.n_splits = n_splits
        self.insertion_n_calls = insertion_n_calls
        self.SVM_n_calls = SVM_n_calls
        self.each_fold_n_jobs = each_fold_n_jobs
        self.tune_gamma = tune_gamma
        self.tune_without_probability = tune_without_probability
        self.best_SVM_params = None
        self.naive_cm = None
        self.letter_viterbi_cm = None
        self.best_insertion_penalty = 0
        self.verbose = verbose

    def fit(self, X, y, calculate_cm=True):
        previous_n_splits = self.n_splits
        if self.n_splits is None:
            # self.n_splits = int(len(X)/2)
            self.n_splits = len(X)
        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        cv_decoders = []
        x_test_all = []
        y_test_all = []
        x_train_all = []
        y_train_all = []
        for i, train_test_i in enumerate(kf.split(X)):
            train_i, test_i = train_test_i
            x_train_i = [X[i] for i in train_i]
            y_train_i = [y[i] for i in train_i]
            x_test_i = [X[i] for i in test_i]
            y_test_i = [y[i] for i in test_i]
            decoder_i = copy.deepcopy(self.decoder)
            decoder_i_is_valid = True
            for step_name, step_obj in decoder_i.steps:
                if step_name == "ZNormalizeBySub":
                    step_obj.test_group = step_obj.train_group[test_i]
                    step_obj.train_group = step_obj.train_group[train_i]
                    for test_group_i in step_obj.test_group:
                        if test_group_i not in step_obj.train_group:
                            decoder_i_is_valid = False
                            continue
                        step_obj.fit(x_train_i)
            if decoder_i_is_valid:
                x_test_all.append(x_test_i)
                y_test_all.append(y_test_i)
                x_train_all.append(x_train_i)
                y_train_all.append(y_train_i)
                cv_decoders.append(decoder_i)

        self.x_test_all = x_test_all
        self.y_test_all = y_test_all
        self.x_train_all = x_train_all
        self.y_train_all = y_train_all
        self.cv_decoders = cv_decoders
        if self.SVM_n_calls > 0:
            self.tune_SVM()

        if calculate_cm and (self.best_SVM_params is not None):
            self.naive_cm = self.obtain_naive_cm()

        if self.insertion_n_calls > 0:
            self.tune_insertion_penalty()
        self.decoder = self.decoder.fit(X, y)
        self.n_splits = previous_n_splits
        return self

    def obtain_naive_cm(self):
        x_test_all = self.x_test_all
        y_test_all = self.y_test_all
        x_train_all = self.x_train_all
        y_train_all = self.y_train_all
        cv_decoders = self.cv_decoders

        def get_decoder_letter_confusion_matrix(
            x_train, y_train, x_test, y_test, decoder, SVM_params
        ):
            y_unique_labels = np.sort(np.unique([e for run in y_train for e in run]))
            decoder.steps[-1][1].SVM_params = SVM_params
            decoder.fit(x_train, y_train)
            decoder.predict(x_test, skip_letter_viterbi=True, skip_grammar_viterbi=True)
            naive_pred_y = decoder.steps[-1][1].naive_prob_letter_label
            y_test_flatten = [e for run in y_test for e in run]
            naive_pred_y_flatten = [e for run in naive_pred_y for e in run]
            return confusion_matrix(
                [e for run in y_test for e in run],
                [e for run in naive_pred_y for e in run],
                labels=y_unique_labels,
            )

        def get_cv_letter_confusion_matrix(SVM_params):
            naive_cm = Parallel(n_jobs=self.each_fold_n_jobs)(
                delayed(get_decoder_letter_confusion_matrix)(
                    x_train, y_train, x_test, y_test, decoder, SVM_params
                )
                for x_train, y_train, x_test, y_test, decoder in zip(
                    x_train_all, y_train_all, x_test_all, y_test_all, cv_decoders
                )
            )
            return np.sum(naive_cm, axis=0)

        return get_cv_letter_confusion_matrix(self.best_SVM_params)

    def obtain_letter_viterbi_cm(self, bigram_dict):
        x_test_all = self.x_test_all
        y_test_all = self.y_test_all
        x_train_all = self.x_train_all
        y_train_all = self.y_train_all
        cv_decoders = self.cv_decoders
        self.set_grammar_related(bigram_prob_dict=bigram_dict)

        def get_decoder_letter_confusion_matrix(
            x_train, y_train, x_test, y_test, decoder, SVM_params
        ):
            y_unique_labels = np.sort(np.unique([e for run in y_train for e in run]))
            decoder.steps[-1][1].SVM_params = SVM_params
            decoder.predict(x_test, skip_grammar_viterbi=True)
            naive_pred_y = decoder.steps[-1][1].letter_viterbi_decode_letter_label
            return confusion_matrix(
                [e for run in y_test for e in run],
                [e for run in naive_pred_y for e in run],
                labels=y_unique_labels,
            )

        def get_cv_letter_confusion_matrix(SVM_params):
            naive_cm = Parallel(n_jobs=self.each_fold_n_jobs)(
                delayed(get_decoder_letter_confusion_matrix)(
                    x_train, y_train, x_test, y_test, decoder, SVM_params
                )
                for x_train, y_train, x_test, y_test, decoder in zip(
                    x_train_all, y_train_all, x_test_all, y_test_all, cv_decoders
                )
            )
            return np.sum(naive_cm, axis=0)

        return get_cv_letter_confusion_matrix(self.best_SVM_params)

    def tune_SVM(self):
        x_test_all = self.x_test_all
        y_test_all = self.y_test_all
        x_train_all = self.x_train_all
        y_train_all = self.y_train_all
        cv_decoders = self.cv_decoders
        region_order = self.decoder.steps[-1][1].region_order

        svm_params_res = [
            _tune_SVM_for_r_i(
                r_i=i,
                use_proba=(not self.tune_without_probability),
                cv_decoders=cv_decoders,
                x_train_all=x_train_all,
                y_train_all=y_train_all,
                x_test_all=x_test_all,
                y_test_all=y_test_all,
                C_power_range=self.C_power_range,
                gamma_power_range=self.gamma_power_range,
                tune_gamma=self.tune_gamma,
                SVM_n_calls=self.SVM_n_calls,
            )
            for i in range(len(region_order))
        ]

        if self.tune_gamma:
            SVM_params = [
                {"C": 10 ** res[0][0], "gamma": 0.1 ** res[0][1]}
                for res in svm_params_res
            ]
        else:
            SVM_params = [
                {"C": 10 ** res[0][0], "gamma": "scale"} for res in svm_params_res
            ]
        # print(
        # [(res[1], param) for res, param in zip(svm_params_res, SVM_params)]
        # )
        if self.verbose:
            self.SVM_tune_res_pretty_print(svm_params_res, SVM_params, region_order)
        self.best_SVM_params = SVM_params
        self.decoder.steps[-1][1].SVM_params = SVM_params
        for decoder in cv_decoders:
            decoder.steps[-1][1].SVM_params = SVM_params

        def fit_each_fold_decoder(x_train, y_train, decoder):
            return decoder.fit(x_train, y_train)

        cv_decoders = Parallel(n_jobs=self.each_fold_n_jobs)(
            delayed(fit_each_fold_decoder)(x_train, y_train, decoder)
            for x_train, y_train, decoder in zip(x_train_all, y_train_all, cv_decoders)
        )
        self.cv_decoders = cv_decoders

    def SVM_tune_res_pretty_print(self, svm_params_res, SVM_params, region_order):
        print("----- tune_SVM_result -----")
        for res, param, r in zip(svm_params_res, SVM_params, region_order):
            print(
                f'region:{r:>4} cost:{res[1]:7.4f} C:{param["C"]:.4f} gamma:{param["gamma"]:.4f}'
            )
        print("----- --------------- -----")

    def tune_insertion_penalty(self):
        cv_decoders = self.cv_decoders
        x_test_all = self.x_test_all
        y_test_all = self.y_test_all

        insertion_penalties_to_test = np.linspace(
            self.insertion_penalty_range[0],
            self.insertion_penalty_range[1],
            self.insertion_n_calls,
            endpoint=True,
        )

        acc_for_all_fold_all_insPen = Parallel(n_jobs=self.insertion_n_calls)(
            delayed(_calculate_all_run_insertion_accuracy)(
                x_test_all, y_test_all, cv_decoders, ins_pen
            )
            for ins_pen in insertion_penalties_to_test
        )

        acc_for_all_insPen = np.mean(acc_for_all_fold_all_insPen, axis=-1)
        best_acc_ind = np.argmax(acc_for_all_insPen)
        best_insPen = insertion_penalties_to_test[best_acc_ind]
        best_acc = acc_for_all_insPen[best_acc_ind]
        best_acc_mask = acc_for_all_insPen == best_acc
        best_adc_ind = np.arange(self.insertion_n_calls)[best_acc_mask]
        best_insPen_ind = best_adc_ind[len(best_adc_ind) // 2]
        best_insPen = insertion_penalties_to_test[best_insPen_ind]
        self.best_insertion_penalty = best_insPen
        self.decoder.insertion_penalty = best_insPen
        if self.verbose:
            print(f"final insertion_penalty: {best_insPen:8.4f} cost:{best_acc:8.4f}")

    def predict(
        self,
        X,
        r_i=None,
        bigram_dict=None,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
        insertion_penalty=None,
        token_label=True,
        skip_letter_viterbi=False,
        skip_grammar_viterbi=False,
        state_prob_combine_method="arithmetic",
        use_cache=True,
        softmax_on_naive_letter_prob=False,
        softmax_on_bigram_matrix=False,
    ):
        # TODO: investigate this! Make sure the viterbi decode is rerun!!
        if insertion_penalty is None:
            insertion_penalty = self.best_insertion_penalty
        return self.decoder.predict(
            X,
            r_i=r_i,
            bigram_dict=bigram_dict,
            words_node_symbols=words_node_symbols,
            words_link_start_end=words_link_start_end,
            words_dictionary=words_dictionary,
            insertion_penalty=insertion_penalty,
            token_label=token_label,
            skip_letter_viterbi=skip_letter_viterbi,
            skip_grammar_viterbi=skip_grammar_viterbi,
            state_prob_combine_method=state_prob_combine_method,
            use_cache=use_cache,
            softmax_on_naive_letter_prob=softmax_on_naive_letter_prob,
            softmax_on_bigram_matrix=softmax_on_bigram_matrix,
        )

    def predict_svm_transition(self, X):
        prob_trans = self.decoder.predict(X, svm_transition=True)
        return np.array(prob_trans)

    def get_trans_class(self):
        return self.decoder.steps[-1][1].trans_class

    def get_region_order(self):
        return self.decoder.steps[-1][1].region_order

    def get_prob_letter_label(self):
        return self.decoder.steps[-1][1].naive_prob_letter_label

    def get_bigram_weighted_letter_label(self):
        return self.decoder.steps[-1][1].bigram_weighted_letter_label

    def get_letter_viterbi_decode_letter_label(self):
        return self.decoder.steps[-1][1].letter_viterbi_decode_letter_label

    def get_naive_prob_letter_label(self):
        return self.decoder.steps[-1][1].naive_prob_letter_label

    def letter_label_to_transition_label(self, y):
        region_order = self.decoder.steps[-1][1].region_order
        LETTERS_TO_DOT = self.decoder.steps[-1][1].LETTERS_TO_DOT
        return letter_label_to_transition_label(y, LETTERS_TO_DOT, region_order)

    def set_grammar_related(
        self,
        bigram_prob_dict=None,
        words_node_symbols=None,
        words_link_start_end=None,
        words_dictionary=None,
    ):
        num_decoders = len(self.cv_decoders)
        if bigram_prob_dict is not None:
            for i in range(num_decoders):
                self.cv_decoders[i].steps[-1][1].add_bigram_dict(bigram_prob_dict)
            self.decoder.steps[-1][1].add_bigram_dict(bigram_prob_dict)
        if words_node_symbols is not None:
            for i in range(num_decoders):
                self.cv_decoders[i].steps[-1][1].words_node_symbols = words_node_symbols
            self.decoder.steps[-1][1].words_node_symbols = words_node_symbols
        if words_link_start_end is not None:
            for i in range(num_decoders):
                self.cv_decoders[i].steps[-1][
                    1
                ].words_link_start_end = words_link_start_end
            self.decoder.steps[-1][1].words_link_start_end = words_link_start_end
        if words_dictionary is not None:
            for i in range(num_decoders):
                self.cv_decoders[i].steps[-1][1].words_dictionary = words_dictionary
            self.decoder.steps[-1][1].words_dictionary = words_dictionary
