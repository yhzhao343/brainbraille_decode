import numpy as np
import numba as nb
from numba import jit, prange, f8, i4, u4, i8, b1
import sys
import numba as nb
from .preprocessing import flatten_fold, flatten_feature
from sklearn.model_selection import StratifiedShuffleSplit
from .metrics import accuracy_score
from functools import partial
from joblib import Parallel, delayed
from lipo import GlobalOptimizer
from copy import deepcopy
from .lm import (
    letter_label,
)


@jit(
    nb.types.Tuple((f8[:, ::1], u4[::1]))(f8[:, ::1], b1[:, ::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def state_proba_to_letter_proba(state_proba, LETTERS_TO_DOT_array):
    num_t, num_state = len(state_proba), len(LETTERS_TO_DOT_array)
    naive_state_proba = np.empty((num_t, num_state), dtype=np.float64)
    naive_state_ind = np.empty(num_t, dtype=np.uint32)
    for time_i in prange(len(state_proba)):
        state_proba_i = state_proba[time_i]
        for letter_i in range(len(LETTERS_TO_DOT_array)):
            letter_mask = LETTERS_TO_DOT_array[letter_i]
            naive_state_proba[time_i, letter_i] = np.prod(
                state_proba_i[letter_mask]
            ) * np.prod(1 - state_proba_i[~letter_mask])
        naive_state_proba[time_i, :] /= naive_state_proba[time_i, :].sum()
        naive_state_ind[time_i] = np.argmax(naive_state_proba[time_i, :])
    return naive_state_proba, naive_state_ind


def get_run_start_end_index(X):
    X_each_run_len = [len(x_i) for x_i in X]
    X_each_run_start_end = [
        (end - X_each_run_len[j], end)
        for j, end in enumerate(
            [np.sum(X_each_run_len[: (i + 1)]) for i in range(len(X_each_run_len))]
        )
    ]
    return X_each_run_start_end


def pred_to_pred_proba(pred_res, num_class, use_softmax=True):
    result_vector = np.eye(num_class)
    if use_softmax:
        result_vector = softmax(result_vector)
    pred_proba_res = np.zeros((pred_res.size, num_class))
    for i in range(pred_res.size):
        pred_proba_res[i] = result_vector[pred_res[i]]
    return pred_proba_res


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


def letter_label_to_state_label(y, LETTERS_TO_DOT, region_order):
    dot_label = np.array(
        [
            [[LETTERS_TO_DOT[l_i][region] for region in region_order] for l_i in run_i]
            for run_i in y
        ]
    )
    return dot_label


@jit(
    f8[:, ::1](f8[:, :], b1[:, ::1], f8[:, ::1]),
    parallel=False,
    nopython=True,
    fastmath=True,
    cache=True,
)
def get_naive_letter_prob(
    state_prob_by_type_run_i, region_by_letter_arr, naive_letter_prob
):
    t = state_prob_by_type_run_i.shape[0]
    l = region_by_letter_arr.shape[0]
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
    b1[:](b1[:], b1[:], b1[:, :], u4),
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def resolve_null(
    curr_nodes_mask,
    non_end_null_node_mask,
    grammar_node_transition_table,
    max_null_resolve_count=5,
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
        error_message = f"Consecutive !Null node chain with length larger than max_null_resolve_count {max_null_resolve_count}"
        raise ValueError(error_message)
    return curr_nodes_mask


def letter_bigram_viterbi_with_grammar_decode(
    letter_probs_array,
    letter_list,
    transition_log_prob_matrix,
    grammar_node_symbols,
    grammar_link_start_end,
    dictionary,
    insert_panelty=0.0,
    max_null_resolve_count=5,
):
    num_entry = len(letter_probs_array)
    letter_to_ind = {letter: i for i, letter in enumerate(letter_list)}
    # number_of_symbols = len(grammar_node_symbols)
    node_letters_spelling = [dictionary[symbol] for symbol in grammar_node_symbols]

    log_prob_table = np.log1p(letter_probs_array)

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
    num_node = node_letters_ind_spelling_mat.shape[0]
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
        float(insert_panelty),
        max_null_resolve_count,
    )

    letters = [
        j
        for i in range(len(reverse_nodes) - 1, -1, -1)
        for j in node_letters_spelling[reverse_nodes[i]]
    ]
    return letters


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
        u4,
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
    max_null_resolve_count=5,
):
    num_entry, num_letter = log_prob_table.shape
    num_node = node_letters_ind_spelling_mat.shape[0]
    node_symbol_index_arr = np.arange(num_node)
    emission_word_log_prob_table[:][:] = -np.inf
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
            next_node_mask,
            non_end_null_node_mask,
            grammar_node_transition_table,
            max_null_resolve_count,
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
        curr_node_mask = emission_word_log_prob_table[:, curr_node_end_i] > -np.inf
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


def clf_fit_pred_score(clf, train_X_i, train_y_i, test_X_i, test_y_i):
    return accuracy_score(
        test_y_i, deepcopy(clf).fit(train_X_i, train_y_i).predict(test_X_i)
    )


def clf_fit(clf, X, y):
    return deepcopy(clf).fit(X, y)


def clf_pred_proba(clf, X):
    return clf.predict_proba(X)


def param_result_format(param_dict):
    sorted_key_list = sorted(param_dict.keys())
    return "".join(
        [
            f" {key}:{param_dict[key]:8.3f}"
            if isinstance(param_dict[key], (int, float))
            else f" {key}:{param_dict[key]}"
            for key in sorted_key_list
        ]
    )


def tune_clf_results_pretty_print(tune_clf_results, region_order, score_name="acc"):
    result_str = "\n".join(
        [
            f"region:{r:>4} {score_name}:{res_i[1]:7.4f} -{param_result_format(res_i[0])}"
            for res_i, r in zip(tune_clf_results, region_order)
        ]
    )
    print(f"----- tune_SVM_result -----\n{result_str}\n----- --------------- -----")


def preprocess_each_fold(
    cv_train_X_i,
    num_trans=None,
    cv_data_slicer_i=None,
    cv_test_X_i=None,
    z_norm_by_group=None,
    fit_z_norm=True,
    flatten_feat_enable=True,
    flatten_fold_enable=True,
):
    if z_norm_by_group is not None:
        if fit_z_norm:
            cv_train_X_i = z_norm_by_group.fit_transform(cv_train_X_i)
        if cv_test_X_i is not None:
            cv_test_X_i = z_norm_by_group.transform(cv_test_X_i)

    if cv_data_slicer_i is not None:
        cv_train_X_i = cv_data_slicer_i.fit_transform(cv_train_X_i, num_trans=num_trans)

        if cv_test_X_i is not None:
            cv_test_X_i = cv_data_slicer_i.transform(cv_test_X_i)

    if flatten_fold_enable:
        cv_train_X_i = flatten_fold(cv_train_X_i)

        if cv_test_X_i is not None:
            cv_test_X_i = flatten_fold(cv_test_X_i)

    if flatten_feat_enable:
        if flatten_fold_enable:
            cv_train_X_i = flatten_feature(cv_train_X_i)
            if cv_test_X_i is not None:
                cv_test_X_i = flatten_feature(cv_test_X_i)
        else:
            cv_train_X_i = [flatten_feature(run_i) for run_i in cv_train_X_i]
            if cv_test_X_i is not None:
                cv_test_X_i = [flatten_feature(run_i) for run_i in cv_test_X_i]

    if cv_test_X_i is not None:
        return cv_train_X_i, cv_test_X_i
    else:
        return cv_train_X_i


def pp_array(arr, delimiter="\t"):
    print(
        np.array2string(
            np.array(arr),
            separator=delimiter,
            formatter={"float_kind": lambda x: f"{x:6.4f}"},
        )[1:-1]
    )


def extract_data_for_hp_tuning(
    train_train_data_i, train_valid_data_i, roi_extract_and_filter_i
):
    train_train_X = roi_extract_and_filter_i.fit_transform(train_train_data_i)
    train_valid_X = roi_extract_and_filter_i.transform(train_valid_data_i)
    return train_train_X, train_valid_X


def lipo_param_search(
    objective_function,
    param_grid,
    param_category_keys,
    log_args_keys,
    flexible_bounds,
    optimum_val_max=1.0,
    maximize=True,
    flexible_bound_threshold=0.1,
    random_state=42,
    n_calls=128,
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
        flexible_bounds=flexible_bounds,
        maximize=maximize,
        flexible_bound_threshold=flexible_bound_threshold,
        random_state=random_state,
    )

    for _ in range(n_calls):
        candidate = optimizer.get_candidate()
        candidate.set(optimizer.function(**candidate.x))
        if optimum_val_max is not None:
            if np.isclose(optimizer.optimum[1], 1.0):
                break

    return optimizer.optimum, optimizer.running_optimum


def clf_predict_proba(clf, X):
    return clf.predict_proba(X)


def clf_predict(clf, X):
    return clf.predict(X)


def clf_fit_each_r(clfs, X, yr, n_jobs=-1):
    clfs = Parallel(n_jobs=n_jobs)(
        delayed(clf_fit)(clf, X, np.array(yr)[:, i]) for i, clf in enumerate(clfs)
    )
    return clfs


def clf_predict_proba_each_r(clfs, X, n_jobs=-1):
    pred_y = Parallel(n_jobs=n_jobs)(delayed(clf_predict_proba)(clf, X) for clf in clfs)
    return np.array(pred_y)


def clf_score_gen(
    clf,
    cv_train_X,
    cv_train_y,
    cv_test_X,
    cv_test_y,
    sub_sample_ratio=0.2,
    random_state=42,
    n_jobs=-1,
):
    if sub_sample_ratio < 1.0:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=sub_sample_ratio, random_state=random_state
        )
        subsample_ind_list = [
            list(sss.split(train_X_i, train_y_i))[0][1]
            for train_X_i, train_y_i in zip(cv_train_X, cv_train_y)
        ]

        cv_train_X, cv_train_y = zip(
            *[
                (np.array(cv_train_X[i])[ind], np.array(cv_train_y[i])[ind])
                for i, ind in enumerate(subsample_ind_list)
            ]
        )

    def clf_score(**kwargs):
        scores = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(clf_fit_pred_score)(
                    deepcopy(clf).set_params(**kwargs),
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
        return avg_acc

    return clf_score


# def tune_lr_for_r()


def tune_clf(
    train_train_extracted_flatten_Xs,
    train_train_state_flatten_labels,
    train_valid_extracted_flatten_Xs,
    train_valid_state_flatten_labels,
    clf,
    param_grid,
    param_category_keys,
    log_args_keys,
    flexible_bounds,
    optimum_val_max=1.0,
    maximize=True,
    flexible_bound_threshold=0.1,
    random_state=42,
    n_calls=128,
    sub_sample_ratio=0.2,
):
    # train_train_state_flatten_labels = [[l_i_j[r_i] for l_i_j in l_i] for l_i in train_train_state_flatten_labels]
    # train_valid_state_flatten_labels = [[l_i_j[r_i] for l_i_j in l_i] for l_i in train_valid_state_flatten_labels]
    func_to_max = clf_score_gen(
        clf,
        train_train_extracted_flatten_Xs,
        train_train_state_flatten_labels,
        train_valid_extracted_flatten_Xs,
        train_valid_state_flatten_labels,
        sub_sample_ratio,
        random_state,
    )
    tune_clf_results, tune_clf_history = lipo_param_search(
        func_to_max,
        param_grid,
        param_category_keys,
        log_args_keys,
        flexible_bounds,
        optimum_val_max=optimum_val_max,
        maximize=maximize,
        flexible_bound_threshold=flexible_bound_threshold,
        random_state=random_state,
        n_calls=n_calls,
    )
    clf = deepcopy(clf).set_params(**tune_clf_results[0])
    return clf, tune_clf_results, tune_clf_history


def get_slices_and_extract_data(
    indx,
    fold_i,
    per_run_data,
    subs,
    train_i,
    test_i,
    train_train_i_list,
    train_valid_i_list,
    roi_extract_and_filter,
    Z_NORM=True,
    verbose=True,
    n_jobs=-1,
):
    train_per_run_data_index = np.array([indx[i] for i in train_i])
    test_per_run_data_index = np.array([indx[i] for i in test_i])
    if verbose:
        print(f"fold_i: {fold_i}")
        print(f"train_runs:{np.array2string(train_per_run_data_index, 120)}")
        print(f"test_runs:{np.array2string(test_per_run_data_index, 120)}")

    train_data_i = [per_run_data[i] for i in train_per_run_data_index]
    train_letter_label_i = [d_i["letter_label"] for d_i in train_data_i]
    train_state_label_i = [d_i["state_label"] for d_i in train_data_i]

    test_data_i = [per_run_data[i] for i in test_per_run_data_index]
    test_letter_label_i = [d_i["letter_label"] for d_i in test_data_i]
    test_state_label_i = [d_i["state_label"] for d_i in test_data_i]

    train_sub_i = [subs[i] for i in train_per_run_data_index]
    test_sub_i = [subs[i] for i in test_per_run_data_index]

    train_train_extracted_Xs, train_valid_extracted_Xs = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(extract_data_for_hp_tuning)(
                [train_data_i[i] for i in train_train_i],
                [train_data_i[i] for i in train_valid_i],
                (
                    deepcopy(roi_extract_and_filter).set_params(
                        z_norm__train_group=[train_sub_i[i] for i in train_train_i],
                        z_norm__test_group=[train_sub_i[i] for i in train_valid_i],
                    )
                    if Z_NORM
                    else deepcopy(roi_extract_and_filter)
                ),
            )
            for train_train_i, train_valid_i in zip(
                train_train_i_list, train_valid_i_list
            )
        )
    )

    train_train_extracted_flatten_Xs = [
        x_i.reshape((np.prod(x_i.shape[:2]), np.prod(x_i.shape[2:])))
        for x_i in train_train_extracted_Xs
    ]
    train_valid_extracted_flatten_Xs = [
        x_i.reshape((np.prod(x_i.shape[:2]), np.prod(x_i.shape[2:])))
        for x_i in train_valid_extracted_Xs
    ]

    train_train_letter_labels = [
        [train_letter_label_i[i] for i in train_train_i]
        for train_train_i in train_train_i_list
    ]
    train_valid_letter_labels = [
        [train_letter_label_i[i] for i in train_valid_i]
        for train_valid_i in train_valid_i_list
    ]
    train_train_state_labels = [
        [train_state_label_i[i] for i in train_train_i]
        for train_train_i in train_train_i_list
    ]
    train_valid_state_labels = [
        [train_state_label_i[i] for i in train_valid_i]
        for train_valid_i in train_valid_i_list
    ]

    train_train_letter_flatten_labels = [
        [j for i in split_i for j in i] for split_i in train_train_letter_labels
    ]
    train_valid_letter_flatten_labels = [
        [j for i in split_i for j in i] for split_i in train_valid_letter_labels
    ]
    train_train_state_flatten_labels = [
        [j for i in split_i for j in i] for split_i in train_train_state_labels
    ]
    train_valid_state_flatten_labels = [
        [j for i in split_i for j in i] for split_i in train_valid_state_labels
    ]

    train_train_state_flatten_labels_by_r = [
        [[e[r_i] for e in run_i] for run_i in train_train_state_flatten_labels]
        for r_i in range(len(train_train_state_flatten_labels[0][0]))
    ]
    train_valid_state_flatten_labels_by_r = [
        [[e[r_i] for e in run_i] for run_i in train_valid_state_flatten_labels]
        for r_i in range(len(train_valid_state_flatten_labels[0][0]))
    ]

    train_letter_flatten_label_i = [j for run_i in train_letter_label_i for j in run_i]
    test_letter_flatten_label_i = [j for run_i in test_letter_label_i for j in run_i]
    train_state_flatten_label_i = [j for run_i in train_state_label_i for j in run_i]
    test_state_flatten_label_i = [j for run_i in test_state_label_i for j in run_i]

    return (
        train_letter_label_i,
        test_letter_label_i,
        train_state_label_i,
        test_state_label_i,
        train_train_extracted_flatten_Xs,
        train_train_state_flatten_labels,
        train_train_letter_flatten_labels,
        train_train_state_flatten_labels_by_r,
        train_valid_extracted_flatten_Xs,
        train_valid_state_flatten_labels,
        train_valid_letter_flatten_labels,
        train_valid_state_flatten_labels_by_r,
        test_sub_i,
        train_sub_i,
        train_state_flatten_label_i,
        train_data_i,
        test_data_i,
    )


def state_prob_to_letter_prob(predict_state_per_run, LETTERS_TO_DOT_array):
    predict_naive_letter_prob_per_run = np.array(
        [
            [
                [
                    (np.prod(d_i[l_mask]) * np.prod(1 - d_i[~l_mask]))
                    for l_mask in LETTERS_TO_DOT_array
                ]
                for d_i in run_i
            ]
            for run_i in predict_state_per_run
        ]
    )

    predict_naive_letter_prob_marg_per_run = np.array(
        [
            run_i / run_i.sum(axis=1)[:, np.newaxis]
            for run_i in predict_naive_letter_prob_per_run
        ]
    )
    predict_naive_letter_prob_letter_index_per_run = [
        np.argmax(run_i, axis=1) for run_i in predict_naive_letter_prob_marg_per_run
    ]
    predict_naive_letter_prob_letter_per_run = [
        [letter_label[l_i] for l_i in run_i]
        for run_i in predict_naive_letter_prob_letter_index_per_run
    ]

    return (
        predict_naive_letter_prob_marg_per_run,
        predict_naive_letter_prob_letter_per_run,
    )


def tune_decode_smoothing(
    train_x,
    train_y,
    uni_count,
    bigram_count,
    initial_proba,
    letter_label,
    decoder,
    smoothing_1d,
    smoothing_2d,
    param_grid,
    param_category_keys,
    log_args_keys,
    flexible_bounds,
    optimum_val_max=1.0,
    maximize=True,
    flexible_bound_threshold=0.1,
    random_state=42,
    n_calls=128,
):
    func_to_max = partial(
        decode_acc_score,
        smoothing_1d=smoothing_1d,
        smoothing_2d=smoothing_2d,
        train_x=train_x,
        train_y=train_y,
        uni_count=uni_count,
        bi_count=bigram_count,
        initial_proba=initial_proba,
        out_label=letter_label,
        decoder=decoder,
    )
    tune_clf_results, tune_clf_history = lipo_param_search(
        func_to_max,
        param_grid,
        param_category_keys,
        log_args_keys,
        flexible_bounds,
        optimum_val_max=optimum_val_max,
        maximize=maximize,
        flexible_bound_threshold=flexible_bound_threshold,
        random_state=random_state,
        n_calls=n_calls,
    )
    decoder = partial(
        decoder,
        **tune_clf_results[0],
        smoothing_1d=smoothing_1d,
        smoothing_2d=smoothing_2d,
        uni_count=uni_count,
        bi_count=bigram_count,
        initial_proba=initial_proba,
        out_label=letter_label,
    )
    return decoder, tune_clf_results, tune_clf_history


def decode_acc_score(
    train_x,
    train_y,
    decoder,
    **kwargs,
):
    pred_y = decoder(
        **kwargs,
        letter_proba_per_run=train_x,
    )
    acc = [accuracy_score(label_i, pred_i) for pred_i, label_i in zip(pred_y, train_y)]
    return np.mean(acc)


def symbol_info_preprocess(
    letter_label, symbol_nodes, symbol_node_trans_pair, node_output_letter
):
    # TODO: add checks to throw error when there are null node besides end_node_ndx still exist
    symbol_node_trans_pair = np.array(symbol_node_trans_pair, dtype=np.uint32)
    letter_2_ind = {l: i for i, l in enumerate(letter_label)}
    symbol_node_out_list, symbol_node_out_spelling = zip(*node_output_letter.items())
    num_symbol_node_out = len(symbol_node_out_list)
    symbol_nodes_out_len = np.array(
        [len(spelling) for spelling in symbol_node_out_spelling], dtype=np.uint32
    )
    max_symbol_node_out_len = np.max(symbol_nodes_out_len)
    symbol_nodes_out_spelling_matrix = np.zeros(
        (num_symbol_node_out, max_symbol_node_out_len), dtype=np.uint32
    )
    for node_i, spelling in enumerate(symbol_node_out_spelling):
        for letter_i, letter in enumerate(spelling):
            symbol_nodes_out_spelling_matrix[node_i][letter_i] = letter_2_ind[letter]

    symbol_node_out_to_ndx = {
        symbol_node: i for i, symbol_node in enumerate(symbol_node_out_list)
    }
    symbol_nodes_spelling_ndx = np.array(
        [symbol_node_out_to_ndx[symbol_node] for symbol_node in symbol_nodes],
        dtype=np.uint32,
    )
    num_symbol_nodes = len(symbol_nodes)
    symbol_node_trans_matrix = np.zeros(
        (num_symbol_nodes, num_symbol_nodes), dtype=np.bool_
    )
    symbol_node_trans = np.zeros((num_symbol_nodes, num_symbol_nodes), dtype=np.bool_)

    dummy_start_node_i = num_symbol_nodes - 1
    end_node_index = num_symbol_nodes - 2

    symbol_node_trans = get_symbol_node_trans(
        symbol_nodes_spelling_ndx,
        symbol_node_trans_pair,
        symbol_nodes_out_len,
        symbol_node_trans_matrix,
        symbol_node_trans,
        n_resolve=5,
        end_node_index=end_node_index,
    )
    return (
        symbol_node_trans,
        symbol_nodes_spelling_ndx,
        symbol_nodes_out_spelling_matrix,
        symbol_nodes_out_len,
        dummy_start_node_i,
        end_node_index,
    )


@jit(
    b1[:, ::1](u4[::1], u4[:, ::1], u4[::1], b1[:, ::1], b1[:, ::1], i4, i4),
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def get_symbol_node_trans(
    symbol_nodes_spelling_ndx,
    symbol_node_trans_pair,
    symbol_nodes_out_len,
    symbol_node_trans_matrix,
    symbol_node_trans_proba,
    n_resolve,
    end_node_index,
):
    """Resolve transition to !NULL state in symbol node transition matrix"""
    num_symbol_nodes = len(symbol_nodes_spelling_ndx)
    for i in range(len(symbol_node_trans_pair)):
        symbol_node_trans_matrix[
            symbol_node_trans_pair[i][0], symbol_node_trans_pair[i][1]
        ] = True
    for resolve_i in range(n_resolve):
        for i in range(num_symbol_nodes):
            if symbol_nodes_out_len[symbol_nodes_spelling_ndx[i]] == 0:
                if i != end_node_index:
                    for j in range(num_symbol_nodes):
                        if symbol_node_trans_matrix[j][i]:
                            symbol_node_trans_matrix[j][i] = False
                            symbol_node_trans_matrix[j] = (
                                symbol_node_trans_matrix[j]
                                | symbol_node_trans_matrix[i]
                            )
    for i in range(num_symbol_nodes):
        for j in range(num_symbol_nodes):
            symbol_node_trans_proba[i][j] = symbol_node_trans_matrix[i][j]
    return symbol_node_trans_proba


@jit(
    f8(f8[:, ::1], f8[:, ::1], u4[::1], u4),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def joint_proba_of_hidden_state_sequence(
    log_emi_proba, log_trans_proba, hidden_state_ind, offset
):
    log_joint_proba = 0
    for i, letter in enumerate(hidden_state_ind):
        log_joint_proba += log_emi_proba[offset + i][letter]
    for i in range(len(hidden_state_ind) - 1):
        log_joint_proba += log_trans_proba[hidden_state_ind[i]][hidden_state_ind[i + 1]]
    return log_joint_proba


@jit(f8(f8[:]), nopython=True, parallel=False, fastmath=True, cache=True)
def logsumexp(x):
    c = x.max()
    return c + np.log1p(np.sum(np.expm1(x - c)))
