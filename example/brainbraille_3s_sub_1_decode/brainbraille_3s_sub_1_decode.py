import shelve
import msgpack
import msgpack_numpy as m
from fastFMRI.file_helpers import load_file
import numpy as np
import time
from sklearn.svm import SVC
from joblib import Parallel, delayed
from scipy.signal import butter, sosfilt
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from datetime import datetime
from brainbraille_decode.HTK import *
from brainbraille_decode.metrics import *
from brainbraille_decode.preprocessing import *
from brainbraille_decode.viterbi_decoder import *
import os

m.patch()


# ----- Loading data start ----- #
experiment = "3s"
subject = "1"
filename = f"brainbraille_{experiment}_sub_{subject}_workspace"
session_workspace = shelve.open(filename)
for key in session_workspace:
    if key.startswith("_"):
        continue
    globals()[key] = session_workspace[key]
session_workspace.close()

file_path = (
    f"./brainbraille_intermediate_{experiment}_result_extracted_data_sub_{subject}.bin"
)
extracted_data = msgpack.unpackb(load_file(file_path, "rb"))
# ----- Loading data end ----- #

# ----- set up results book keeping data structure start ----- #
pred_trans_label_by_type = {}
y_trans_label_by_type = {}

for r_i, r in enumerate(regressor_types):
    if r not in pred_trans_label_by_type:
        pred_trans_label_by_type[r] = []
    if r not in y_trans_label_by_type:
        y_trans_label_by_type[r] = []

test_label_list = []
naive_prob_letter_label_list = []
naive_cm_list = []

stimulus_letter_viterbi_cm_list = []
mackenzie_soukoreff_letter_viterbi_cm_list = []

stimulus_pred_y_list = []

stimulus_pred_bigram_weighted_letter_label_list = []
stimulus_pred_letter_viterbi_decode_letter_label_list = []

aw2aw_stimulus_pred_y_list = []
aw2aw_stimulus_pred_bigram_weighted_letter_label_list = []
aw2aw_stimulus_pred_letter_viterbi_decode_letter_label_list = []

mackenzie_soukoreff_pred_y_list = []
mackenzie_soukoreff_pred_bigram_weighted_letter_label_list = []
mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list = []

aw2aw_mackenzie_soukoreff_pred_y_list = []
aw2aw_mackenzie_soukoreff_pred_bigram_weighted_letter_label_list = []
aw2aw_mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list = []

# ----- set up results book keeping data structure end ----- #

# ----- Leave one run out cross validation start ----- #
for split_i, split in enumerate(experiment_split):
    s_time = time.time()
    train_sub_i = subs[split["train"]]
    test_sub_i = subs[split["test"]]
    extracted_data_label = extracted_data[split_i]

    train_data = extracted_data_label["train"]["data"]
    train_data = [d[:, 0:6] for d in train_data]
    train_label = extracted_data_label["train"]["label"]
    test_data = extracted_data_label["test"]["data"]
    test_data = [d[:, 0:6] for d in test_data]
    test_label = extracted_data_label["test"]["label"]
    test_label_list += test_label
    
    inner_n_jobs = -1
    data_slicer = DataSlice(EXTRA_FRAME, DELAY_FRAME, EVENT_LEN_FRAME, EVENT_INTERVAL_FRAME)
    svc_param = {
        "C": [1.0, 1000.0],
        "gamma": [0.01, 0.05],
    }

    svm_param_transform = {}
    param_category_keys = []
    log_args_keys = ["C", "gamma"]

    butter_filter = ButterworthBandpassFilter(
        BANDPASS_LOW_CUT, BANDPASS_HIGH_CUT, SF_Hz, BANDPASS_ORDER
    )

    convert_to_trans_prob_CV = BrainBrailleDataToTransProbCV(
        LETTERS_TO_DOT,
        regressor_types,
        SVC(kernel="rbf", cache_size=2000, break_ties=True, class_weight=None),
        data_slicer,
        svc_param,
        param_category_keys=param_category_keys,
        log_args_keys=log_args_keys,
        param_transform_dict=svm_param_transform,
        train_group=train_sub_i,
        test_group=test_sub_i,
        z_normalize=True,
        n_calls=50,
        inner_n_jobs=inner_n_jobs,
    )

    trans_prob_to_state_prob = TransProbToStateProb(LETTERS_TO_DOT, regressor_types)
    state_prob_to_letter_prob = StateProbaToLetterProb(LETTERS_TO_DOT, regressor_types)

    viterbi_decoder = LetterProbaToLetterDecode(
        LETTERS_TO_DOT,
        regressor_types,
        bigram_dict=stimulus_letter_bigram_prob_dict,
        words_node_symbols=stimulus_words_node_symbols,
        words_link_start_end=stimulus_words_link_start_end,
        words_dictionary=unique_stimulus_word_dictionary,
        insertion_penalty=0,
        insertion_penalty_lower=-10.0,
        insertion_penalty_higher=10.0,
        softmax_on_bigram_matrix=True,
        CV_tune_insertion_penalty=True,
        skip_letter_viterbi=False,
        skip_grammar_viterbi=False,
        random_state=42,
        n_calls=32,
    )

    svm_cv_pipe = Pipeline(
        steps=[
            ("bandpass", butter_filter),
            ("transition_probability_CV", convert_to_trans_prob_CV),
            ("transition_to_state_prob", trans_prob_to_state_prob),
            ("state_to_letter_prob", state_prob_to_letter_prob),
            ("viterbi_decoder", viterbi_decoder),
        ]
    )
    fit_start_time = time.time()
    svm_cv_pipe.fit(train_data, train_label)
    test_predict_letter = svm_cv_pipe.predict(test_data)
    fit_end_time = time.time()
    print(f"total time {fit_end_time - fit_start_time}s")

    naive_cm_list.append(state_prob_to_letter_prob.get_naive_letter_cm(test_label))
    test_trans_class = letter_label_to_transition_label(test_label, LETTERS_TO_DOT, regressor_types)
    pred_trans_class = convert_to_trans_prob_CV.get_trans_class()
    naive_acc = np.diag(naive_cm_list[-1]).sum() /  naive_cm_list[-1].sum()
    print(f'--- naive letter {naive_acc:.3f}')

    for r_i, r in enumerate(regressor_types):
        pred_trans_label_by_type[r] += [item[r_i] for run in pred_trans_class for item in run]
        y_trans_label_by_type[r] += [item[r_i] for run in test_trans_class for item in run]

    stimulus_letter_viterbi_cm_list.append(viterbi_decoder.obtain_letter_viterbi_cm(test_label))
    stimulus_pred_y_list += test_predict_letter
    naive_prob_letter_label_list += viterbi_decoder.naive_prob_letter_label
    stimulus_pred_bigram_weighted_letter_label_list += viterbi_decoder.bigram_weighted_letter_label
    stimulus_pred_letter_viterbi_decode_letter_label_list += viterbi_decoder.letter_viterbi_decode_letter_label
    print(f'--- stimulus {accuracy_score(test_label[0], stimulus_pred_y_list[-1]):.3f}')

    viterbi_decoder.re_tune(
        bigram_dict=stimulus_letter_bigram_prob_dict,
        words_node_symbols=aw2aw_stimulus_words_node_symbols,
        words_link_start_end=aw2aw_stimulus_words_link_start_end,
        words_dictionary=unique_stimulus_word_dictionary,
    )

    aw2aw_stimulus_pred_y_list += viterbi_decoder.predict()
    aw2aw_stimulus_pred_bigram_weighted_letter_label_list += viterbi_decoder.bigram_weighted_letter_label
    aw2aw_stimulus_pred_letter_viterbi_decode_letter_label_list += viterbi_decoder.letter_viterbi_decode_letter_label
    mackenzie_soukoreff_letter_viterbi_cm_list.append(viterbi_decoder.obtain_letter_viterbi_cm(test_label))
    print(f'--- aw2aw_stimulus {accuracy_score(test_label[0], aw2aw_stimulus_pred_y_list[-1]):.3f}')

    viterbi_decoder.re_tune(
        bigram_dict=mackenzie_soukoreff_letter_bigram_prob_dict,
        words_node_symbols=mackenzie_soukoreff_words_node_symbols,
        words_link_start_end=mackenzie_soukoreff_words_link_start_end,
        words_dictionary=unique_mackenzie_soukoreff_word_dictionary,
    )
    mackenzie_soukoreff_pred_y_list += viterbi_decoder.predict()
    mackenzie_soukoreff_pred_bigram_weighted_letter_label_list += viterbi_decoder.bigram_weighted_letter_label
    mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list += viterbi_decoder.letter_viterbi_decode_letter_label
    print(f'--- mackenzie_soukoreff {accuracy_score(test_label[0], mackenzie_soukoreff_pred_y_list[-1]):.3f}')

    viterbi_decoder.re_tune(
        bigram_dict=mackenzie_soukoreff_letter_bigram_prob_dict,
        words_node_symbols=aw2aw_mackenzie_soukoreff_words_node_symbols,
        words_link_start_end=aw2aw_mackenzie_soukoreff_words_link_start_end,
        words_dictionary=unique_mackenzie_soukoreff_word_dictionary,
    )
    aw2aw_mackenzie_soukoreff_pred_y_list += viterbi_decoder.predict()
    aw2aw_mackenzie_soukoreff_pred_bigram_weighted_letter_label_list += viterbi_decoder.bigram_weighted_letter_label
    aw2aw_mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list += viterbi_decoder.letter_viterbi_decode_letter_label
    print(f'--- mackenzie_soukoreff aw2aw {accuracy_score(test_label[0], aw2aw_mackenzie_soukoreff_pred_y_list[-1]):.3f}')
    e_time = time.time()
    pred_y_str = [e for run_i in test_predict_letter for e in run_i]
    test_label_str = [e for run_i in test_label for e in run_i]
    print(
        f'fold: {split_i} time: {e_time - s_time}s acc: {accuracy_score(test_label_str, pred_y_str)}\n{"".join(pred_y_str)}\n{"".join(test_label_str)}'
    )

# ----- Leave one run out cross validation end ----- #

# ----- save CV accuracy results start ----- #

SVM_result_cache_path = f'./brainbraille_SVM_result_cache_{STIMULI_LABEL_INTERVAL_s}s_{subject}_{datetime.utcnow().strftime("%m_%d_%H_%M_%S")}.bin'

cached_results = {
    "pred_trans_label_by_type": pred_trans_label_by_type,
    "y_trans_label_by_type": y_trans_label_by_type,
    "test_label_list": test_label_list,
    "naive_prob_letter_label_list": naive_prob_letter_label_list,
    "naive_cm_list": naive_cm_list,
    "stimulus_letter_viterbi_cm_list": stimulus_letter_viterbi_cm_list,
    "mackenzie_soukoreff_letter_viterbi_cm_list": mackenzie_soukoreff_letter_viterbi_cm_list,
    "stimulus_pred_y_list": stimulus_pred_y_list,
    "stimulus_pred_bigram_weighted_letter_label_list": stimulus_pred_bigram_weighted_letter_label_list,
    "stimulus_pred_letter_viterbi_decode_letter_label_list": stimulus_pred_letter_viterbi_decode_letter_label_list,
    "aw2aw_stimulus_pred_y_list": aw2aw_stimulus_pred_y_list,
    "aw2aw_stimulus_pred_bigram_weighted_letter_label_list": aw2aw_stimulus_pred_bigram_weighted_letter_label_list,
    "aw2aw_stimulus_pred_letter_viterbi_decode_letter_label_list": aw2aw_stimulus_pred_letter_viterbi_decode_letter_label_list,
    "mackenzie_soukoreff_pred_y_list": mackenzie_soukoreff_pred_y_list,
    "mackenzie_soukoreff_pred_bigram_weighted_letter_label_list": mackenzie_soukoreff_pred_bigram_weighted_letter_label_list,
    "mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list": mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list,
    "aw2aw_mackenzie_soukoreff_pred_y_list": aw2aw_mackenzie_soukoreff_pred_y_list,
    "aw2aw_mackenzie_soukoreff_pred_bigram_weighted_letter_label_list": aw2aw_mackenzie_soukoreff_pred_bigram_weighted_letter_label_list,
    "aw2aw_mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list": aw2aw_mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list,
}
write_file(
    msgpack.packb(cached_results, use_bin_type=True), SVM_result_cache_path, "wb"
)
# ----- save CV accuracy results end ----- #


# ----- prettyprint results report start ----- #
region_accuracies, confusion_matrices = zip(
    *[
        (
            accuracy_score(y_trans_label_by_type[r], pred_trans_label_by_type[r]),
            confusion_matrix(y_trans_label_by_type[r], pred_trans_label_by_type[r]),
        )
        for r_i, r in enumerate(regressor_types)
    ]
)
print("\t".join(regressor_types))
print("\t".join([f"{acc:.4f}" for acc in region_accuracies]))
#   print(regressor_types)
#   print(np.array_str(np.array(region_accuracies), precision=4, suppress_small=True))

for r, m in zip(regressor_types, confusion_matrices):
    print(f"\n{r}")
    print("\n".join([",".join([f"{w:4}" for w in line]) for line in m]))

accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(test_label_list, naive_prob_letter_label_list)
    ]
)

print(
    f"naive accuracy:{np.mean(accuracy_list):6.4f} std:{np.std(accuracy_list):6.4f}\n"
)
# info_per_select =  [information_transfer_per_selection(mackenzie_soukoreff_letter_prior_prob_dict, cm) for cm in naive_cm_list]
all_label = [l for run in test_label_list for l in run]
num_label = np.unique(all_label).size
all_naive_pred = [l for run in naive_prob_letter_label_list for l in run]
# cm = np.sum(naive_cm_list, axis=0)
cm_list = [
    confusion_matrix(label, pred_y)
    for label, pred_y in zip(test_label_list, naive_prob_letter_label_list)
]
cm = np.sum(cm_list, axis=0)
# letter_cm = np.sum( , axis=0)
naive_info_per_select = naive_information_transfer_per_selection(
    num_label, accuracy_score(all_label, all_naive_pred)
)
better_info_per_select = information_transfer_per_selection(
    mackenzie_soukoreff_letter_prior_prob_dict, cm
)
print(
    f" Naive information transfer per selection: {naive_info_per_select:6.4f} ITR: {60 / STIMULI_LABEL_INTERVAL_s * naive_info_per_select:6.4f}"
)
print(
    f"Better information transfer per selection: {better_info_per_select:6.4f} ITR: {60 / STIMULI_LABEL_INTERVAL_s * better_info_per_select:6.4f}"
)
# print(f'naive info_per_select: {np.mean(info_per_select)}\t{np.std(info_per_select)}')
# bad_info_per_select = [naive_information_transfer_per_selection(27, acc) for acc in accuracy_list]
# print(f'naive info_per_select: {np.mean(bad_info_per_select)}\t{np.std(bad_info_per_select)}')
# print(classification_report(test_label_list[0], naive_prob_letter_label_list[0]))

print("\n====================stimulus grammar====================")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, stimulus_pred_bigram_weighted_letter_label_list
        )
    ]
)
print(f"bigram:\n{np.mean(accuracy_list):6.4f}\t{np.std(accuracy_list):6.4f}")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, stimulus_pred_letter_viterbi_decode_letter_label_list
        )
    ]
)
letter_viterbi_mean_acc = np.mean(accuracy_list)
print(
    f"letter viterbi accuracy: {letter_viterbi_mean_acc:6.4f}\tstd:{np.std(accuracy_list):6.4f}"
)
# info_per_select =  [information_transfer_per_selection(stimulus_letter_prior_prob_dict, cm) for cm in stimulus_letter_viterbi_cm_list]
# cm = np.sum(stimulus_letter_viterbi_cm_list, axis=0)
cm_list = [
    confusion_matrix(label, pred_y)
    for label, pred_y in zip(
        test_label_list, stimulus_pred_letter_viterbi_decode_letter_label_list
    )
]
cm = np.sum(cm_list, axis=0)
naive_info_per_select = naive_information_transfer_per_selection(
    num_label, letter_viterbi_mean_acc
)
better_info_per_select = information_transfer_per_selection(
    stimulus_letter_prior_prob_dict, cm
)
print(
    f" Naive letter viterbi information transfer per selection: {naive_info_per_select:6.4f} ITR: {60 / STIMULI_LABEL_INTERVAL_s * naive_info_per_select:6.4f}"
)
print(
    f"Better letter viterbi information transfer per selection: {better_info_per_select:6.4f} ITR: {60 / STIMULI_LABEL_INTERVAL_s * better_info_per_select:6.4f}"
)

print("grammar viterbi:")
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_list, stimulus_pred_y_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_list, stimulus_pred_y_list)
    ]
)
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(test_label_list, stimulus_pred_y_list)
    ]
)
print(np.array2string(accuracy_list, precision=4, suppress_small=True))
print(
    f"classification result:\naccuracy:{np.mean(accuracy_list):6.4f} {np.std(accuracy_list):6.4f}"
)
print("-----------")
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(f"letter acc:\n{np.mean(acc_list):6.4f} {np.std(acc_list):6.4f}\n")
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(f"letter corr:\n{np.mean(corr_list):6.4f} {np.std(corr_list):6.4f}")
print("-----------")
test_label_word_list = [
    letter_label_to_word_label(pred) for pred in stimulus_pred_y_list
]
pred_y_word_list = [letter_label_to_word_label(label) for label in test_label_list]
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(
    f"word results:\nAccuracy:{np.mean(acc_list):6.4f} std:{np.std(acc_list):6.4f}\n"
)
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(
    f"word corr results:\nCorrect:{np.mean(corr_list):6.4f} std:{np.std(corr_list):6.4f}"
)
print("-----------")

print("\n=================aw2aw stimulus grammar==================")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, aw2aw_stimulus_pred_bigram_weighted_letter_label_list
        )
    ]
)
print(f"bigram:\n{np.mean(accuracy_list):6.4f} std:{np.std(accuracy_list):6.4f}")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list,
            aw2aw_stimulus_pred_letter_viterbi_decode_letter_label_list,
        )
    ]
)
print(
    f"letter viterbi:\n{np.mean(accuracy_list):6.4f} std:{np.std(accuracy_list):6.4f}"
)

print("grammar viterbi:")
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_list, aw2aw_stimulus_pred_y_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_list, aw2aw_stimulus_pred_y_list)
    ]
)
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(test_label_list, aw2aw_stimulus_pred_y_list)
    ]
)
print(np.array2string(accuracy_list, precision=4, suppress_small=True))
print(
    f"classification accuracy:\n{np.mean(accuracy_list):6.4f}\t{np.std(accuracy_list):6.4f}"
)
print("-----------")
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(f"letter acc:\n{np.mean(acc_list):6.4f} std:{np.std(acc_list):6.4f}\n")
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(f"letter corr:\n{np.mean(corr_list):6.4f} std:{np.std(corr_list):6.4f}")
print("-----------")
test_label_word_list = [
    letter_label_to_word_label(pred) for pred in aw2aw_stimulus_pred_y_list
]
pred_y_word_list = [letter_label_to_word_label(label) for label in test_label_list]
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(f"word acc:\n{np.mean(acc_list):6.4f} std:{np.std(acc_list):6.4f}\n")
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(
    f"word corr results:\nCorrect:{np.mean(corr_list):6.4f} std:{np.std(corr_list):6.4f}"
)
print("-----------")

print("====================mackenzie_soukoreff grammar====================")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list,
            mackenzie_soukoreff_pred_bigram_weighted_letter_label_list,
        )
    ]
)
print(f"bigram:\n{np.mean(accuracy_list):6.4f} std: {np.std(accuracy_list):6.4f}")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list,
            mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list,
        )
    ]
)
letter_viterbi_mean_acc = np.mean(accuracy_list)
print(
    f"letter viterbi:\n{letter_viterbi_mean_acc} std:{np.std(accuracy_list):6.4f}"
)

# cm = np.sum(mackenzie_soukoreff_letter_viterbi_cm_list, axis=0)
cm_list = [
    confusion_matrix(label, pred_y)
    for label, pred_y in zip(
        test_label_list,
        mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list,
    )
]
cm = np.sum(cm_list, axis=0)
naive_info_per_select = naive_information_transfer_per_selection(
    num_label, letter_viterbi_mean_acc
)
# print(mackenzie_soukoreff_letter_viterbi_cm_list)
better_info_per_select = information_transfer_per_selection(
    mackenzie_soukoreff_letter_prior_prob_dict, cm
)
print(
    f" Naive letter viterbi information transfer per selection: {naive_info_per_select:6.4f}\tITR: {60 / STIMULI_LABEL_INTERVAL_s * naive_info_per_select:6.4f}"
)
print(
    f"Better letter viterbi information transfer per selection: {better_info_per_select:6.4f}\tITR: {60 / STIMULI_LABEL_INTERVAL_s * better_info_per_select:6.4f}"
)

print("grammar viterbi:")
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_list, mackenzie_soukoreff_pred_y_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_list, mackenzie_soukoreff_pred_y_list)
    ]
)
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(test_label_list, mackenzie_soukoreff_pred_y_list)
    ]
)
print(np.array2string(np.array(accuracy_list), precision=4, suppress_small=True))

print(
    f"classification results:\nAccuracy: {np.mean(accuracy_list):6.4f} std: {np.std(accuracy_list):6.4f}"
)
print("-----------")
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(
    f"letter results:\nAccuracy:{np.mean(acc_list):6.4f} std:{np.std(acc_list):6.4f}\n"
)
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(f"letter corr:\n{np.mean(corr_list):6.4f}\t{np.std(corr_list):6.4f}")
print("-----------")
test_label_word_list = [
    letter_label_to_word_label(pred) for pred in mackenzie_soukoreff_pred_y_list
]
pred_y_word_list = [letter_label_to_word_label(label) for label in test_label_list]
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(f"word acc:\n{np.mean(acc_list):6.4f}\t{np.std(acc_list):6.4f}\n")
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(f"word corr:\n{np.mean(corr_list):6.4f}\t{np.std(corr_list):6.4f}")
print("-----------")

print("=================aw2aw mackenzie_soukoreff grammar==================")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list,
            aw2aw_mackenzie_soukoreff_pred_bigram_weighted_letter_label_list,
        )
    ]
)
print(f"bigram:\n{np.mean(accuracy_list):6.4f}\t{np.std(accuracy_list):6.4f}")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list,
            aw2aw_mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list,
        )
    ]
)
print(
    f"letter viterbi:\n{np.mean(accuracy_list):6.4f} std: {np.std(accuracy_list):6.4f}"
)
print("grammar viterbi:")
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(
            test_label_list, aw2aw_mackenzie_soukoreff_pred_y_list
        )
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(
            test_label_list, aw2aw_mackenzie_soukoreff_pred_y_list
        )
    ]
)
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, aw2aw_mackenzie_soukoreff_pred_y_list
        )
    ]
)
print(np.array2string(accuracy_list, precision=4, suppress_small=True))
print(
    f"classification accuracy:\n{np.mean(accuracy_list):6.4f} std: {np.std(accuracy_list):6.4f}"
)
print("-----------")
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(f"letter acc:\n{np.mean(acc_list):6.4f}\t{np.std(acc_list):6.4f}\n")
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(f"letter corr:\n{np.mean(corr_list):6.4f}\t{np.std(corr_list):6.4f}")
print("-----------")
test_label_word_list = [
    letter_label_to_word_label(pred)
    for pred in aw2aw_mackenzie_soukoreff_pred_y_list
]
pred_y_word_list = [letter_label_to_word_label(label) for label in test_label_list]
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_word_list, pred_y_word_list)
    ]
)
print(np.array2string(acc_list, precision=4, suppress_small=True))
print(f"word acc:\n{np.mean(acc_list):6.4f}\t{np.std(acc_list):6.4f}\n")
print(np.array2string(corr_list, precision=4, suppress_small=True))
print(f"word corr:\n{np.mean(corr_list):6.4f}\t{np.std(corr_list):6.4f}")
print("-----------")
# ----- prettyprint results report end ----- #