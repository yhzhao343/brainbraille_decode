import os
import time
from datetime import datetime

import msgpack
import msgpack_numpy as m
import numpy as np
from brainbraille_decode.cross_validation import BrainBrailleCVGen
from brainbraille_decode.metrics import (
    accuracy_score,
    confusion_matrix,
    information_transfer_per_selection,
    letter_label_to_word_label,
    naive_information_transfer_per_selection,
    tok_acc,
    tok_corr,
)
from brainbraille_decode.preprocessing import (
    ButterworthBandpassFilter,
    DataSlice,
    ROIandCalibrationExtractor,
)
from brainbraille_decode.viterbi_decoder import (
    BrainBrailleDataToTransProbCV,
    LetterProbaToLetterDecode,
    StateProbaToLetterProb,
    TransProbToStateProb,
    letter_label_to_transition_label,
)
from fastFMRI.file_helpers import load_file, write_file
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

m.patch()


# # Load cached intermediate files

# In[2]:

# FILE_TO_USE = '3S'
# FILE_TO_USE = '1S5_THAD'
FILE_TO_USE = '1S5_FEYI'

info_dict = {
    "3S": {
        "ENV_VAR":"BRAINBRAILLE_INTERMEDIATE_3S_PER_RUN_RAW_DATA_FILE_PATH",
        "file_name":"brainbraille_intermediate_3s_per_run_raw_data.bin",
        "url":"https://gtvault-my.sharepoint.com/:u:/g/personal/yzhao343_gatech_edu/EYfSrpjQzcxKtVXkP1OeZrQBD6N1gNbiA3yqGjM6OhRoGg?e=3Vu9J1"
    },
    "1S5_THAD": {
        "ENV_VAR": "BRAINBRAILLE_INTERMEDIATE_1S5_THAD_PER_RUN_RAW_DATA_FILE_PATH",
        "file_name": "brainbraille_intermediate_thad_1s5_run_raw_data.bin",
        "url": "https://gtvault-my.sharepoint.com/:u:/g/personal/yzhao343_gatech_edu/EdR7OAnM9W5JjD2FcSxWYa4Bs9vtvU157Zug4QiKmHvTvQ?e=6mXYkN"
    },
    "1S5_FEYI": {
        "ENV_VAR": "BRAINBRAILLE_INTERMEDIATE_1S5_FEYI_PER_RUN_RAW_DATA_FILE_PATH",
        "file_name": "brainbraille_intermediate_feyi_1s5_run_raw_data.bin",
        "url": "https://gtvault-my.sharepoint.com/:u:/g/personal/yzhao343_gatech_edu/EUMLUnsFKNNNhkxWtf6q5pkBvqDWYHM817GuBSRywYJZ8A?e=b0JQRf"
    }
}

ENV_VAR = info_dict[FILE_TO_USE]["ENV_VAR"]
file_name = info_dict[FILE_TO_USE]["file_name"]
url = info_dict[FILE_TO_USE]["url"]

if ENV_VAR in os.environ:
    brainbraille_intermediate_per_run_raw_data_file_path = os.environ[ENV_VAR]
else:
    brainbraille_intermediate_per_run_raw_data_file_path = file_name
    if not os.path.exists(brainbraille_intermediate_per_run_raw_data_file_path):
        from onedrivedownloader import download
        # If the link expires, let me know and I will renew the link
        download(
            url=url,
            filename=brainbraille_intermediate_per_run_raw_data_file_path,
        )

brainbraille_3s_per_run_data = msgpack.unpackb(
    load_file(brainbraille_intermediate_per_run_raw_data_file_path, "rb"),
    strict_map_key=False,
)
per_run_data = brainbraille_3s_per_run_data["per_run_data"]
grammar_info = brainbraille_3s_per_run_data["grammar_info"]
LETTERS_TO_DOT = brainbraille_3s_per_run_data["LETTERS_TO_DOT"]


# # Print each run data info

# In[4]:


subs = np.array([info["sub"] for info in per_run_data], dtype=int)
runs = np.array([info["run"] for info in per_run_data], dtype=int)
sess = np.array([info["ses"] for info in per_run_data], dtype=int)
print(f"subs: {np.array2string(subs, max_line_width=120)}")
print(f"runs: {np.array2string(runs, max_line_width=120)}")
print(f"sess: {np.array2string(sess, max_line_width=120)}")
cv_generator = BrainBrailleCVGen(subs, runs, sess)


# # Subject dependent experiments

# In[5]:


RUN_SVM = True
BANDPASS_LOW_CUT = 0.01
BANDPASS_HIGH_CUT = 0.2
BANDPASS_ORDER = 1
DELAY_S = 3
EXTRA_TIME_S = 3
TR_s = per_run_data[0]["TR_s"]
STIMULI_LABEL_INTERVAL_s = per_run_data[0]["EVENT_INTERVAL_S"]
EXTRA_FRAME = int(EXTRA_TIME_S / TR_s)
DELAY_FRAME = int(DELAY_S / TR_s)
SF_Hz = 1 / TR_s
NUM_FRAME_PER_LABEL = int(STIMULI_LABEL_INTERVAL_s / TR_s)
EVENT_LEN_S = per_run_data[0]["EVENT_LEN_S"]
EVENT_INTERVAL_S = per_run_data[0]["EVENT_INTERVAL_S"]
EVENT_LEN_FRAME = int(EVENT_LEN_S / TR_s)
EVENT_INTERVAL_FRAME = int(EVENT_INTERVAL_S / TR_s)
regressor_types = per_run_data[0]["regressor_types"]
inner_n_jobs = -1


# In[6]:


# sub = 1
sub = 4
n = 1
train_index, test_index = cv_generator.sub_dependent_leave_n_run_out(sub=sub, n=n)
print(train_index)
print("------------------------------------------------------")
print(test_index)


# In[8]:


if RUN_SVM:
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

    train_index, test_index = cv_generator.sub_dependent_leave_n_run_out(sub=sub, n=n)

    for fold_i, (train_i, test_i) in enumerate(zip(train_index, test_index)):
        print(
            f"fold_i: {fold_i}\ntrain_runs:{np.array2string(train_i, 120)}\ntest_runs:{np.array2string(test_i, 120)}"
        )
        s_time = time.time()
        train_data = [per_run_data[i] for i in train_i]
        train_label = [d_i["letter_label"] for d_i in train_data]
        test_data = [per_run_data[i] for i in test_i]
        test_label = [d_i["letter_label"] for d_i in test_data]
        test_label_list += test_label
        train_sub_i = [int(train_data_i["sub"]) for train_data_i in train_data]
        test_sub_i = [int(test_data_i["sub"]) for test_data_i in test_data]
        test_label_flatten = [l for run in test_label for l in run]
        
        roi_and_calib_extractor = ROIandCalibrationExtractor()

        data_slicer = DataSlice(
            EXTRA_FRAME,
            DELAY_FRAME,
            EVENT_LEN_FRAME,
            EVENT_INTERVAL_FRAME,
        )
        svc_param = {
            "C": [1.0, 1000.0],
            "gamma": [0.01, 0.05],
        }

        svm_param_transform = {}
        log_args_keys = ["C", "gamma"]

        butter_filter = ButterworthBandpassFilter(
            BANDPASS_LOW_CUT,
            BANDPASS_HIGH_CUT,
            SF_Hz,
            BANDPASS_ORDER,
        )
        param_category_keys = []

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
        state_prob_to_letter_prob = StateProbaToLetterProb(
            LETTERS_TO_DOT, regressor_types
        )

        viterbi_decoder = LetterProbaToLetterDecode(
            LETTERS_TO_DOT,
            regressor_types,
            bigram_dict=grammar_info["stimulus_letter_bigram_prob_dict"],
            words_node_symbols=grammar_info["stimulus_words_node_symbols"],
            words_link_start_end=grammar_info["stimulus_words_link_start_end"],
            words_dictionary=grammar_info["unique_stimulus_word_dictionary"],
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
                ("data_extractor", roi_and_calib_extractor),
                ("bandpass", butter_filter),
                ("transition_probability_CV", convert_to_trans_prob_CV),
                ("transition_to_state_prob", trans_prob_to_state_prob),
                ("state_to_letter_prob", state_prob_to_letter_prob),
                ("viterbi_decoder", viterbi_decoder),
            ]
        )

        fit_start_time = time.time()
        svm_cv_pipe.fit(train_data, train_label)
        # svm_cv_pipe.transform(test_data)
        test_predict_letter = svm_cv_pipe.predict(test_data)
        fit_end_time = time.time()
        print(f"total time {fit_end_time - fit_start_time}s")

        naive_cm_list.append(state_prob_to_letter_prob.get_naive_letter_cm(test_label))
        test_trans_class = letter_label_to_transition_label(
            test_label, LETTERS_TO_DOT, regressor_types
        )
        pred_trans_class = convert_to_trans_prob_CV.get_trans_class()
        naive_acc = np.diag(naive_cm_list[-1]).sum() / naive_cm_list[-1].sum()
        print(f"--- naive letter {naive_acc:.3f}")
        
        for r_i, r in enumerate(regressor_types):
            pred_trans_label_by_type[r] += [
                item[r_i] for run in pred_trans_class for item in run
            ]
            y_trans_label_by_type[r] += [
                item[r_i] for run in test_trans_class for item in run
            ]

        stimulus_letter_viterbi_cm_list.append(
            viterbi_decoder.obtain_letter_viterbi_cm(test_label)
        )
        stimulus_pred_y_list += test_predict_letter
        naive_prob_letter_label_list += viterbi_decoder.naive_prob_letter_label
        stimulus_pred_bigram_weighted_letter_label_list += (
            viterbi_decoder.bigram_weighted_letter_label
        )
        stimulus_pred_letter_viterbi_decode_letter_label_list += (
            viterbi_decoder.letter_viterbi_decode_letter_label
        )
        print(
            f"--- stimulus {accuracy_score(test_label_flatten, [l for run in test_predict_letter for l in run]):.3f}"
        )

        viterbi_decoder.re_tune(
            bigram_dict=grammar_info["stimulus_letter_bigram_prob_dict"],
            words_node_symbols=grammar_info["aw2aw_stimulus_words_node_symbols"],
            words_link_start_end=grammar_info["aw2aw_stimulus_words_link_start_end"],
            words_dictionary=grammar_info["unique_stimulus_word_dictionary"],
        )

        aw2aw_stimulus_pred_y_fold_i = viterbi_decoder.predict()
        aw2aw_stimulus_pred_y_list += aw2aw_stimulus_pred_y_fold_i
        aw2aw_stimulus_pred_bigram_weighted_letter_label_list += (
            viterbi_decoder.bigram_weighted_letter_label
        )
        aw2aw_stimulus_pred_letter_viterbi_decode_letter_label_list += (
            viterbi_decoder.letter_viterbi_decode_letter_label
        )
        mackenzie_soukoreff_letter_viterbi_cm_list.append(
            viterbi_decoder.obtain_letter_viterbi_cm(test_label)
        )
        print(
            f"--- aw2aw_stimulus {accuracy_score(test_label_flatten, [l for run in aw2aw_stimulus_pred_y_fold_i for l in run]):.3f}"
        )

        viterbi_decoder.re_tune(
            bigram_dict=grammar_info["mackenzie_soukoreff_letter_bigram_prob_dict"],
            words_node_symbols=grammar_info["mackenzie_soukoreff_words_node_symbols"],
            words_link_start_end=grammar_info[
                "mackenzie_soukoreff_words_link_start_end"
            ],
            words_dictionary=grammar_info["unique_mackenzie_soukoreff_word_dictionary"],
        )
        mackenzie_soukoreff_pred_y_fold_i = viterbi_decoder.predict()
        mackenzie_soukoreff_pred_y_list += mackenzie_soukoreff_pred_y_fold_i
        mackenzie_soukoreff_pred_bigram_weighted_letter_label_list += (
            viterbi_decoder.bigram_weighted_letter_label
        )
        mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list += (
            viterbi_decoder.letter_viterbi_decode_letter_label
        )
        print(
            f"--- mackenzie_soukoreff {accuracy_score(test_label_flatten, [l for run in mackenzie_soukoreff_pred_y_fold_i for l in run]):.3f}"
        )

        viterbi_decoder.re_tune(
            bigram_dict=grammar_info["mackenzie_soukoreff_letter_bigram_prob_dict"],
            words_node_symbols=grammar_info[
                "aw2aw_mackenzie_soukoreff_words_node_symbols"
            ],
            words_link_start_end=grammar_info[
                "aw2aw_mackenzie_soukoreff_words_link_start_end"
            ],
            words_dictionary=grammar_info["unique_mackenzie_soukoreff_word_dictionary"],
        )
        aw2aw_mackenzie_soukoreff_pred_y_fold_i = viterbi_decoder.predict()
        aw2aw_mackenzie_soukoreff_pred_y_list += aw2aw_mackenzie_soukoreff_pred_y_fold_i
        aw2aw_mackenzie_soukoreff_pred_bigram_weighted_letter_label_list += (
            viterbi_decoder.bigram_weighted_letter_label
        )
        aw2aw_mackenzie_soukoreff_pred_letter_viterbi_decode_letter_label_list += (
            viterbi_decoder.letter_viterbi_decode_letter_label
        )
        print(
            f"--- mackenzie_soukoreff aw2aw {accuracy_score(test_label_flatten, [l for run in aw2aw_mackenzie_soukoreff_pred_y_fold_i for l in run]):.3f}"
        )
        e_time = time.time()
        pred_y_str = [e for run_i in test_predict_letter for e in run_i]
        test_label_str = [e for run_i in test_label for e in run_i]
        print(
            f'fold: {fold_i} time: {e_time - s_time}s acc: {accuracy_score(test_label_str, pred_y_str)}\n{"".join(pred_y_str)}\n{"".join(test_label_str)}'
        )

# In[ ]:


SVM_result_cache_path = f'./brainbraille_SVM_result_cache_{FILE_TO_USE}_{STIMULI_LABEL_INTERVAL_s}s_sub_{sub}_test_size_{n}_{datetime.utcnow().strftime("%m_%d_%H_%M_%S")}.bin'
if RUN_SVM:
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


# In[ ]:


# i = 8
# print(accuracy_score(test_label_list[i], pred_y_list[i]))
# if RUN_SVM or USE_SVM_RESULT_CACHE:
delimiter = "\t"
grammar_decode_ITR = grammar_info["grammar_decode_ITR"]


# delimiter = ','
def pp_array(arr, delimiter="\t"):
    print(
        np.array2string(
            np.array(arr),
            max_line_width=160,
            separator=delimiter,
            formatter={"float_kind": lambda x: f"{x:6.4f}"},
        )[1:-1]
    )


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
    print("\n".join([delimiter.join([f"{w:4}" for w in line]) for line in m]))

accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(test_label_list, naive_prob_letter_label_list)
    ]
)

print("\n====================Naive letter accuracy====================")
naive_acc = np.mean(accuracy_list)
naive_std = np.std(accuracy_list, ddof=1)
print(f"naive accuracy:{naive_acc:6.4f} std:{naive_std:6.4f}\n")
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
    grammar_info["mackenzie_soukoreff_letter_prior_prob_dict"], cm
)
naive_ITR = 60 / STIMULI_LABEL_INTERVAL_s * naive_info_per_select
better_ITR = 60 / STIMULI_LABEL_INTERVAL_s * better_info_per_select
print(
    f" Naive information transfer per selection: {naive_info_per_select:6.4f} ITR: {naive_ITR:6.4f}"
)
print(
    f"Better information transfer per selection: {better_info_per_select:6.4f} ITR: {better_ITR:6.4f}"
)


# print(f'{naive_acc:6.4f}\t{naive_std:6.4f}\t{naive_ITR:6.4f}\t{better_ITR:6.4f}')
# print(delimiter.join(np.array2string))
# print(np.array2string(
#         np.array(),
#         separator=delimiter,
#         formatter={'float_kind': lambda x: f'{x:6.4f}'}
#     )[1:-1]
# )
print("-----------")
pp_array([naive_acc, naive_std, naive_ITR, better_ITR], delimiter)
# print(f'naive info_per_select: {np.mean(info_per_select)}\t{np.std(info_per_select)}')
# bad_info_per_select = [naive_information_transfer_per_selection(27, acc) for acc in accuracy_list]
# print(f'naive info_per_select: {np.mean(bad_info_per_select)}\t{np.std(bad_info_per_select)}')
# print(classification_report(test_label_list[0], naive_prob_letter_label_list[0]))

print("\n====================stim_9 letter grammar====================")

accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, stimulus_pred_bigram_weighted_letter_label_list
        )
    ]
)
bigram_acc = np.mean(accuracy_list)
bigram_std = np.std(accuracy_list, ddof=1)
print(f"bigram:\n{bigram_acc:6.4f}\t{bigram_std:6.4f}")
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, stimulus_pred_letter_viterbi_decode_letter_label_list
        )
    ]
)
letter_viterbi_mean_acc = np.mean(accuracy_list)
letter_viterbi_mean_std = np.std(accuracy_list, ddof=1)
print(
    f"letter viterbi accuracy: {letter_viterbi_mean_acc:6.4f}\tstd:{letter_viterbi_mean_std:6.4f}"
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
    grammar_info["stimulus_letter_prior_prob_dict"], cm
)
naive_ITR = 60 / STIMULI_LABEL_INTERVAL_s * naive_info_per_select
better_ITR = 60 / STIMULI_LABEL_INTERVAL_s * better_info_per_select
print(
    f" Naive letter viterbi information transfer per selection: {naive_info_per_select:6.4f} ITR: {naive_ITR:6.4f}"
)
print(
    f"Better letter viterbi information transfer per selection: {better_info_per_select:6.4f} ITR: {better_ITR:6.4f}"
)
print("-----------")
pp_array(
    [
        bigram_acc,
        bigram_std,
        letter_viterbi_mean_acc,
        letter_viterbi_mean_std,
        naive_ITR,
        better_ITR,
    ],
    delimiter,
)

print("\n====================stim_9 SFFW grammar====================")
print("\ngrammar viterbi:")
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
print(
    np.array2string(accuracy_list, max_line_width=120, precision=4, suppress_small=True)
)
clf_acc = np.mean(accuracy_list)
clf_std = np.std(accuracy_list, ddof=1)
stim_sffw_ITR = grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["stim"]["SFFW"] * clf_acc
print(
    f"classification result:\naccuracy:{clf_acc:6.4f} {clf_std:6.4f} ITR:{stim_sffw_ITR:6.4f}"
)
print("-----------")
l_acc = np.mean(acc_list)
l_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter acc:\n{l_acc:6.4f} {l_acc_std:6.4f}\n")
l_corr = np.mean(corr_list)
l_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter corr:\n{l_corr:6.4f} {l_corr_std:6.4f}")
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
w_acc = np.mean(acc_list)
w_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word results:\nAccuracy:{w_acc:6.4f} std:{w_acc_std:6.4f}\n")
w_corr = np.mean(corr_list)
w_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word corr results:\nCorrect:{w_corr:6.4f} std:{w_corr_std:6.4f}")

print("-----------")
per_selection = (
    grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["stim"]["SFFW"]
    / 60
    * STIMULI_LABEL_INTERVAL_s
)
pp_array(
    [
        per_selection,
        clf_acc,
        clf_std,
        l_acc,
        l_acc_std,
        l_corr,
        l_corr_std,
        w_acc,
        w_acc_std,
        w_corr,
        w_corr_std,
        stim_sffw_ITR,
    ],
    delimiter,
)

print("\n=================stim_9 W2W grammar==================")

print("\ngrammar viterbi:")
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
print(
    np.array2string(accuracy_list, max_line_width=120, precision=4, suppress_small=True)
)
clf_acc = np.mean(accuracy_list)
clf_std = np.std(accuracy_list, ddof=1)
print(f"classification accuracy:\n{clf_acc:6.4f}\t{clf_std:6.4f}")
print("-----------")
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
l_acc = np.mean(acc_list)
l_acc_std = np.std(acc_list, ddof=1)
print(f"letter acc:\n{l_acc:6.4f} std:{l_acc_std:6.4f}\n")
l_corr = np.mean(corr_list)
l_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter corr:\n{l_corr:6.4f} std:{l_corr_std:6.4f}")
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
w_acc = np.mean(acc_list)
w_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word acc:\n{w_acc:6.4f} std:{w_acc_std:6.4f}\n")
w_corr = np.mean(corr_list)
w_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word corr results:\nCorrect:{w_corr:6.4f} std:{w_corr_std:6.4f}")

print("-----------")
per_selection = (
    grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["stim"]["W2W"]
    / 60
    * STIMULI_LABEL_INTERVAL_s
)
stim_sffw_ITR = grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["stim"]["W2W"] * clf_acc
pp_array(
    [
        per_selection,
        clf_acc,
        clf_std,
        l_acc,
        l_acc_std,
        l_corr,
        l_corr_std,
        w_acc,
        w_acc_std,
        w_corr,
        w_corr_std,
        stim_sffw_ITR,
    ],
    delimiter,
)

print("\n====================M&S_500 letter grammar====================")

accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(
            test_label_list, mackenzie_soukoreff_pred_bigram_weighted_letter_label_list
        )
    ]
)
bigram_acc = np.mean(accuracy_list)
bigram_std = np.std(accuracy_list, ddof=1)
print(f"bigram:\n{bigram_acc:6.4f}\t{bigram_std:6.4f}")
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
letter_viterbi_mean_std = np.std(accuracy_list, ddof=1)
print(
    f"letter viterbi accuracy: {letter_viterbi_mean_acc:6.4f}\tstd:{letter_viterbi_mean_std:6.4f}"
)
# info_per_select =  [information_transfer_per_selection(mackenzie_soukoreff_letter_prior_prob_dict, cm) for cm in mackenzie_soukoreff_letter_viterbi_cm_list]
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
better_info_per_select = information_transfer_per_selection(
    grammar_info["stimulus_letter_prior_prob_dict"], cm
)
naive_ITR = 60 / STIMULI_LABEL_INTERVAL_s * naive_info_per_select
better_ITR = 60 / STIMULI_LABEL_INTERVAL_s * better_info_per_select
print(
    f" Naive letter viterbi information transfer per selection: {naive_info_per_select:6.4f} ITR: {naive_ITR:6.4f}"
)
print(
    f"Better letter viterbi information transfer per selection: {better_info_per_select:6.4f} ITR: {better_ITR:6.4f}"
)
print("-----------")
pp_array(
    [
        bigram_acc,
        bigram_std,
        letter_viterbi_mean_acc,
        letter_viterbi_mean_std,
        naive_ITR,
        better_ITR,
    ],
    delimiter,
)


print("\n====================M&S_500 SFFW grammar====================")


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
clf_acc = np.mean(accuracy_list)
clf_std = np.std(accuracy_list, ddof=1)
print(
    np.array2string(
        np.array(accuracy_list), max_line_width=120, precision=4, suppress_small=True
    )
)
print(f"classification results:\nAccuracy: {clf_acc:6.4f} std: {clf_std:6.4f}")
print("-----------")
l_acc = np.mean(acc_list)
l_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter results:\nAccuracy:{l_acc:6.4f} std:{l_acc_std:6.4f}\n")
l_corr = np.mean(corr_list)
l_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter corr:\n{l_acc_std:6.4f}\t{l_corr:6.4f}")
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
w_acc = np.mean(acc_list)
w_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word acc:\n{w_acc:6.4f}\t{w_acc_std:6.4f}\n")
w_corr = np.mean(corr_list)
w_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word corr:\n{w_corr:6.4f}\t{w_corr_std:6.4f}")
print("-----------")

per_selection = (
    grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["MS"]["SFFW"]
    / 60
    * STIMULI_LABEL_INTERVAL_s
)
stim_sffw_ITR = grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["MS"]["SFFW"] * clf_acc

pp_array(
    [
        per_selection,
        clf_acc,
        clf_std,
        l_acc,
        l_acc_std,
        l_corr,
        l_corr_std,
        w_acc,
        w_acc_std,
        w_corr,
        w_corr_std,
        stim_sffw_ITR,
    ],
    delimiter,
)

print("\n=================M&S_500 W2W grammar==================")

print("grammar viterbi:")
acc_list = np.array(
    [
        tok_acc(label, pred_y)
        for label, pred_y in zip(test_label_list, aw2aw_mackenzie_soukoreff_pred_y_list)
    ]
)
corr_list = np.array(
    [
        tok_corr(label, pred_y)
        for label, pred_y in zip(test_label_list, aw2aw_mackenzie_soukoreff_pred_y_list)
    ]
)
accuracy_list = np.array(
    [
        accuracy_score(label, pred_y)
        for label, pred_y in zip(test_label_list, aw2aw_mackenzie_soukoreff_pred_y_list)
    ]
)
print(
    np.array2string(accuracy_list, max_line_width=120, precision=4, suppress_small=True)
)
clf_acc = np.mean(accuracy_list)
clf_std = np.std(accuracy_list, ddof=1)
print(
    f"classification accuracy:\n{np.mean(accuracy_list):6.4f} std: {np.std(accuracy_list, ddof=1):6.4f}"
)
print("-----------")
l_acc = np.mean(acc_list)
l_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter acc:\n{clf_acc:6.4f}\t{clf_std:6.4f}\n")
l_corr = np.mean(corr_list)
l_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"letter corr:\n{l_corr:6.4f}\t{l_corr_std:6.4f}")
print("-----------")
test_label_word_list = [
    letter_label_to_word_label(pred) for pred in aw2aw_mackenzie_soukoreff_pred_y_list
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
w_acc = np.mean(acc_list)
w_acc_std = np.std(acc_list, ddof=1)
print(np.array2string(acc_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word acc:\n{w_acc:6.4f}\t{w_acc_std:6.4f}\n")
w_corr = np.mean(corr_list)
w_corr_std = np.std(corr_list, ddof=1)
print(np.array2string(corr_list, max_line_width=120, precision=4, suppress_small=True))
print(f"word corr:\n{w_corr:6.4f}\t{w_corr_std:6.4f}")
print("-----------")

per_selection = (
    grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["MS"]["W2W"]
    / 60
    * STIMULI_LABEL_INTERVAL_s
)
stim_sffw_ITR = grammar_decode_ITR[STIMULI_LABEL_INTERVAL_s]["MS"]["W2W"] * clf_acc

pp_array(
    [
        per_selection,
        clf_acc,
        clf_std,
        l_acc,
        l_acc_std,
        l_corr,
        l_corr_std,
        w_acc,
        w_acc_std,
        w_corr,
        w_corr_std,
        stim_sffw_ITR,
    ],
    delimiter,
)

