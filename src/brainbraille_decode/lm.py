import os
import subprocess
import numpy as np
from fastFMRI.file_helpers import write_file, load_file, delete_file_if_exists
from functools import partial

letter_label=' abcdefghijklmnopqrstuvwxyz'

def add_k(k, counts, dtype=np.float64):
    return np.array(counts, dtype=dtype) + k

def add_k_gen(k, dtype=np.float64):
    return partial(add_k, k=k, dtype=dtype)

def counts_to_proba(counts, smoothing=add_k_gen(1, np.float64)):
    new_counts = smoothing(counts=counts)
    proba = (new_counts.T / new_counts.sum(axis=-1)).T
    return proba

def txt_to_np_array(txt):
    if isinstance(txt, str):
        txt = txt.encode("ascii")
    txt = np.frombuffer(txt, dtype=np.int8)
    # offset the ascii value so a = 1, b = 2, etc
    txt = txt - 0x60
    # set ascii value for " " to 0
    txt[txt == -64] = 0
    # change upper-cased letter's value to lower case
    txt[txt < 0] = txt[txt < 0] + 32
    return txt


def get_one_gram_feat_vector(txt, normalize=False, int_dtype=np.int64, float_dtype=np.float64):
    txt = txt_to_np_array(txt)
    vec = np.zeros(27, dtype=int_dtype)
    # do n-gram count with a default value to prevent proba = 0
    for i in range(vec.size):
        vec[i] += np.sum(txt == i)
    if normalize:
        vec = counts_to_proba(vec, add_k_gen(0, float_dtype))
    return vec


def get_two_gram_feat_vector(txt, normalize=False, int_dtype=np.int64, float_dtype=np.float64):
    txt = txt_to_np_array(txt)
    vec = np.zeros((27, 27), dtype=int_dtype)
    for i, c_0 in enumerate(txt[:-1]):
        c_1 = txt[i + 1]
        vec[c_0][c_1] += 1
    if normalize:
        vec = counts_to_proba(vec, add_k_gen(0, float_dtype))
    return vec


def get_ngram_prob_dict(content_string, n, space_tok="_space_", use_log_prob=True):
    content_string_bigram_string = get_srilm_ngram(
        content_string, n=n, _no_sos="", _no_eos="", _sort=""
    )
    content_string_bigram_string = content_string_bigram_string.split("\n\n\\end\\")[0]
    log_prob_list = []
    for i in range(n, 0, -1):
        splited_string = content_string_bigram_string.split(f"\\{i}-grams:\n")
        content_string_bigram_string = splited_string[0]
        i_gram_string = splited_string[1].rstrip()
        i_gram_string_lines = [line.split() for line in i_gram_string.split("\n")]
        i_gram_string_lines = [
            [item if item != space_tok else " " for item in line[0 : i + 1]]
            for line in i_gram_string_lines
        ]
        log_prob_dict = {}

        probs = np.array([float(line[0]) for line in i_gram_string_lines])
        if not use_log_prob:
            probs = np.power(10, probs)

        for line_i, items in enumerate(i_gram_string_lines):
            if ("<s>" in items) or ("</s>" in items):
                continue
            second_level_key = items[-1]
            if i > 1:
                key = "".join(items[1:-1])
                if key not in log_prob_dict:
                    log_prob_dict[key] = {}
                log_prob_dict[key][second_level_key] = probs[line_i]
            else:
                log_prob_dict[second_level_key] = probs[line_i]
        log_prob_list.append(log_prob_dict)
    return log_prob_list


def get_srilm_ngram(content, n=2, SRILM_PATH=None, **kwargs):
    if SRILM_PATH is None:
        SRILM_PATH = os.environ.get("SRILM_PATH")
    cmd = f"{SRILM_PATH}/ngram-count"
    content_path = "./content.txt"
    write_file(content, content_path)
    ngram_out_path = f"./{n}gram.lm"
    params = ["-text", content_path, "-order", str(n), "-lm", ngram_out_path] + [
        f'{key_value[i].replace("_", "-")}' if i == 0 else str(key_value[i])
        for key_value in kwargs.items()
        for i in range(2)
    ]
    subprocess.run([cmd] + params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ngram_content = load_file(ngram_out_path)
    delete_file_if_exists(content_path)
    delete_file_if_exists(ngram_out_path)
    return ngram_content