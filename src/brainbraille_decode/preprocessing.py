import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from fastFMRI.roi import (
    get_roi_from_flatten_t,
    get_calib_roi_from_flatten_roi,
    get_aggregated_roi_from_flatten,
)


def flatten_fold(arr):
    return [e_i for run_i in arr for e_i in run_i]


def flatten_feature(arr):
    arr = np.array(arr)
    arr = arr.reshape((arr.shape[0], np.prod(arr.shape[1:])))
    return arr


class ButterworthBandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, lowcut, highcut, fs, order=4, axis=0):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.sos = butter(
            order, [lowcut, highcut], analog=False, btype="band", output="sos", fs=fs
        )
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, axis=None):
        axis_to_use = self.axis if axis is None else axis
        return np.array(
            Parallel(n_jobs=-1)(
                delayed(sosfilt)(self.sos, x_i, axis_to_use) for x_i in X
            )
        )


class ZNormalizeByGroup(BaseEstimator, TransformerMixin):
    def __init__(self, train_group=None, test_group=None):
        self.train_group = np.array(train_group) if train_group is not None else None
        self.test_group = np.array(test_group) if test_group is not None else None
        self.z_norm_params_by_sub = {}
        self.default_mean = None
        self.default_std = None
        self.X = None

    def fit(self, X, y=None, train_group=None):
        if train_group is None:
            train_group = (
                self.train_group if self.train_group is not None else np.zeros(len(X))
            )

        X = np.array(X)
        self.X = X.copy()
        unique_train = np.unique(train_group)
        for sub_i in unique_train:
            sub_mask = self.train_group == sub_i
            X_sub = self.X[sub_mask, :]
            X_sub_mean = X_sub.mean(axis=(0, 1))
            X_sub_std = X_sub.std(axis=(0, 1))
            self.z_norm_params_by_sub[sub_i] = {"mean": X_sub_mean, "std": X_sub_std}
        return self

    def transform(self, X, test_group=None):
        X = np.array(X)
        fitted = False
        train_transform = False
        if self.X is not None:
            fitted = True
            if np.allclose(X.shape, self.X.shape):
                if np.allclose(X, self.X):
                    train_transform = True

        if fitted and train_transform:
            train_group = (
                self.train_group if self.train_group is not None else np.zeros(len(X))
            )
            unique_train_group = np.unique(train_group)
            for sub_i in unique_train_group:
                sub_mask = self.train_group == sub_i
                X_sub = X[sub_mask, :]
                X[sub_mask, :] = (
                    X_sub - self.z_norm_params_by_sub[sub_i]["mean"]
                ) / self.z_norm_params_by_sub[sub_i]["std"]
        else:
            if test_group is None:
                test_group = (
                    self.test_group if self.test_group is not None else np.zeros(len(X))
                )
            unique_test_group = np.unique(test_group)
            for sub_i in unique_test_group:
                sub_mask = self.test_group == sub_i
                X_sub = X[sub_mask, :]
                if sub_i in self.z_norm_params_by_sub:
                    X[sub_mask, :] = (
                        X_sub - self.z_norm_params_by_sub[sub_i]["mean"]
                    ) / self.z_norm_params_by_sub[sub_i]["std"]
        return X


class DataSlice(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        extra_frame,
        delay_frame,
        event_len_frame,
        event_interval_frame,
        feature_mask=None,
        num_trans=0,
    ):
        self.extra_frame = extra_frame
        self.delay_frame = delay_frame
        self.event_len_frame = event_len_frame
        self.event_interval_frame = event_interval_frame
        self.num_trans = num_trans
        self.feature_mask = feature_mask

    def fit(self, X, y=None, num_trans=None):
        if (num_trans is None) or (num_trans <= 0):
            if (y is not None) and (self.num_trans <= 0):
                self.num_trans = len(y[0]) - 1
        else:
            self.num_trans = num_trans
        return self

    def transform(self, X):
        slice_indice_start = (
            np.arange(0, self.event_len_frame * self.num_trans, self.event_len_frame)
            + self.delay_frame
        )
        slice_indice_end = (
            slice_indice_start + self.event_len_frame * 2 + self.extra_frame
        )
        sliced_x = np.array(
            [
                [
                    x_i[start_i:end_i, :]
                    for start_i, end_i in zip(slice_indice_start, slice_indice_end)
                ]
                for x_i in X
            ]
        )
        if self.feature_mask is not None:
            sliced_x = sliced_x[:, :, self.feature_mask]
        return sliced_x


class DataTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, num_delay_frame, num_frame_per_label):
        self.num_delay_frame = num_delay_frame
        self.num_frame_per_label = num_frame_per_label
        self.num_frame_to_trim_at_end = 0

    def fit(self, X, y):
        self.num_frame_to_trim_at_end = int(
            np.mean(
                [
                    len(x_i)
                    - (len(y_i) * self.num_frame_per_label)
                    - self.num_delay_frame
                    for x_i, y_i in zip(X, y)
                ]
            )
        )
        return self

    def transform(self, X):
        return [
            x_i[self.num_delay_frame : len(x_i) - self.num_frame_to_trim_at_end, :]
            for x_i in X
        ]


class ROIandCalibrationExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        t_threshold_quantile=0.98,
        cc_discard_size_ratio=0.5,
        no_overlap=True,
        num_contrast_to_keep=6,
        aggregated_threshold_quantile=0.8,
        aggregated_cc_discard_size_ratio=0.5,
        aggregated_no_overlap=True,
        roi_by_sub=None,
        calib_roi_by_sub=None,
    ):
        self.t_threshold_quantile = t_threshold_quantile
        self.cc_discard_size_ratio = cc_discard_size_ratio
        self.no_overlap = no_overlap
        self.num_contrast_to_keep = num_contrast_to_keep
        self.aggregated_threshold_quantile = aggregated_threshold_quantile
        self.aggregated_cc_discard_size_ratio = aggregated_cc_discard_size_ratio
        self.aggregated_no_overlap = aggregated_no_overlap
        self.roi_by_sub = {}
        if roi_by_sub is not None:
            self.roi_by_sub.update(roi_by_sub)
        self.calib_roi_by_sub = {}
        if calib_roi_by_sub is not None:
            self.calib_roi_by_sub.update(calib_roi_by_sub)

    def fit(self, X, y=None):
        for x_i in X:
            x_i["roi"] = get_roi_from_flatten_t(
                x_i["t_val"],
                x_i["motor_mask"],
                t_threshold_quantile=self.t_threshold_quantile,
                cc_discard_size_ratio=self.cc_discard_size_ratio,
                no_overlap=self.no_overlap,
                num_contrast_to_keep=self.num_contrast_to_keep,
            )
            x_i["calib_roi"] = get_calib_roi_from_flatten_roi(x_i["roi"])
        subs = np.array([info["sub"] for info in X], dtype=int)
        X_ind = np.arange(len(X))
        unique_subs = np.unique(subs)
        for sub_i in unique_subs:
            sub_i_index = X_ind[subs == sub_i]
            X_sub_i = [X[i] for i in sub_i_index]
            self.roi_by_sub[sub_i] = get_aggregated_roi_from_flatten(
                [X_sub_i_j["roi"] for X_sub_i_j in X_sub_i],
                [X_sub_i_j["motor_mask"] for X_sub_i_j in X_sub_i],
                aggregation_threshold_quantile=self.aggregated_threshold_quantile,
                cc_discard_size_ratio=self.aggregated_cc_discard_size_ratio,
                no_overlap=self.aggregated_no_overlap,
            )
            self.calib_roi_by_sub[sub_i] = get_calib_roi_from_flatten_roi(
                self.roi_by_sub[sub_i]
            )
        return self

    def transform(self, X, y=None):
        extracted_X = np.array(
            [
                np.array(
                    [
                        np.array(
                            x_i["flatten_func_image"][roi_i, :], dtype=np.float64
                        ).sum(axis=0)
                        for roi_i in self.roi_by_sub[int(x_i["sub"])][
                            x_i["motor_mask"], :
                        ].T
                    ]
                ).T
                for x_i in X
            ]
        )
        extracted_X_calib = np.array(
            [
                np.array(
                    x_i["flatten_func_image"][
                        self.calib_roi_by_sub[int(x_i["sub"])][x_i["motor_mask"]]
                    ]
                ).sum(axis=0)
                for x_i in X
            ]
        )

        relative_data_X = [(x_i - x_i[0, :]) / x_i[0, :] for x_i in extracted_X]
        relative_calib_X = [(x_i - x_i[0]) / x_i[0] for x_i in extracted_X_calib]
        calibrated_X = [
            data_i - calib_i[:, np.newaxis]
            for data_i, calib_i in zip(relative_data_X, relative_calib_X)
        ]
        return calibrated_X
