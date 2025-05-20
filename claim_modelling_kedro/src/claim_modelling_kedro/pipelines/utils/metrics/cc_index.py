import numpy as np

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum, MetricType
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric


class NormalizedConcentrationIndex(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_normalized_concentration_index, **kwargs)

    @staticmethod
    def _weighted_concentration_index(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None):
        """
        Compute the weighted Concentration Index.

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.
            sample_weight (np.ndarray): Array of sample weights. If None, all weights are set to 1.

        Returns:
            float: Weighted Concentration Index.
        """
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        if sample_weight is None:
            sample_weight = np.ones_like(y_true)

        # Combine all data into a single array
        data = np.asarray(np.c_[y_true, y_pred, sample_weight], dtype=float)

        # Sort by predicted values and ensure stable sorting
        data_sorted = data[np.argsort(data[:, 1], kind="stable")]

        # Compute cumulative sums
        cum_true = np.cumsum(data_sorted[:, 0] * data_sorted[:, 2])  # cumulative sums of true * weight
        cum_weight = np.cumsum(data_sorted[:, 2])  # cumulative sums of weights
        total_true = np.sum(data_sorted[:, 0] * data_sorted[:, 2])  # sum of all true * weight
        sum_of_weights = cum_weight[-1]

        return 1 / 2 - np.sum(cum_true) / total_true / sum_of_weights

    @staticmethod
    def _weighted_normalized_concentration_index(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None):
        """
        Compute the normalized weighted Concentration Index.

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.
            sample_weight (np.ndarray): Array of sample weights. If None, all weights are set to 1.

        Returns:
            float: Normalized weighted Concentration Index.
        """
        concentration_index_pred = NormalizedConcentrationIndex._weighted_concentration_index(y_true, y_pred, sample_weight)
        concentration_index_true = NormalizedConcentrationIndex._weighted_concentration_index(y_true, y_true, sample_weight)

        # Avoid division by zero
        if concentration_index_true == 0:
            return 0.0

        return concentration_index_pred / concentration_index_true

    def _get_name(self) -> str:
        return "Normalized Concentration Index"

    def _get_short_name(self) -> str:
        return "NCI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_CONCENTRATION_INDEX
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_CONCENTRATION_INDEX
        return MetricEnum.CONCENTRATION_INDEX

    def is_larger_better(self) -> bool:
        return True
