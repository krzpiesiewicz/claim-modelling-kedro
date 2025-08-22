from abc import ABC, abstractmethod

import pandas as pd


from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MAX_ABS_BIAS_BINS, BinsMetricType
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric


class BinsMetric(Metric, ABC):
    def __init__(self, config: Config, n_bins: int, exposure_weighted=False, claim_nb_weighted=False):
        super().__init__(config=config, exposure_weighted=exposure_weighted, claim_nb_weighted=claim_nb_weighted)
        self.n_bins = n_bins

    @abstractmethod
    def eval(self, group_stats_df: pd.DataFrame) -> float:
        ...


class MaxAbsBiasBinsMetric(BinsMetric):
    def __init__(self, config: Config, n_bins: int, **kwargs):
        super().__init__(config=config, n_bins=n_bins, **kwargs)

    def eval(self, group_stats_df: pd.DataFrame) -> float:
        return group_stats_df.abs_bias_deviation.max()

    def _get_name(self) -> str:
        return f"Max Absolute Bias {self.n_bins}-Bins Metric"

    def _get_short_name(self) -> str:
        return f"MAB_{self.n_bins}B"

    def get_enum(self) -> BinsMetricType:
        return MAX_ABS_BIAS_BINS(n_bins=self.n_bins)

    def is_larger_better(self) -> bool:
        return False
