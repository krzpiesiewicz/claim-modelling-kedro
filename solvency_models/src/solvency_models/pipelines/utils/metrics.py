from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p01_init.metric_config import TWEEDIE_DEV, WEIGHTED_TWEEDIE_DEV, MetricEnum


class Metric(ABC):
    def __init__(self, config: Config, pred_col: str, sklearn_like_metric: Callable = None, exposure_weighted=False):
        self.config = config
        self.target_col = config.mdl_task.target_col
        self.pred_col = pred_col
        self.sklearn_like_metric = sklearn_like_metric
        self.exposure_weighted = exposure_weighted

    @classmethod
    def from_enum(cls, config: Config, enum: MetricEnum, pred_col: str):
        match enum:
            case MetricEnum.RMSE:
                return RootMeanSquaredError(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_RMSE:
                return WeightedRootMeanSquaredError(config, pred_col=pred_col)
            case MetricEnum.R2:
                return R2(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_R2:
                return WeightedR2(config, pred_col=pred_col)
            case MetricEnum.MBD:
                return MeanBiasDeviation(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_MBD:
                return WeightedMeanBiasDeviation(config, pred_col=pred_col)
            case MetricEnum.POISSON_DEV:
                return MeanPoissonDeviance(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_POISSON_DEV:
                return WeightedMeanPoissonDeviance(config, pred_col=pred_col)
            case MetricEnum.GAMMA_DEV:
                return MeanGammaDeviance(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_GAMMA_DEV:
                return WeightedMeanGammaDeviance(config, pred_col=pred_col)
            case MetricEnum.TWEEDIE_DEV:
                return MeanTweedieDeviance(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_TWEEDIE_DEV:
                return WeightedMeanTweedieDeviance(config, pred_col=pred_col)
            case MetricEnum.SPEARMAN:
                return SpearmanCorrelation(config, pred_col=pred_col)
            case MetricEnum.SPEARMAN_RANK:
                return SpearmanRankCorrelation(config, pred_col=pred_col)
            case MetricEnum.GINI:
                return NormalizedGiniCoefficient(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_GINI:
                return WeightedNormalizedGiniCoefficient(config, pred_col=pred_col)
            # Handle the parametrized cases with TweedieDev(p)
            case TWEEDIE_DEV(p):
                return MeanTweedieDeviance(config, pred_col=pred_col, power=p)
            case WEIGHTED_TWEEDIE_DEV(p):
                return WeightedMeanTweedieDeviance(config, pred_col=pred_col, power=p)
            case _:
                raise ValueError(f"The metric enum: {enum} is not supported by the Metric class.")

    @abstractmethod
    def get_enum(self) -> MetricEnum:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_short_name(self) -> str:
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        pass

    def eval(self, target_df: pd.DataFrame, prediction_df: pd.DataFrame) -> float:
        assert self.sklearn_like_metric is not None
        y_true = target_df[self.target_col]
        y_pred = prediction_df[self.pred_col]

        if self.exposure_weighted:
            sample_weight = target_df[self.config.data.policy_exposure_col]
            return self.sklearn_like_metric(y_true, y_pred, sample_weight=sample_weight)
        else:
            return self.sklearn_like_metric(y_true, y_pred)


class RootMeanSquaredError(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.rmse, **kwargs)

    @staticmethod
    def rmse(y_true, y_pred, **kwargs):
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred, **kwargs)
        return np.sqrt(mse)

    def get_name(self) -> str:
        return "Root Mean Squared Error"

    def get_short_name(self) -> str:
        return "RMSE"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.RMSE

    def is_larger_better(self) -> bool:
        return False


class WeightedRootMeanSquaredError(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.rmse, exposure_weighted=True,
                         **kwargs)

    @staticmethod
    def rmse(y_true, y_pred, **kwargs):
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred, **kwargs)
        return np.sqrt(mse)

    def get_name(self) -> str:
        return "Weighted Root Mean Squared Error"

    def get_short_name(self) -> str:
        return "wRMSE"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_RMSE

    def is_larger_better(self) -> bool:
        return False


class R2(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.r2_score, **kwargs)

    def get_name(self) -> str:
        return "R2"

    def get_short_name(self) -> str:
        return "R2"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.R2

    def is_larger_better(self) -> bool:
        return True


class WeightedR2(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.r2_score, exposure_weighted=True, **kwargs)

    def get_name(self) -> str:
        return "Weighted R2"

    def get_short_name(self) -> str:
        return "wR2"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_R2

    def is_larger_better(self) -> bool:
        return True


class MeanBiasDeviation(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.mbd, **kwargs)

    @staticmethod
    def mbd(y_true, y_pred, **kwargs):
        return np.mean(y_true - y_pred)

    def get_name(self) -> str:
        return "Mean Bias Deviation"

    def get_short_name(self) -> str:
        return "MBD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.MBD  # Assuming you add MBD to MetricEnum

    def is_larger_better(self) -> bool:
        raise Exception(f"{self.get_name()} is not a maximization nor minimization metric - cannot be used as a loss function.")


class WeightedMeanBiasDeviation(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.wmbd, exposure_weighted=True, **kwargs)

    @staticmethod
    def wmbd(y_true, y_pred, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = np.ones_like(y_true)
        # Weighted mean bias deviation
        return np.average(y_true - y_pred, weights=sample_weight)

    def get_name(self) -> str:
        return "Weighted Mean Bias Deviation"

    def get_short_name(self) -> str:
        return "wMBD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_MBD  # Assuming you add WEIGHTED_MBD to MetricEnum

    def is_larger_better(self) -> bool:
        raise Exception(
            f"{self.get_name()} is not a maximization nor minimization metric - cannot be used as a loss function.")


class MeanPoissonDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_poisson_deviance, **kwargs)

    def get_name(self) -> str:
        return "Mean Poisson Deviance"

    def get_short_name(self) -> str:
        return "MPD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.POISSON_DEV

    def is_larger_better(self) -> bool:
        return False


class WeightedMeanPoissonDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_poisson_deviance, exposure_weighted=True,
                         **kwargs)

    def get_name(self) -> str:
        return "Weighted Mean Poisson Deviance"

    def get_short_name(self) -> str:
        return "wMPD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_POISSON_DEV

    def is_larger_better(self) -> bool:
        return False


class MeanGammaDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_gamma_deviance, **kwargs)

    def get_name(self) -> str:
        return "Mean Gamma Deviance"

    def get_short_name(self) -> str:
        return "MGD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.GAMMA_DEV

    def is_larger_better(self) -> bool:
        return False


class WeightedMeanGammaDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_gamma_deviance, exposure_weighted=True,
                         **kwargs)

    def get_name(self) -> str:
        return "Weighted Mean Gamma Deviance"

    def get_short_name(self) -> str:
        return "wMGD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_GAMMA_DEV

    def is_larger_better(self) -> bool:
        return False


class MeanTweedieDeviance(Metric):
    def __init__(self, config: Config, power: float, **kwargs):
        super().__init__(config, **kwargs)
        self.power = power

    @staticmethod
    def sklearn_like_metric(self, y_true, y_pred, **kwargs):
        return partial(sklearn.metrics.mean_tweedie_deviance, power=self.power, **kwargs)

    def get_name(self) -> str:
        return f"Mean Tweedie Deviance (p={self.power})"

    def get_short_name(self) -> str:
        return f"MTD_{self.power}"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.TWEEDIE_DEV

    def is_larger_better(self) -> bool:
        return False


class WeightedMeanTweedieDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, exposure_weighted=True, **kwargs)

    @staticmethod
    def sklearn_like_metric(self, y_true, y_pred, **kwargs):
        return partial(sklearn.metrics.mean_tweedie_deviance, power=self.power, **kwargs)

    def get_name(self) -> str:
        return "Weighted Mean Tweedie Deviance"

    def get_short_name(self) -> str:
        return "wMTD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_TWEEDIE_DEV

    def is_larger_better(self) -> bool:
        return False


class SpearmanCorrelation(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

    def get_name(self) -> str:
        return "Spearman Correlation"

    def get_short_name(self) -> str:
        return "SC"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.SPEARMAN

    def eval(self, target_df: pd.DataFrame, prediction_df: pd.DataFrame) -> float:
        y_true = target_df[self.target_col]
        y_pred = prediction_df[self.pred_col]
        return scipy.stats.spearmanr(y_true, y_pred).statistic

    def is_larger_better(self) -> bool:
        return True


class SpearmanRankCorrelation(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

    def get_name(self) -> str:
        return "Spearman Rank Correlation"

    def get_short_name(self) -> str:
        return "SRC"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.SPEARMAN_RANK

    def eval(self, target_df: pd.DataFrame, prediction_df: pd.DataFrame) -> float:
        y_true = target_df[self.target_col].rank()
        y_pred = prediction_df[self.pred_col].rank()
        return scipy.stats.spearmanr(y_true, y_pred).statistic

    def is_larger_better(self) -> bool:
        return True


def _weighted_gini(y_true, y_pred, sample_weight=None):
    assert len(y_true) == len(y_pred)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    all_data = np.asarray(np.c_[y_true, y_pred, sample_weight, np.arange(len(y_true))], dtype=float)
    # Sort by predicted values, then by index in case of ties
    all_data = all_data[np.lexsort((all_data[:, 3], -1 * all_data[:, 1]))]
    weighted_losses = all_data[:, 0] * all_data[:, 2]
    cumulative_weight = all_data[:, 2].sum()

    weighted_gini_sum = np.cumsum(weighted_losses).sum() / weighted_losses.sum()
    weighted_gini_sum -= (cumulative_weight + 1) / 2.0
    return weighted_gini_sum / cumulative_weight


class NormalizedGiniCoefficient(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.normalized_gini, **kwargs)

    @staticmethod
    def gini(y_true, y_pred):
        return _weighted_gini(y_true, y_pred)
        # assert len(actual) == len(pred)
        # all_data = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        # all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
        # total_losses = all_data[:, 0].sum()
        # gini_sum = all_data[:, 0].cumsum().sum() / total_losses
        # gini_sum -= (len(actual) + 1) / 2.0
        # return gini_sum / len(actual)

    @staticmethod
    def normalized_gini(y_true, y_pred, **kwargs):
        return NormalizedGiniCoefficient.gini(y_true, y_pred) / NormalizedGiniCoefficient.gini(y_true, y_true)

    def get_name(self) -> str:
        return "Normalized Gini Coefficient"

    def get_short_name(self) -> str:
        return "GINI"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.GINI  # Assuming you add GINI to MetricEnum

    def is_larger_better(self) -> bool:
        return True


class WeightedNormalizedGiniCoefficient(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.weighted_normalized_gini, exposure_weighted=True, **kwargs)

    @staticmethod
    def weighted_gini(y_true, y_pred, sample_weight=None):
        return _weighted_gini(y_true, y_pred, sample_weight)

    @staticmethod
    def weighted_normalized_gini(y_true, y_pred, sample_weight=None):
        return WeightedNormalizedGiniCoefficient.weighted_gini(y_true, y_pred, sample_weight) / \
            WeightedNormalizedGiniCoefficient.weighted_gini(y_true, y_true, sample_weight)

    def get_name(self) -> str:
        return "Weighted Normalized Gini Coefficient"

    def get_short_name(self) -> str:
        return "wGINI"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_GINI  # Assuming you add WEIGHTED_GINI to MetricEnum

    def is_larger_better(self) -> bool:
        return True
