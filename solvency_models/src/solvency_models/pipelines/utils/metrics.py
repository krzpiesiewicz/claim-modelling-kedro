from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn

from solvency_models.pipelines.p01_init.config import Config, MetricEnum


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
            case MetricEnum.POISSON_DEV:
                return MeanPoissonDeviance(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_POISSON_DEV:
                return WeightedMeanPoissonDeviance(config, pred_col=pred_col)
            case MetricEnum.GAMMA_DEV:
                return MeanGammaDeviance(config, pred_col=pred_col)
            case MetricEnum.WEIGHTED_GAMMA_DEV:
                return WeightedMeanGammaDeviance(config, pred_col=pred_col)
            case MetricEnum.SPEARMAN:
                return SpearmanCorrelation(config, pred_col=pred_col)
            case MetricEnum.SPEARMAN_RANK:
                return SpearmanRankCorrelation(config, pred_col=pred_col)
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


class R2(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.r2_score, **kwargs)

    def get_name(self) -> str:
        return "R2"

    def get_short_name(self) -> str:
        return "R2"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.R2


class WeightedR2(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.r2_score, exposure_weighted=True, **kwargs)

    def get_name(self) -> str:
        return "Weighted R2"

    def get_short_name(self) -> str:
        return "wR2"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.WEIGHTED_R2


class MeanPoissonDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_poisson_deviance, **kwargs)

    def get_name(self) -> str:
        return "Mean Poisson Deviance"

    def get_short_name(self) -> str:
        return "MPD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.POISSON_DEV


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


class MeanGammaDeviance(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_gamma_deviance, **kwargs)

    def get_name(self) -> str:
        return "Mean Gamma Deviance"

    def get_short_name(self) -> str:
        return "MGD"

    def get_enum(self) -> MetricEnum:
        return MetricEnum.GAMMA_DEV


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
