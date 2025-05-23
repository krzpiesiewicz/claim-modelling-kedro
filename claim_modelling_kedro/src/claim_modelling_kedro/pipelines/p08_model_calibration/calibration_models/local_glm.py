from abc import ABC
from typing import Dict, Any
import numpy as np
import pandas as pd
import hyperopt as hp
import warnings
from statsmodels.tools.sm_exceptions import DomainWarning

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.mdl_info_config import ModelEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import get_sample_weight
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel
from claim_modelling_kedro.pipelines.p07_data_science.models import StatsmodelsPoissonGLM, StatsmodelsGammaGLM, \
    StatsmodelsTweedieGLM, get_statsmodels_metric
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_models.kernel import KernelEnum, \
    compute_kernel_weights
from claim_modelling_kedro.pipelines.utils.dataframes import index_ordered_by_col
from claim_modelling_kedro.pipelines.utils.metrics import Metric


class LocalStatsmodelsGLMCalibration(CalibrationModel, ABC):
    def __init__(self, config: Config, **kwargs):
        CalibrationModel.__init__(self, config=config, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        sorted_idx = index_ordered_by_col(pure_predictions_df, order_col=self.pure_pred_col)
        pure_predictions_df = pure_predictions_df.loc[sorted_idx, :]
        target_df = target_df.loc[sorted_idx, :]
        sample_weight = get_sample_weight(self.config, target_df)
        sample_weight = sample_weight.loc[sorted_idx]

        self._min_pure_y = pure_predictions_df[self.pure_pred_col].min()
        self._max_pure_y = pure_predictions_df[self.pure_pred_col].max()

        self._k = int(np.ceil(self._frac * len(target_df)))
        self._y_pred = []
        self._y_pure_pred = []

        match self._model_enum:
            case ModelEnum.STATSMODELS_POISSON_GLM:
                model_class = StatsmodelsPoissonGLM
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                model_class = StatsmodelsGammaGLM
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                model_class = StatsmodelsTweedieGLM
            case _:
                raise ValueError(f"Unsupported model enum: {self._model_enum}")

        warnings.filterwarnings("ignore", category=DomainWarning)

        model = model_class(
            config=self.config,
            target_col=self.target_col,
            pred_col=self.pred_col,
            hparams={"fit_intercept": False, "link": self._link},
            call_updated_hparams=True
        )

        for i in range(len(sorted_idx)):
            center_value = pure_predictions_df[self.pure_pred_col].iloc[i]
            distances = np.abs(pure_predictions_df[self.pure_pred_col].values - center_value)
            idx = np.argsort(distances)[:self._k]
            left, right = idx.min(), idx.max() + 1

            local_target_df = target_df.iloc[left:right, :]
            local_sample_weight = sample_weight.iloc[left:right]
            local_distances = distances[left:right]
            kernel_weights = compute_kernel_weights(local_distances, self._kernel_enum)
            local_sample_weight = local_sample_weight * kernel_weights
            # GLM
            local_features_df = pd.DataFrame({"intercept": np.ones(len(local_target_df))}, index=local_target_df.index)
            model.fit(local_features_df, local_target_df, sample_weight=local_sample_weight)
            y_pred = model.predict(local_features_df[:1]).iloc[0,0]
            # Simple mean for tests
            # y_pred = np.average(local_target_df[self.target_col], weights=local_sample_weight)
            self._y_pred.append(y_pred)
            self._y_pure_pred.append(center_value)

        self._y_pred = np.array(self._y_pred)
        self._y_pure_pred = np.array(self._y_pure_pred)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> pd.Series:
        y_pred = pure_predictions_df[self.pure_pred_col].values
        y_pred = np.clip(y_pred, self._min_pure_y, self._max_pure_y)
        pred = []
        for yp in y_pred:
            dist = np.abs(self._y_pure_pred - yp)
            idx = np.argsort(dist)[0]
            pred.append(self._y_pred[idx])
        return pd.Series(pred, index=pure_predictions_df.index)

    def metric(self) -> Metric:
        return get_statsmodels_metric(self.config, self._model_enum, self.pred_col)


    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "frac": 0.2,
        }

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "frac": hp.uniform("frac", 0.01, 0.2),
        }

    def _updated_hparams(self):
        self._model_enum = ModelEnum(self._hparams.get("model"))
        self._kernel_enum = KernelEnum(self._hparams.get("kernel"))
        self._link = self._hparams.get("link")
        self._frac = float(self._hparams.get("frac"))
        self._k = None
        self._y_pred = None
        self._y_pure_pred = None
        self._min_pure_y = None
        self._max_pure_y = None

    def summary(self):
        return f"LocalStatsmodelsGLMCalibration: frac={self._frac}, kernel={self._kernel_enum.value}, model: {self._model_enum.value}, link: {self._link}"

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)
