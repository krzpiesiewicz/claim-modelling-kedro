from abc import ABC
from typing import Union, Dict, Any, List
import logging
import numpy as np
import pandas as pd
from cir_model import CenteredIsotonicRegression
from sklearn.isotonic import IsotonicRegression

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric, RootMeanSquaredError


logger = logging.getLogger(__name__)


class IsotonicLikeCalibrationModel(CalibrationModel, ABC):
    def __init__(self, config: Config, **kwargs):
        logger.debug(f"IsotonicLikeCalibrationModel.__init__")
        CalibrationModel.__init__(self, config, **kwargs)
        self._y_min = None

    def _set_y_min(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        combined_df = pd.concat([pure_predictions_df[[self.pure_pred_col]], target_df[[self.target_col]]], axis=1)
        sorted_df = combined_df.sort_values(by=self.pure_pred_col, ascending=True)
        logger.debug(f"{sorted_df=}")
        # Find the first non-zero position and replace the values before it (including that position)
        # with the mean of the target values up to that position so that the balance is preserved
        first_nonzero_position = sorted_df[sorted_df[self.target_col] > 0].index[0]
        logger.debug(f"{first_nonzero_position=}")
        pos = first_nonzero_position
        smallest_df = sorted_df.loc[:pos, self.target_col]
        logger.debug(f"{smallest_df=}")
        self._y_min = np.mean(smallest_df)
        logger.debug(f"{self._y_min=}")
        smallest_idx = smallest_df.index
        logger.debug(f"{smallest_idx=}")
        adjusted_target_df = target_df.copy()
        adjusted_target_df.loc[smallest_idx, self.target_col] = self._y_min
        return adjusted_target_df

    def get_y_min(self) -> float:
        return self._y_min

    def metric(self) -> Metric:
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)


class SklearnLikeIsotonicRegression(IsotonicLikeCalibrationModel, SklearnModel):
    def __init__(self, config: Config, model_class, **kwargs):
        IsotonicLikeCalibrationModel.__init__(self, config, call_updated_hparams=False, **kwargs)
        SklearnModel.__init__(self, config=config, model_class=model_class,
                              pred_col=self.pred_col, target_col=self.target_col, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        logger.debug(f"CSklearnLikeIsotonicRegression._fit called with kwargs: {kwargs}")
        if self.force_positive:
            target_df = self._set_y_min(pure_predictions_df, target_df)
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        SklearnModel._fit(self, pure_predictions_df, target_df, **kwargs)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        pred = SklearnModel._predict(self, pure_predictions_df)
        return pred

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self.force_positive = self.get_hparams().get("force_positive")
        self._y_min = None
        SklearnModel._updated_hparams(self)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "increasing": True,
            "out_of_bounds": "clip",
            "force_positive": True
        }

    @classmethod
    def _sklearn_hparams_names(cls) -> List[str]:
        return ["increasing", "out_of_bounds"]


class CIRModelCenteredIsotonicRegression(SklearnLikeIsotonicRegression):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=CenteredIsotonicRegression, **kwargs)


class SklearnIsotonicRegression(SklearnLikeIsotonicRegression):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=IsotonicRegression, **kwargs)
