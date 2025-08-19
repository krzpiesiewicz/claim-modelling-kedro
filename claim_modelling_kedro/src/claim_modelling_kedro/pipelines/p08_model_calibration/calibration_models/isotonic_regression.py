from abc import ABC
from typing import Union, Dict, Any, List
import logging
import numpy as np
import pandas as pd
from cir_model import CenteredIsotonicRegression
from sklearn.isotonic import IsotonicRegression

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric, RootMeanSquaredError


logger = logging.getLogger(__name__)


class IsotonicLikeCalibrationModel(CalibrationModel, ABC):
    def __init__(self, config: Config, **kwargs):
        logger.debug("IsotonicLikeCalibrationModel.__init__")
        CalibrationModel.__init__(self, config, **kwargs)

    def _clip_lowest_and_highest_bins(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        combined_df = pd.concat(
            [pure_predictions_df[[self.pure_pred_col]], target_df[[self.target_col]]], axis=1
        )
        sorted_df = combined_df.sort_values(by=self.pure_pred_col, ascending=True)
        n_obs = len(sorted_df)

        adjusted_target_df = target_df.copy()

        # Clip lower bin
        n_low = int(n_obs * self._clip_low_bin)

        lowest_idx = sorted_df.index[:n_low] if n_low > 0 else None
        sample_weight = get_sample_weight(self.config, target_df)
        if self._force_positive:
            if n_low > 0:
                if sorted_df.loc[lowest_idx, self.target_col].max() <= 0:
                    logger.info("All values in the lowest bin are non-positive. The first position of positive value will be found.")
                    find_first_positive = True
                else:
                    find_first_positive = False
            else:
                find_first_positive = True
            if find_first_positive:
                first_positive_position = sorted_df[sorted_df[self.target_col] > 0].index[0]
                lowest_idx = sorted_df.index[:first_positive_position]
        if lowest_idx is not None:
            self._y_min = np.average(sorted_df.loc[lowest_idx, self.target_col], weights=sample_weight.loc[lowest_idx])
            adjusted_target_df.loc[lowest_idx, self.target_col] = self._y_min
            logger.debug(f"Clipped lower bin: {n_low} obs → mean={self._y_min}")

        # Clip upper bin
        n_high = int(n_obs * self._clip_high_bin)
        if n_high > 0:
            highest_idx = sorted_df.index[-n_high:]
            self._y_max = np.average(sorted_df.loc[highest_idx, self.target_col], weights=sample_weight.loc[lowest_idx])
            adjusted_target_df.loc[highest_idx, self.target_col] = self._y_max
            logger.debug(f"Clipped upper bin: {n_high} obs → mean={self._y_max}")

        return adjusted_target_df

    def get_y_min(self) -> float:
        return self._y_min

    def get_y_max(self) -> float:
        return self._y_max

    def metric(self) -> Metric:
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)


class SklearnLikeIsotonicRegression(IsotonicLikeCalibrationModel, SklearnModel):
    def __init__(self, config: Config, model_class, **kwargs):
        IsotonicLikeCalibrationModel.__init__(self, config, call_updated_hparams=False, **kwargs)
        SklearnModel.__init__(self, config=config, model_class=model_class,
                              pred_col=self.pred_col, target_col=self.target_col, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame,
             sample_train_keys: pd.Index = None, sample_val_keys: pd.Index = None, **kwargs) -> None:
        logger.debug("SklearnLikeIsotonicRegression._fit called")
        if sample_train_keys is not None:
            pure_predictions_df = pure_predictions_df.loc[sample_train_keys, :]
            target_df = target_df.loc[sample_train_keys, :]
        target_df = self._clip_lowest_and_highest_bins(pure_predictions_df, target_df)
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        SklearnModel._fit(self, pure_predictions_df, target_df, **kwargs)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        return SklearnModel._predict(self, pure_predictions_df)

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self._force_positive = self.get_hparams().get("force_positive")
        self._clip_low_bin = float(self.get_hparams().get("clip_low_bin", 0.0))
        self._clip_high_bin = float(self.get_hparams().get("clip_high_bin", 0.0))

        if not (0.0 <= self._clip_low_bin <= 1.0) or not (0.0 <= self._clip_high_bin <= 1.0):
            raise ValueError("clip_low_bin and clip_high_bin must be between 0.0 and 1.0")

        if self._clip_low_bin + self._clip_high_bin > 1.0:
            raise ValueError("The sum of clip_low_bin and clip_high_bin must not exceed 1.0")

        self._y_min = None
        self._y_max = None
        SklearnModel._updated_hparams(self)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "increasing": True,
            "out_of_bounds": "clip",
            "force_positive": True,
            "clip_low_bin": 0.0,
            "clip_high_bin": 0.0,
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
