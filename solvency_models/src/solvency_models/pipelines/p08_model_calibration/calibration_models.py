import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression
from sklearn.linear_model import PoissonRegressor

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.utils.utils import assert_pandas_no_lacking_indexes, trunc_target_index, \
    preds_as_dataframe_with_col_name


logger = logging.getLogger(__name__)


class CalibrationModel(ABC):
    def __init__(self, config: Config, fit_kwargs: Dict[str, Any] = None):
        self.config = config
        self.target_col = config.mdl_task.target_col
        self.pure_pred_col = config.clb.pure_prediction_col
        self.calib_pred_col = config.clb.calibrated_prediction_col
        self._fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self._hparams = None
        self._features_importances = None

    def fit(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray],
            target_df: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs):
        assert_pandas_no_lacking_indexes(pure_predictions_df, target_df, "pure_predictions_df", "target_df")
        target_df = trunc_target_index(pure_predictions_df, target_df)
        self._fit(pure_predictions_df, target_df, **self._fit_kwargs, **kwargs)

    def predict(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        preds = self._predict(pure_predictions_df)
        preds = preds_as_dataframe_with_col_name(pure_predictions_df, preds, self.calib_pred_col)
        return preds

    def transform(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return self.predict(pure_predictions_df)

    @abstractmethod
    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass


class CIRModelCenteredIsotonicRegression(CalibrationModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = CenteredIsotonicRegression(increasing=True, out_of_bounds="clip")

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = pure_predictions_df[self.pure_pred_col]
        y = target_df[self.target_col]
        logger.debug(f"_fit- x: {x}\ny: {y}")
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = pure_predictions_df[self.pure_pred_col]
        logger.debug(f"_predict - x: {x}")
        return self.model.transform(x)


class SklearnIsotonicRegression(CalibrationModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = pure_predictions_df[self.pure_pred_col]
        y = target_df[self.target_col]
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = pure_predictions_df[self.pure_pred_col]
        return self.model.transform(x)


class SklearnPoissonGLM(CalibrationModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PoissonRegressor(fit_intercept=True, alpha=0)

    def create_features(self, pure_predictions_df: pd.DataFrame) -> pd.DataFrame:
        features_df = pure_predictions_df.loc[:, [self.pure_pred_col]].copy()
        # features_df["log(1+x)"] = np.log1p(features_df[self.pure_pred_col])
        return features_df

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = self.create_features(pure_predictions_df)
        y = target_df[self.target_col]
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = self.create_features(pure_predictions_df)
        return self.model.predict(x)