from abc import ABC, abstractmethod
from typing import Union
import logging
import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.utils.dataframes import assert_pandas_no_lacking_indexes, trunc_target_index


logger = logging.getLogger(__name__)


class CalibrationModel(PredictiveModel, ABC):
    def __init__(self, config: Config, **kwargs):
        logger.debug(f"CalibrationModel.__init__")
        PredictiveModel.__init__(self, config, target_col=config.mdl_task.target_col,
                         pred_col=config.clb.calibrated_prediction_col, **kwargs)
        self.pure_pred_col = config.clb.pure_prediction_col
        self.calib_pred_col = config.clb.calibrated_prediction_col

    def fit(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray],
            target_df: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs):
        logger.debug("CalibrationModel fit called")
        assert_pandas_no_lacking_indexes(pure_predictions_df, target_df, "pure_predictions_df", "target_df")
        target_df = trunc_target_index(pure_predictions_df, target_df)
        super().fit(pure_predictions_df, target_df, **kwargs)

    def predict(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        logger.debug("CalibrationModel predict called")
        preds = super().predict(pure_predictions_df)
        return preds

    def transform(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return self.predict(pure_predictions_df)

    @abstractmethod
    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass
