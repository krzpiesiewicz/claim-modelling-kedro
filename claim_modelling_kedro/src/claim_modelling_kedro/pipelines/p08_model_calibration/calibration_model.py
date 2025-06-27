from abc import ABC, abstractmethod
from typing import Union
import logging
import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.clb_config import RebalanceInTotalsMethod
from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel, get_sample_weight
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
        if not "sample_weight" in kwargs:
            sample_weight = get_sample_weight(self.config, target_df)
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
        if "sample_weight" in kwargs:
            kwargs["sample_weight"] = kwargs["sample_weight"].loc[target_df.index]
        super().fit(pure_predictions_df, target_df, **kwargs)
        logger.debug(f"{self.config.clb.post_calibration_rebalancing_enabled=}")
        if self.config.clb.post_calibration_rebalancing_enabled:
            preds = super().predict(pure_predictions_df)
            logger.debug(f"{self.config.clb.post_clb_rebalance_method=}")
            match self.config.clb.post_clb_rebalance_method:
                case RebalanceInTotalsMethod.SCALE:
                    sample_weight = kwargs["sample_weight"]
                    if sample_weight is None:
                        logger.debug("sample_weight is not provided, using unweighted averages")
                        weighted_avg_target = np.average(target_df[self.target_col])
                        weighted_avg_pred = np.average(preds[self.pred_col])
                    else:
                        logger.debug("sample_weight is provided, using it for weighted averages")
                        weighted_avg_target = np.average(target_df[self.target_col], weights=sample_weight)
                        weighted_avg_pred = np.average(preds[self.pred_col], weights=sample_weight)
                    self._post_clb_scale_factor = weighted_avg_target / weighted_avg_pred
                    logger.debug(f"Post calibration scale factor: {self._post_clb_scale_factor}")

    def predict(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        logger.debug("CalibrationModel predict called")
        preds = super().predict(pure_predictions_df)
        logger.debug(f"{self.config.clb.post_calibration_rebalancing_enabled=}")
        if self.config.clb.post_calibration_rebalancing_enabled:
            logger.debug(f"{self.config.clb.post_clb_rebalance_method=}")
            match self.config.clb.post_clb_rebalance_method:
                case RebalanceInTotalsMethod.SCALE:
                    logger.debug(f"Applying post calibration scale factor: {self._post_clb_scale_factor}")
                    preds[self.calib_pred_col] = preds[self.calib_pred_col] * self._post_clb_scale_factor
        return preds

    def transform(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return self.predict(pure_predictions_df)

    @abstractmethod
    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass
