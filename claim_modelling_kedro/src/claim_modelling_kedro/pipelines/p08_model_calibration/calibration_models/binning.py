from typing import Dict, Any

import logging
import numpy as np
import pandas as pd
from hyperopt import hp

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import get_sample_weight
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index
from claim_modelling_kedro.pipelines.utils.metrics import RootMeanSquaredError


logger = logging.getLogger(__name__)


class EqualBinsMeansCalibration(CalibrationModel):
    def __init__(self, config: Config, n_bins: int = 10, **kwargs):
        super().__init__(config, **kwargs)
        self._n_bins = n_bins
        self._bin_bounds = None
        self._bin_means = None

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        logger.debug("EqualBinsMeansCalibration _fit called")
        sample_weight = get_sample_weight(config=self.config, target_df=target_df)
        y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(
            target_df[self.target_col],
            pure_predictions_df[self.pure_pred_col],
            sample_weight
        )

        # Use qcut with duplicates='drop' to handle repeated prediction values
        try:
            _, self._bin_bounds = pd.qcut(
                y_pred,
                q=self._n_bins,
                labels=False,
                retbins=True,
                duplicates='drop'
            )
            self._n_bins = len(self._bin_bounds) - 1
        except ValueError as e:
            logger.warning(f"pd.qcut failed: {e}. Falling back to uniform binning.")
            self._bin_bounds = np.linspace(np.min(y_pred), np.max(y_pred), self._n_bins + 1)
        bin_indices = np.digitize(y_pred, self._bin_bounds[1:-1], right=False)

        # Compute means per bin
        self._bin_means = {}
        for bin_idx in range(self._n_bins):
            bin_mask = bin_indices == bin_idx
            if bin_mask.any():
                self._bin_means[bin_idx] = np.average(y_true[bin_mask], weights=sample_weight[bin_mask])

        # Global mean fallback
        self._global_mean = np.average(y_true, weights=sample_weight)

        logger.debug(f"Fitted {self._n_bins} bins: bounds={self._bin_bounds}, means={self._bin_means}")

    def _predict(self, pure_predictions_df: pd.DataFrame) -> pd.Series:
        logger.debug("EqualBinsMeansCalibration _predict called")
        y_pure_pred = pure_predictions_df[self.pure_pred_col].values
        bin_indices = np.digitize(y_pure_pred, self._bin_bounds[1:-1], right=False)
        calibrated = [self._bin_means.get(idx, self._global_mean) for idx in bin_indices]
        return pd.Series(calibrated, index=pure_predictions_df.index)

    def metric(self):
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {"n_bins": 10}

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {"n_bins": hp.quniform("n_bins", 2, 100, 1)}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)

    def summary(self):
        return f"EqualBinsMeansCalibration with {self._n_bins} bins.\nBounds: {self._bin_bounds}\nMeans: {self._bin_means}"

    def _updated_hparams(self):
        if self._hparams is not None:
            self._n_bins = int(self._hparams.get("n_bins"))
