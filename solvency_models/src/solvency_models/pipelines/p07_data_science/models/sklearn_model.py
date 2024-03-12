import logging
from abc import ABC
from typing import Dict, Any

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p07_data_science.model import PredictiveModel

logger = logging.getLogger(__name__)


class SklearnModel(PredictiveModel, ABC):
    def __init__(self, config: Config, model_class, model_hparams: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.model_hparams = model_hparams.copy() if model_hparams is not None else {}
        self.model_const_hparams = config.ds.model_const_hparams
        self.model_hparams.update(self.model_const_hparams)
        logger.debug(f"const_hparams: {self.model_hparams}")
        logger.debug(f"hparams: {self.model_hparams}")
        self.model = model_class(**self.model_hparams)

    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        self.model.fit(features_df, target_df[self.target_col])

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        return self.model.predict(features_df)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.model.get_params(deep)

    def summary(self) -> str:
        # sklearn models do not have a summary method like statsmodels
        # You can print the model itself as a form of summary
        return f"model: {self.model} with hyper parameters: {self.model_hparams}"
