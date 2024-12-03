import copy
import logging
from abc import ABC
from typing import Dict, Any, Union

import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel

logger = logging.getLogger(__name__)


class SklearnModel(PredictiveModel, ABC):
    def __init__(self, config: Config, model_class, hparams: Dict[str, Any] = None,
                 fit_kwargs: Dict[str, Any] = None, **kwargs):
        super().__init__(config=config, fit_kwargs=fit_kwargs)
        self.model_class = model_class
        model_hparams = self.get_default_hparams().copy()
        if hparams is not None:
            model_hparams.update(hparams)
        for key, value in kwargs.items():
            if key in model_hparams:
                model_hparams[key] = value
            else:
                logger.warning(f"Unknown hyperparameter {key} for model {self.model_class}")
        self.model = self.model_class(**model_hparams)
        self.update_hparams(model_hparams)
        logger.debug(f"hparams: {self._hparams}")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        if type(target_df) is pd.DataFrame:
            target_df = target_df[self.target_col]
        self.model.fit(features_df, target_df, **kwargs)

    def _predict(self, features_df: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        return self.model.predict(features_df)

    def _updated_hparams(self):
        self.model = self.model_class(**self._hparams)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = self.model.get_params(deep)
        if deep:
            params["config"] = copy.deepcopy(self.config)
        else:
            params["config"] = self.config
        return params

    def summary(self) -> str:
        # sklearn models do not have a summary method like statsmodels
        # You can print the model itself as a form of summary
        return f"model: {self.model} with hyper parameters: {self._hparams}"
