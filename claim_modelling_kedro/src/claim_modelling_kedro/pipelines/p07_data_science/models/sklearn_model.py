import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Collection

import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel

logger = logging.getLogger(__name__)


class SklearnModel(PredictiveModel, ABC):
    def __init__(self, config: Config, target_col: str, pred_col: str, model_class, hparams: Dict[str, Any] = None,
                 fit_kwargs: Dict[str, Any] = None, **kwargs):
        logger.debug(f"SklearnModel.__init__ with model_class: {model_class}")
        PredictiveModel.__init__(self, config=config, target_col=target_col, pred_col=pred_col, fit_kwargs=fit_kwargs,
                                 call_updated_hparams=False)
        self.model_class = model_class
        if kwargs is not None and len(kwargs) > 0:
            hparams = hparams.copy() if hparams is not None else {}
            hparams.update(kwargs)
        self.update_hparams(hparams)
        logger.debug(f"hparams: {self._hparams}")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        if type(target_df) is pd.DataFrame:
            target_df = target_df[self.target_col]
        self.model.fit(features_df, target_df, **kwargs)

    def _predict(self, features_df: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        return self.model.predict(features_df)

    @classmethod
    @abstractmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        pass

    def _updated_hparams(self):
        sklearn_hparams_names = self.__class__._sklearn_hparams_names()
        sklearn_params = {key: val for key, val in self.get_hparams().items() if key in sklearn_hparams_names}
        logger.debug(f"{sklearn_params=}")
        self.model = self.model_class(**sklearn_params)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = self.model.get_params(deep)
        logger.debug(f"self.model.get_params(deep={deep}): {params}")
        if deep:
            params["config"] = copy.deepcopy(self.config)
        else:
            params["config"] = self.config
        params["target_col"] = self.target_col
        params["pred_col"] = self.pred_col
        params["fit_kwargs"] = self._fit_kwargs
        return params

    def summary(self) -> str:
        # sklearn models do not have a summary method like statsmodels
        # You can print the model itself as a form of summary
        return f"model: {self.model} with hyper parameters: {self._hparams}"
