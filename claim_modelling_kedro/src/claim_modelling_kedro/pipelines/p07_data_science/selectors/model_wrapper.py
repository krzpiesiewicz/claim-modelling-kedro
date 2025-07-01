from typing import Optional

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.select import SelectorModel
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path

import logging

logger = logging.getLogger(__name__)


class ModelWrapperFeatureSelector(SelectorModel):
    def __init__(self, config: Config, model_class = None):
        super().__init__(config=config)
        if model_class is None:
            model_class = config.ds.model_class
        if isinstance(model_class, str):
            model_class = get_class_from_path(config.ds.model_class)
        self._model = model_class(config=self.config, target_col=self.config.mdl_task.target_col,
                                  pred_col=self.config.mdl_task.prediction_col,
                                  hparams=self.config.ds.fs.params)

    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame,
            sample_train_keys: Optional[pd.Index] = None, sample_val_keys: Optional[pd.Index] = None, **kwargs) -> pd.DataFrame:
        logger.debug(f"{self.config.ds.fs.params=}")
        self._model.update_hparams(self.config.ds.fs.params)
        self._model.fit(features_df, target_df,
                        sample_train_keys=sample_train_keys, sample_val_keys=sample_val_keys, **kwargs)
        self._set_features_importance(self._model.get_features_importance())
        self._set_selected_features_from_importance()
