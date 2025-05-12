import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.select import SelectorModel
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path


class ModelWrapperFeatureSelector(SelectorModel):
    def __init__(self, config: Config, model_class = None):
        super().__init__(config=config)
        if model_class is None:
            model_class = config.ds.model_class
        if isinstance(model_class, str):
            model_class = get_class_from_path(config.ds.model_class)
        self._model = model_class(config=self.config, target_col=self.config.mdl_task.target_col,
                                  pred_col=self.config.mdl_task.prediction_col)

    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        self._model.update_hparams(self.config.ds.fs_params)
        self._model.fit(features_df, target_df, **kwargs)
        self._set_features_importances(self._model.get_features_importances())
        self._set_selected_features_from_importances()
