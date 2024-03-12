from typing import Dict, Any

from sklearn.feature_selection import RFE

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p07_data_science.select import SelectorModel
from solvency_models.pipelines.utils.utils import get_class_from_path


class RecursiveFeatureEliminationSelector(SelectorModel):
    def __init__(self, config: Config):
        super().__init__(config=config)
        if "rfe" in config.ds.fs_params and "estimator_kwargs" in config.ds.fs_params["rfe"]:
            self.estimator_kwargs = config.ds.fs_params["rfe"]["estimator_kwargs"]
        if self.estimator_kwargs is None:
            self.estimator_kwargs = {}
        self.estimator = get_class_from_path(config.ds.model_class)(config=config, **self.estimator_kwargs)
        self.selector = None

    def fit(self, features_df, target_df):
        X = features_df
        y = target_df[self.target_col]
        self.selector = RFE(self.estimator, n_features_to_select=self.max_n_features,
                            importance_getter=(lambda estimator: estimator.get_features_importances()))
        self.selector.fit(X, y)
        self._set_selected_features(self.selector.feature_names_in_[self.selector.support_])
        importances = self.selector.estimator_.get_features_importances().copy()
        importances.index = self.selector.feature_names_in_[importances.index]
        self._set_features_importances(importances)
