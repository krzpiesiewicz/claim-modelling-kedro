from abc import ABC
from typing import Dict, Any, Collection, Union, Optional, List, Tuple

import numpy as np
import pandas as pd
from hyperopt import hp
from lightgbm import LGBMRegressor

from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.utils.metrics.metric import MeanPoissonDeviance, MeanGammaDeviance, \
    MeanTweedieDeviance
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys


class LightGBMRegressorABC(PredictiveModel):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        PredictiveModel.__init__(self, config=config, target_col=target_col, pred_col=pred_col, **kwargs)

    def _updated_hparams(self):
        self._random_state = int(self._hparams.get("random_state", 0))
        self._val_size = float(self._hparams.get("val_size", 0.25))
        # Cast hyperparameters to appropriate types
        for param in ["n_estimators", "max_depth", "min_child_samples"]:
            if param in self._hparams:
                self._hparams[param] = int(self._hparams[param])
        self.model = None

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "learning_rate",
            "n_estimators",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "random_state",
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(0.2)),
            "n_estimators": hp.quniform("n_estimators", 50, 1000, 10),
            "max_depth": hp.quniform("max_depth", 2, 10, 1),
            "min_child_samples": hp.quniform("min_child_samples", 10, 100, 5),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-6), np.log(10.0)),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-6), np.log(10.0)),
        }

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            # Default hyperparameters for LightGBM
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 0,
            "val_size": 0.25,
            "force_col_wise": True,
            "early_stopping_rounds": 50,
            "objective": None,  # to be set in child classes
            "eval_metric": None,  # to be set in child classes
        }

    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame,
             sample_train_keys: Optional[pd.Index] = None, sample_val_keys: Optional[pd.Index] = None, **kwargs) -> None:
        # Ensure target_df is DataFrame for stratified split
        index = features_df.index.intersection(target_df.index)
        if sample_train_keys is not None:
            index = index.intersection(sample_train_keys.union(sample_val_keys if sample_val_keys is not None else pd.Index([])))
            sample_train_keys = sample_train_keys.intersection(index)
        if sample_train_keys is None:
            sample_train_keys = index.difference(sample_val_keys or pd.Index([]))
        fit_kwargs = kwargs.copy()
        if sample_val_keys is None:
            split_kwargs = {}
            sample_weight = kwargs.get("sample_weight", None)
            split_kwargs["sample_weight"] = sample_weight
            sample_train_keys, sample_val_keys = get_stratified_train_test_split_keys(
                target_df.loc[index,:],
                stratify_target_col=self.target_col,
                test_size=self._val_size,
                shuffle=True,
                random_seed=self._random_state,
                verbose=False,
                **split_kwargs
            )
        X_train = features_df.loc[sample_train_keys,:]
        y_train = target_df.loc[sample_train_keys,:]
        X_val = features_df.loc[sample_val_keys,:]
        y_val = target_df.loc[sample_val_keys,:]
        y_train = y_train[self.target_col]
        y_val = y_val[self.target_col]
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        if "sample_weight" in fit_kwargs:
            fit_kwargs["eval_sample_weight"] = [fit_kwargs["sample_weight"].loc[sample_val_keys]]
            fit_kwargs["sample_weight"] = fit_kwargs["sample_weight"].loc[sample_train_keys]
        self.model = LGBMRegressor(**self.get_hparams())
        self.model.fit(X_train, y_train, **fit_kwargs)

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        return self.model.predict(features_df)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        return dict(config=self.config)
        # return dict(config=copy.deepcopy(self.config))

    def summary(self) -> str:
        return f"model: {self.model} with hyper parameters: {self.get_hparams()}"


class LightGBMPoissonRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col, **kwargs)

    def metric(self):
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        defaults = super().get_default_hparams()
        defaults.update({
            "eval_metric": "poisson",
            "objective": "poisson",
        })
        return defaults


class LightGBMGammaRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col, **kwargs)

    def metric(self):
        return MeanGammaDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        defaults = super().get_default_hparams()
        defaults.update({
            "eval_metric": "gamma",
            "objective": "gamma",
        })
        return defaults


class LightGBMTweedieRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col,  **kwargs)

    def metric(self):
        power = self._hparams.get("tweedie_variance_power", None)
        if power is not None:
            return MeanTweedieDeviance(self.config, pred_col=self.pred_col, power=power)
        from claim_modelling_kedro.pipelines.utils.metrics.gini import ConcentrationCurveGiniIndex
        return ConcentrationCurveGiniIndex(self.config)

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        space = super().get_hparams_space()
        space.update({
            "tweedie_variance_power": hp.uniform("tweedie_variance_power", 1.1, 1.9),
        })
        return space

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        defaults = super().get_default_hparams()
        defaults.update({
            "tweedie_variance_power": 1.5,
            "eval_metric": "tweedie",
            "objective": "tweedie",
        })
        return defaults
