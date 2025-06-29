from abc import ABC
from typing import Dict, Any, Collection, Union, Optional, List, Tuple

import numpy as np
import pandas as pd
from hyperopt import hp
from lightgbm import LGBMRegressor

from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.utils.metrics.metric import MeanPoissonDeviance, MeanGammaDeviance, \
    MeanTweedieDeviance
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys


class LightGBMRegressorABC(SklearnModel, ABC):
    def __init__(self, config, target_col: str, pred_col: str, model_class, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col, model_class=model_class, **kwargs)

    def _updated_hparams(self):
        self._random_state = int(self._hparams.get("random_state", 0))
        self._val_size = float(self._hparams.get("val_size", 0.25))
        # Cast hyperparameters to appropriate types
        for param in ["n_estimators", "max_depth", "min_child_samples"]:
            if param in self._hparams:
                self._hparams[param] = int(self._hparams[param])
        SklearnModel._updated_hparams(self)

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

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], eval_set: Optional[List[Tuple]] = None, eval_names: Optional[List[str]] = None, **kwargs) -> None:
        # Ensure target is a Series if DataFrame
        if type(target_df) is not None and hasattr(target_df, "columns") and self.target_col in target_df:
            target_df = target_df[self.target_col]
        # Ensure target_df is DataFrame for stratified split
        if eval_set is None and isinstance(target_df, pd.Series):
            target_df = target_df.to_frame(name=self.target_col)
        fit_kwargs = kwargs.copy()
        sample_weight = fit_kwargs.pop("sample_weight", None)
        if eval_set is None:
            split_kwargs = {}
            if sample_weight is not None:
                split_kwargs["sample_weight"] = sample_weight
            train_keys, val_keys = get_stratified_train_test_split_keys(
                target_df,
                stratify_target_col=self.target_col,
                test_size=self._val_size,
                shuffle=True,
                random_seed=self._random_state,
                verbose=False,
                **split_kwargs
            )
            X_train = features_df.loc[train_keys,:] if isinstance(features_df, pd.DataFrame) else features_df[train_keys]
            y_train = target_df.loc[train_keys,:] if isinstance(target_df, pd.DataFrame) else target_df[train_keys]
            X_val = features_df.loc[val_keys,:] if isinstance(features_df, pd.DataFrame) else features_df[val_keys]
            y_val = target_df.loc[val_keys,:] if isinstance(target_df, pd.DataFrame) else target_df[val_keys]
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            features_df = X_train
            target_df = y_train
        else:
            fit_kwargs["eval_set"] = eval_set
        if eval_names is not None:
            fit_kwargs["eval_names"] = eval_names
        self.model.fit(features_df, target_df, **fit_kwargs)


class LightGBMPoissonRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        if "hparams" not in kwargs:
            kwargs["hparams"] = {}
        kwargs["hparams"]["eval_metric"] = "poisson"
        super().__init__(config=config, target_col=target_col, pred_col=pred_col,
                         model_class=LGBMRegressor, objective="poisson", **kwargs)

    def metric(self):
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)


class LightGBMGammaRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col,
                         model_class=LGBMRegressor, **kwargs)

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
        if "hparams" not in kwargs:
            kwargs["hparams"] = {}
        kwargs["hparams"]["eval_metric"] = "tweedie"
        super().__init__(config=config, target_col=target_col, pred_col=pred_col,
                         model_class=LGBMRegressor, objective="tweedie", **kwargs)

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
        })
        return defaults
