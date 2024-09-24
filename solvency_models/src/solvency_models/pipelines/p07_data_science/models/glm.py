from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from hyperopt import hp
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p01_init.mdl_info_config import ModelEnum
from solvency_models.pipelines.p07_data_science.model import PredictiveModel
from solvency_models.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from solvency_models.pipelines.utils.metrics import Metric, RootMeanSquaredError, MeanPoissonDeviance, \
    MeanGammaDeviance, NormalizedGiniCoefficient, MeanTweedieDeviance


class StatsmodelsGLM(PredictiveModel):
    """
    It"s important to note that sm.GLM does not have hyperparameters in
    the same sense as machine learning models like Random Forest or
    Gradient Boosting do. Parameters like learning rate, tree depth, etc.,
    can be tuned in those models. The parameters of sm.GLM are more about
    specifying the structure of the model and the data, rather than
    tuning the model"s learning process.
    """

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        match config.mdl_info.model:
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                self.family = sm.families.Gaussian()
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                self.family = sm.families.Gamma()
            case ModelEnum.STATSMODELS_POISSON_GLM:
                self.family = sm.families.Poisson()
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                self.family = sm.families.Tweedie()
            case _:
                raise ValueError(
                    f"""Family for model {config.mdl_info.model} not supported in Stastsmodels GLM. Supported families are:
    - \"Gaussian\" for {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
    - \"Gamma\" for {ModelEnum.STATSMODELS_GAMMA_GLM},
    - \"Poisson\" for {ModelEnum.STATSMODELS_POISSON_GLM},
    - \"Tweedie\" fpr {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")
        self.update_hparams({})
        self.model = None

    def metric(self) -> Metric:
        match self.config.mdl_info.model:
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                return RootMeanSquaredError(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                return MeanGammaDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_POISSON_GLM:
                return MeanPoissonDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                return NormalizedGiniCoefficient(self.config, pred_col=self.pred_col)
            case _:
                raise ValueError(
                    f"""Family for model {self.config.mdl_info.model} not supported in Stastsmodels GLM. Supported are:
            - {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
            - {ModelEnum.STATSMODELS_GAMMA_GLM},
            - {ModelEnum.STATSMODELS_POISSON_GLM},
            - {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")

    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        if type(target_df) is pd.DataFrame:
            target_df = target_df[self.target_col]
        self.model = sm.GLM(target_df, features_df, family=self.family).fit(**kwargs)
        self._set_features_importances(np.abs(self.model.params))

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        return self.model.predict(features_df)

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        return dict(config=self.config)
        # return dict(config=copy.deepcopy(self.config))

    def _updated_hparams(self):
        pass

    def summary(self):
        return self.model.summary()


class SklearnPoissonGLM(SklearnModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=PoissonRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", cls._get_solver_options())
        }

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        hparams = {}
        hparams["alpha"] = float(hopt_hparams["alpha"])
        hparams["fit_intercept"] = bool(hopt_hparams["fit_intercept"])
        hparams["solver"] = cls._get_solver_options()[int(hopt_hparams["solver"])]
        return hparams

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0
        }


class SklearnGammaGLM(SklearnModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=GammaRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanGammaDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        hparams = {}
        hparams["alpha"] = float(hopt_hparams["alpha"])
        hparams["fit_intercept"] = bool(hopt_hparams["fit_intercept"])
        hparams["solver"] = cls._get_solver_options()[int(hopt_hparams["solver"])]
        return hparams

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0
        }


class SklearnTweedieGLM(SklearnModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=TweedieRegressor, **kwargs)

    def metric(self) -> Metric:
        hparams = self.get_hparams()
        if "power" in hparams:
            return MeanTweedieDeviance(self.config, pred_col=self.pred_col, power=hparams["power"])
        return NormalizedGiniCoefficient(self.config)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "power": hp.uniform("power", 1.0, 2.0),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        hparams = {}
        hparams["alpha"] = float(hopt_hparams["alpha"])
        hparams["fit_intercept"] = bool(hopt_hparams["fit_intercept"])
        hparams["power"] = float(hopt_hparams["power"])
        hparams["solver"] = cls._get_solver_options()[int(hopt_hparams["solver"])]
        return hparams

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "power": 1.5,
            "alpha": 1.0,
            "fit_intercept": True,
            "link": "auto",
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0
        }
