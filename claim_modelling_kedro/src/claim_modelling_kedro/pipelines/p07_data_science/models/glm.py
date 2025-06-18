import logging
from abc import ABC
from typing import Dict, Any, List, Union, Collection

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyglmnet
from hyperopt import hp
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.ds_config import ModelEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric, RootMeanSquaredError, MeanPoissonDeviance, \
    MeanGammaDeviance, MeanTweedieDeviance
from claim_modelling_kedro.pipelines.utils.metrics.gini import ConcentrationCurveGiniIndex

logger = logging.getLogger(__name__)


def get_statsmodels_metric(config: Config, model_enum: ModelEnum, pred_col: str) -> Metric:
    match model_enum:
        case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
            return RootMeanSquaredError(config, pred_col=pred_col)
        case ModelEnum.STATSMODELS_GAMMA_GLM:
            return MeanGammaDeviance(config, pred_col=pred_col)
        case ModelEnum.STATSMODELS_POISSON_GLM:
            return MeanPoissonDeviance(config, pred_col=pred_col)
        case ModelEnum.STATSMODELS_TWEEDIE_GLM:
            return ConcentrationCurveGiniIndex(config, pred_col=pred_col)
        case _:
            raise ValueError(
                f"""Family for model {model_enum} not supported in Stastsmodels GLM. Supported are:
        - {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
        - {ModelEnum.STATSMODELS_GAMMA_GLM},
        - {ModelEnum.STATSMODELS_POISSON_GLM},
        - {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")


class StatsmodelsGLM(PredictiveModel, ABC):
    """
    Generalized Linear Model (GLM) implementation using Statsmodels.

    This class provides a flexible framework for fitting GLMs with various
    distributions and link functions. It leverages the `statsmodels` library
    to fit models and supports multiple families such as Gaussian, Gamma,
    Poisson, and Tweedie.

    Key Features:
        - Supports multiple GLM families and link functions.
        - Handles weighted fitting through sample weights.
        - Automatically adjusts predictions to enforce constraints (e.g., non-negative predictions).

    Note:
        Unlike machine learning models such as Random Forest or Gradient Boosting,
        `statsmodels.GLM` does not have hyperparameters like learning rate or tree depth.
        Instead, it focuses on specifying the structure of the model and the data.

    Args:
        config (Config): Configuration object containing model settings.
        model_enum (ModelEnum): Enum specifying the GLM family to use.
        **kwargs: Additional keyword arguments for customization.

    Methods:
        - metric: Returns the evaluation metric based on the GLM family.
        - _updated_hparams: Updates model parameters based on hyperparameters.
        - _fit: Fits the GLM to the provided features and target data.
        - _predict: Generates predictions using the fitted model.
        - get_hparams_space: Returns the hyperparameter search space.
        - get_default_hparams: Returns the default hyperparameter values.
        - get_params: Returns the model parameters for compatibility with sklearn.
        - summary: Returns a summary of the fitted model.

    Supported Families:
        - Gaussian: For continuous data with normal distribution.
        - Gamma: For continuous positive data with skewness.
        - Poisson: For count data.
        - Tweedie: For data with a combination of continuous and discrete components.

    Example Usage:
        config = Config(...)
        model = StatsmodelsGLM(config, model_enum=ModelEnum.STATSMODELS_GAMMA_GLM)
        model._fit(features_df, target_df)
        predictions = model._predict(features_df)
    """

    def __init__(self, config: Config, model_enum: ModelEnum, **kwargs):
        self._model_enum = model_enum
        PredictiveModel.__init__(self, config, **kwargs)

    def _get_model_enum(self) -> ModelEnum:
        return self._model_enum

    def metric(self) -> Metric:
        return get_statsmodels_metric(config=self.config, model_enum=self._get_model_enum(), pred_col=self.pred_col)

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self.model = None
        self._min_y = None
        force_min_y_pred = self._hparams.get("force_min_y_pred")
        self._force_min_y_pred = force_min_y_pred if force_min_y_pred != "auto" else False
        self._fit_intercept = self._hparams.get("fit_intercept", True)
        self._intercept_scale = self._hparams.get("intercept_scale", 1.0)
        self._trg_divisor = self._hparams.get("trg_divisor", 1.0)
        if isinstance(self._intercept_scale, str) and self._intercept_scale not in ["mean"]:
            raise ValueError(f"Intercept scale '{self._intercept_scale=}' not supported. Supported is 'mean' or float.")
        if isinstance(self._trg_divisor, str) and self._trg_divisor not in ["mean"]:
            raise ValueError(f"Target divisor '{self._trg_divisor=}' not supported. Supported is 'mean' or float.")
        match self._get_model_enum():
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                self.family = sm.families.Gaussian()
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                link = self._hparams.get("link")
                match link or "inverse":
                    case "log":
                        link = sm.families.links.Log()
                    case "inverse":
                        link = sm.families.links.InversePower()
                        if force_min_y_pred == "auto":
                            self._force_min_y_pred = True
                    case _:
                        raise ValueError(f"Link '{link}' not supported for Gamma model.")
                # var = None
                # if "variance" in kwargs:
                #     match kwargs["variance"]:
                #         case "mu":
                #             var = sm.families.varfuncs.mu
                #         case "mu_squared":
                #             var = sm.families.varfuncs.mu_squared
                #         case _:
                #             match = re.match(r"power\((\d+(\.\d+)?)\)", kwargs["variance"])
                #             if match:
                #                 p = float(match.group(1))
                #                 var = sm.families.varfuncs.Power(p)
                self.family = sm.families.Gamma(link=link)
            case ModelEnum.STATSMODELS_POISSON_GLM:
                self.family = sm.families.Poisson()
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                link = self._hparams.get("link")
                var_power = self._hparams.get("power")
                if var_power is None:
                    raise ValueError("Tweedie model for Power link requires power parameter.")
                match link or "log":
                    case "log":
                        link = sm.families.links.Log()
                    case "power":
                        if var_power == 1:
                            raise ValueError(
                                "power = 1 is not valid for Tweedie with Power link due to division by zero.")
                        link = sm.families.links.Power(power=1 / (1 - var_power))
                self.family = sm.families.Tweedie(link=link, var_power=var_power)
            case _:
                raise ValueError(
                    f"""Family for model {self._get_model_enum()} not supported in Stastsmodels GLM. Supported families are:
            - \"Gaussian\" for {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
            - \"Gamma\" for {ModelEnum.STATSMODELS_GAMMA_GLM},
            - \"Poisson\" for {ModelEnum.STATSMODELS_POISSON_GLM},
            - \"Tweedie\" fpr {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        logger.debug("StatsmodelsGLM _fit called")
        logger.debug(f"{self.get_hparams()=}")
        if type(target_df) is pd.DataFrame:
            y = target_df[self.target_col]
        else:
            y = target_df
        if self._force_min_y_pred:
            self._min_y = np.min(y)
        self._y_mean = np.mean(y)
        if self._fit_intercept:
            features_df = sm.add_constant(features_df)
            if isinstance(self._intercept_scale, str):
                assert self._intercept_scale == "mean"
                intercept_scale = self._y_mean
            else:
                intercept_scale = self._intercept_scale
            features_df["const"] = intercept_scale
        if isinstance(self._trg_divisor, str):
            assert self._trg_divisor == "mean"
            trg_divisor = self._y_mean
        else:
            trg_divisor = self._trg_divisor
        logger.info(f"Target mean: {self._y_mean}, Target divisor: {trg_divisor}")
        logger.info(f"{y=}")
        y = y / trg_divisor
        logger.info(f"{y=}")
        logger.debug(f"{features_df.head()=}")
        logger.debug(f"{y.head()=}")
        if "sample_weight" in kwargs:
            weights = kwargs["sample_weight"]
            kwargs = {key: val for key, val in kwargs.items() if key != "sample_weight"}
            logger.debug(
                f"Weights - max: {np.max(weights)}, min: {np.min(weights)}, contains NaN: {np.isnan(weights).any()}, contains +/–inf: {np.isinf(weights).any() or np.isneginf(weights).any()}")
            self.model = sm.GLM(y, features_df, family=self.family, var_weights=weights).fit(**kwargs)
        else:
            self.model = sm.GLM(y, features_df, family=self.family).fit(**kwargs)
        logger.debug(f"Model fitted - {self.model.summary()}")
        self._set_features_importance(np.abs(self.model.params))

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        logger.debug("StatsmodelsGLM _predict called")
        logger.debug(f"{self.get_hparams()=}")
        if self._fit_intercept:
            features_df = sm.add_constant(features_df)
            if isinstance(self._intercept_scale, str) and self._intercept_scale == "mean":
                intercept_scale = self._y_mean
            else:
                intercept_scale = self._intercept_scale
            features_df["const"] = intercept_scale
        logger.debug(f"{features_df.head()=}")
        y_pred = self.model.predict(features_df)
        if isinstance(self._trg_divisor, str):
            assert self._trg_divisor == "mean"
            trg_divisor = self._y_mean
        else:
            trg_divisor = self._trg_divisor
        y_pred = y_pred * trg_divisor
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        logger.debug(f"_force_min_y_pred: {self._force_min_y_pred}, _min_y: {self._min_y}")
        if self._force_min_y_pred:
            y_pred = np.maximum(y_pred, self._min_y)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        return y_pred

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {"link": None, "power": None, "force_min_y_pred": "auto", "fit_intercept": True,
                "intercept_scale": 1.0, "trg_divisor": 1.0}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        return dict(config=self.config)
        # return dict(config=copy.deepcopy(self.config))

    def summary(self):
        return self.model.summary()


class StatsmodelsPoissonGLM(StatsmodelsGLM):
    """
    Generalized Linear Model (GLM) using the Poisson family from Statsmodels.

    This class implements a GLM for Poisson-distributed target variables,
    commonly used for modeling count data. It inherits from `StatsmodelsGLM`
    and specifies the Poisson family.

    Args:
        config (Config): Configuration object containing model settings.
        **kwargs: Additional keyword arguments for customization.
    """
    def __init__(self, config: Config, **kwargs):
        StatsmodelsGLM.__init__(self, config=config, model_enum=ModelEnum.STATSMODELS_POISSON_GLM, **kwargs)


class StatsmodelsGammaGLM(StatsmodelsGLM):
    """
    Generalized Linear Model (GLM) using the Gamma family from Statsmodels.

    This class implements a GLM for Gamma-distributed target variables,
    often used for modeling continuous positive data with skewness.
    It inherits from `StatsmodelsGLM` and specifies the Gamma family.

    Args:
        config (Config): Configuration object containing model settings.
        **kwargs: Additional keyword arguments for customization.
    """
    def __init__(self, config: Config, **kwargs):
        StatsmodelsGLM.__init__(self, config=config, model_enum=ModelEnum.STATSMODELS_GAMMA_GLM, **kwargs)


class StatsmodelsTweedieGLM(StatsmodelsGLM):
    """
    Generalized Linear Model (GLM) using the Tweedie family from Statsmodels.

    This class implements a GLM for Tweedie-distributed target variables,
    which is suitable for modeling data with a combination of continuous
    and discrete components (e.g., insurance claims). It inherits from
    `StatsmodelsGLM` and specifies the Tweedie family.

    Args:
        config (Config): Configuration object containing model settings.
        **kwargs: Additional keyword arguments for customization.
    """
    def __init__(self, config: Config, **kwargs):
        StatsmodelsGLM.__init__(self, config=config, model_enum=ModelEnum.STATSMODELS_TWEEDIE_GLM, **kwargs)


class StatsmodelsGaussianGLM(StatsmodelsGLM):
    """
    Generalized Linear Model (GLM) using the Gaussian family from Statsmodels.

    This class implements a GLM for Gaussian-distributed target variables,
    commonly used for modeling continuous data. It inherits from `StatsmodelsGLM`
    and specifies the Gaussian family.

    Args:
        config (Config): Configuration object containing model settings.
        **kwargs: Additional keyword arguments for customization.
    """
    def __init__(self, config: Config, **kwargs):
        StatsmodelsGLM.__init__(self, config=config, model_enum=ModelEnum.STATSMODELS_GAUSSIAN_GLM, **kwargs)


class SklearnGLM(SklearnModel, ABC):
    def __init__(self, config: Config, model_class, **kwargs):
        SklearnModel.__init__(self, config=config, model_class=model_class, **kwargs)

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        np.random.seed(self._random_state)
        super()._fit(features_df, target_df, **kwargs)
        features_importance = pd.Series(np.abs(self.model.coef_))
        if hasattr(self.model, "feature_names_in_"):
            features_importance.index = self.model.feature_names_in_
        self._set_features_importance(features_importance)

    def _updated_hparams(self):
        if "max_iter" in self._hparams and self._hparams["max_iter"] is not None and not np.isnan(self._hparams["max_iter"]):
            self._hparams["max_iter"] = int(self._hparams["max_iter"])
        self._random_state = self._hparams.get("random_state", 0)
        SklearnModel._updated_hparams(self)


class SklearnPoissonGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        SklearnGLM.__init__(self, config=config, model_class=PoissonRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "alpha",
            "fit_intercept",
            "solver",
            "max_iter",
            "tol",
            "warm_start",
            "verbose",
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", np.log(1e-6), np.log(100)),
            "max_iter": hp.quniform("max_iter", 50, 500, 10),
            "tol": hp.loguniform("tol", np.log(1e-6), np.log(1e-2)),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", cls._get_solver_options())
        }

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "newton-cholesky",
            "max_iter": 1000,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0,
            "random_state": 0,
        }


class SklearnGammaGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        logger.debug(f"{kwargs=}")
        SklearnGLM.__init__(self, config=config, model_class=GammaRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanGammaDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "alpha",
            "fit_intercept",
            "solver",
            "max_iter",
            "tol",
            "warm_start",
            "verbose",
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", np.log(1e-6), np.log(100)),
            "max_iter": hp.quniform("max_iter", 50, 500, 10),
            "tol": hp.loguniform("tol", np.log(1e-6), np.log(1e-2)),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", cls._get_solver_options())
        }

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "newton-cholesky",
            "max_iter": 1000,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0,
            "random_state": 0,
        }


class SklearnTweedieGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        SklearnGLM.__init__(self, config=config, model_class=TweedieRegressor, **kwargs)

    def metric(self) -> Metric:
        hparams = self.get_hparams()
        if "power" in hparams:
            return MeanTweedieDeviance(self.config, pred_col=self.pred_col, power=hparams["power"])
        return ConcentrationCurveGiniIndex(self.config)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "alpha",
            "fit_intercept",
            "power",
            "solver",
            "max_iter",
            "tol",
            "warm_start",
            "verbose",
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", np.log(1e-6), np.log(100)),
            "max_iter": hp.quniform("max_iter", 50, 500, 10),
            "tol": hp.loguniform("tol", np.log(1e-6), np.log(1e-2)),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "power": hp.uniform("power", 1.0, 2.0),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "power": 1.5,
            "alpha": 1.0,
            "fit_intercept": True,
            "link": "auto",
            "solver": "newton-cholesky",
            "max_iter": 1000,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0,
            "random_state": 0,
        }


class PyGLMNetGLM(PredictiveModel):
    def __init__(self, config: Config, model_enum: ModelEnum = None, **kwargs):
        self._model_enum = model_enum or config.ds.model
        PredictiveModel.__init__(self, config, **kwargs)

    def _get_model_enum(self) -> ModelEnum:
        return self._model_enum

    def metric(self) -> Metric:
        match self._get_model_enum():
            case ModelEnum.PYGLMNET_GAMMA_GLM:
                return MeanGammaDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.PYGLMNET_POISSON_GLM:
                return MeanPoissonDeviance(self.config, pred_col=self.pred_col)
            case _:
                raise ValueError(f"Family for model {self._get_model_enum()} not supported in PyGLMNetGLM.")

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self.model = None
        self._min_y = None
        force_min_y_pred = self._hparams.get("force_min_y_pred")
        self._force_min_y_pred = force_min_y_pred if force_min_y_pred != "auto" else False
        self._alpha = self._hparams.get("alpha")
        self._reg_lambda = self._hparams.get("reg_lambda")
        self._fit_intercept = self._hparams.get("fit_intercept")
        self._max_iter = int(self._hparams.get("max_iter"))
        self._tol = self._hparams.get("tol")
        self._learning_rate = self._hparams.get("learning_rate")
        self._solver = self._hparams.get("solver")
        self._distr = self._hparams.get("distr")
        self._random_state = self._hparams.get("random_state")
        if self._distr is None:
            match self._get_model_enum():
                case ModelEnum.PYGLMNET_POISSON_GLM:
                    self._distr = "poisson"
                case ModelEnum.PYGLMNET_GAMMA_GLM:
                    self._distr = "gamma"
                case _:
                    raise ValueError(
                        f"""Distribution for model {self._get_model_enum()} not supported in PyGLMNet. Supported distributions are:
                - \"poisson\" for {ModelEnum.PYGLMNET_POISSON_GLM},
                - \"gamma\" for {ModelEnum.PYGLMNET_GAMMA_GLM}.""")
        elif self._distr not in ["poisson", "gamma"]:
            raise ValueError(f"Distribution '{self._distr}' not supported in PyGLMNet. Supported are: \"poisson\", \"gamma\".")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        logger.debug("PyGLMNetGLM _fit called")
        logger.debug(f"{self.get_hparams()=}")
        y = target_df[self.target_col].values if isinstance(target_df, pd.DataFrame) else target_df
        x = features_df.values if isinstance(features_df, pd.DataFrame) else features_df

        if self._force_min_y_pred:
            self._min_y = np.min(y)

        if "sample_weight" in kwargs:
            weights = kwargs["sample_weight"]
            weights = weights.values if isinstance(weights, pd.DataFrame) else weights

            # Check if weights are integers (can be float type but must have integer values)
            if not np.all(np.isclose(weights, weights.astype(int))):
                raise ValueError("Sample weights must have integer values.")

            logger.debug(
                f"Weights - max: {np.max(weights)}, min: {np.min(weights)}, contains NaN: {np.isnan(weights).any()}, contains +/–inf: {np.isinf(weights).any() or np.isneginf(weights).any()}"
            )

            # Add more observations to the model according to the weights
            x = np.repeat(x, weights.astype(int), axis=0)
            y = np.repeat(y, weights.astype(int), axis=0)

        self.model = pyglmnet.GLM(
            distr=self._distr,
            alpha=self._alpha,
            reg_lambda=self._reg_lambda,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
            tol=self._tol,
            learning_rate=self._learning_rate,
            solver=self._solver,
            random_state=self._random_state,
        )
        self.model.fit(x, y)
        logger.debug(f"Model fitted - {self.summary()}")
        importance = np.abs(self.model.beta_)
        logger.debug(f"{importance=}")
        if isinstance(features_df, pd.DataFrame):
            importance = pd.Series(importance, index=features_df.columns)
            logger.debug(f"{importance=}")
        self._set_features_importance(importance)

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        x = features_df.values if isinstance(features_df, pd.DataFrame) else features_df
        y_pred = self.model.predict(x)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        logger.debug(f"_force_min_y_pred: {self._force_min_y_pred}, _min_y: {self._min_y}")
        if self._force_min_y_pred:
            y_pred = np.maximum(y_pred, self._min_y)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        return y_pred

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.uniform("alpha", 0.0, 1.0),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-6), np.log(1)),
            "eta": hp.loguniform("eta", np.log(1e-5), np.log(2)),
            "max_iter": hp.quniform("max_iter", 50, 500, 10),
            "tol": hp.loguniform("tol", np.log(1e-5), np.log(1e-2)),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
            "solver": hp.choice("solver", ["batch-gradient", "cdfast"])
        }

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 0.5,
            "reg_lambda": 0.1,
            "eta": 2.0,
            "fit_intercept": True,
            "force_min_y_pred": False,
            "max_iter": 1000,
            "tol": 1e-6,
            "learning_rate": 0.01,
            "solver": "batch-gradient",
            "random_state": 0,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)

    def summary(self):
        return f"PyGLM intercept: {self.model.beta0_}, coefficients: {self.model.beta_}"
