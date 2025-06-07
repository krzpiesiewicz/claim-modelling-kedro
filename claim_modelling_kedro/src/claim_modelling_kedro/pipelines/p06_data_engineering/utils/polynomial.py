import logging
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader

logger = logging.getLogger(__name__)

_polynomial_artifact_path = "model_de/polynomial"
_polynomial_model_name = "features polynomial model"


def fit_transform_features_polynomial(config: Config, sample_features_df: pd.DataFrame) -> pd.DataFrame:
    poly = PolynomialFeatures(
        degree=config.de.poly_degree,
        include_bias=config.de.poly_include_bias,
        interaction_only=config.de.poly_interaction_only
    )

    logger.info("Fitting PolynomialFeatures model...")
    transformed_array = poly.fit_transform(sample_features_df)
    feature_names = poly.get_feature_names_out(sample_features_df.columns)
    transformed_df = pd.DataFrame(transformed_array, index=sample_features_df.index, columns=feature_names)
    logger.info("Fitted PolynomialFeatures model.")

    # Log the model
    MLFlowModelLogger(poly, _polynomial_model_name).log_model(_polynomial_artifact_path)
    logger.info("Logged PolynomialFeatures model to MLFlow.")
    logger.info(f"Transformed features with PolynomialFeatures, resulting in {transformed_df.shape[1]} features.")

    return transformed_df


def transform_features_polynomial_by_mlflow_model(features_df: pd.DataFrame, mlflow_run_id: str = None) -> pd.DataFrame:
    logger.info("Transforming features using PolynomialFeatures from MLFlow model...")
    poly_model = MLFlowModelLoader(_polynomial_model_name).load_model(path=_polynomial_artifact_path, run_id=mlflow_run_id)
    transformed_array = poly_model.transform(features_df)
    feature_names = poly_model.get_feature_names_out(features_df.columns)
    transformed_df = pd.DataFrame(transformed_array, index=features_df.index, columns=feature_names)
    logger.info(f"Transformed features with PolynomialFeatures, resulting in {transformed_df.shape[1]} features.")
    return transformed_df
