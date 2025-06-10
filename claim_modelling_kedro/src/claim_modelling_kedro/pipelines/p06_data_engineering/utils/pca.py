import logging
import pandas as pd
from sklearn.decomposition import PCA

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader

logger = logging.getLogger(__name__)

_pca_artifact_path = "model_de/pca"
_pca_model_name = "features pca model"


def get_pca_column_name(i: int, n: int) -> str:
    """
    Returns the name of a PCA column in the format "PC{i}", where i is zero-padded to match the length of the decimal representation of n.
    Args:
        i (int): Column number (1-indexed).
        n (int): Number determining the length of the decimal representation.

    Returns:
        str: Column name in the format "PC{i}".
    """
    width = len(str(n))
    return f"PC{i:0{width}d}"


def add_index_and_pc_columns(
        transformed_fearues: pd.DataFrame, index: pd.Index
) -> pd.DataFrame:
    """
    Adds index and PCA column names to the transformed features DataFrame.
    """
    columns = [get_pca_column_name(i, transformed_fearues.shape[1]) for i in range(transformed_fearues.shape[1])]
    return pd.DataFrame(transformed_fearues, index=index, columns=columns)


def transform_features_pca_by_mlflow_model(features_df: pd.DataFrame, mlflow_run_id: str = None) -> pd.DataFrame:
    logger.info("Transforming features using PCA from MLFlow model...")
    pca_model = MLFlowModelLoader(_pca_model_name).load_model(path=_pca_artifact_path, run_id=mlflow_run_id)
    transformed_df = pca_model.transform(features_df)
    transformed_df = add_index_and_pc_columns(transformed_df, index=features_df.index)
    logger.info(f"Transformed features with PCA.")
    logger.info(f"Transformed features with PCA, resulting in {transformed_df.shape[1]} components.")
    return transformed_df


def fit_transform_features_pca(config: Config, sample_features_df: pd.DataFrame) -> pd.DataFrame:
    svd_solver = "auto"
    pca_model = PCA(n_components=config.de.pca_n_components, random_state=config.de.pca_random_state,
                    svd_solver=svd_solver)
    logger.info("Fitting PCA model...")
    transformed_df = pca_model.fit_transform(sample_features_df)
    transformed_df = add_index_and_pc_columns(transformed_df, index=sample_features_df.index)
    logger.info("Fitted PCA model.")
    # Log the PCA model to MLFlow
    MLFlowModelLogger(pca_model, _pca_model_name).log_model(_pca_artifact_path)
    logger.info("Logged PCA model to MLFlow.")
    explained_var_ratio = pca_model.explained_variance_ratio_
    explained_variance_df = pd.DataFrame({
        "explained_variance_percent": explained_var_ratio * 100,
        "cumulative_variance_percent": explained_var_ratio.cumsum() * 100
    }, index=transformed_df.columns).round(2)
    explained_variance_df = explained_variance_df.astype(str) + '%'
    logger.info(f"Transformed features with PCA, resulting in {transformed_df.shape[1]} components.\n"
                f"Explained variance per PCA component (%):\n{explained_variance_df}")
    return transformed_df

