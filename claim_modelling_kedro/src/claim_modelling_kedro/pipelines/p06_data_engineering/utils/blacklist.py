import logging
import os
import tempfile
from typing import Dict, List

import mlflow
import pandas as pd


logger = logging.getLogger(__name__)

_features_blacklist_artifact_dir = "model_de"
_features_blacklist_filename = "features_blacklist.txt"


def parse_features_blacklist(features_blacklist: str) -> List[str]:
    """
    Parsuje blacklistę cech ze stringa.
    Pomija komentarze (linie zaczynające się od #) i puste linie.
    """
    if not features_blacklist:
        return []
    return [
        line.split("#")[0].strip()
        for line in features_blacklist.splitlines()
        if line.strip() and not line.strip().startswith("#") and line.split("#")[0].strip()
    ]


def remove_features_from_blacklist(transformed_features_df: pd.DataFrame, features_blacklist: List[str]):
    if features_blacklist:
        logger.info("Removing features from the blacklist:\n" + \
                    ",\n".join([f"  - {ftr}" for ftr in features_blacklist]))
        transformed_features_df = transformed_features_df.drop(columns=features_blacklist, errors="ignore")
    return transformed_features_df


def remove_features_from_mlflow_blacklist(features_df: pd.DataFrame, mlflow_run_id: str) -> pd.DataFrame:
    # Load the feature blacklist from MLFlow
    blacklist_path = mlflow.artifacts.download_artifacts(
        run_id=mlflow_run_id,
        artifact_path=f"{_features_blacklist_artifact_dir}/{_features_blacklist_filename}"
    )
    with open(blacklist_path, "r") as f:
        features_blacklist_text = f.read()
    features_blacklist = parse_features_blacklist(features_blacklist_text)
    return remove_features_from_blacklist(features_df, features_blacklist)


def save_features_blacklist_to_mlflow(features_blacklist_text: str):
    logger.info("Saving features blacklist to MLFlow...")
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, _features_blacklist_filename)
        with open(file_path, "w") as f:
            f.write(features_blacklist_text)
        mlflow.log_artifact(file_path, artifact_path=_features_blacklist_artifact_dir)
    logger.info(f"Saved features blacklist to MLFlow as {_features_blacklist_artifact_dir}/{_features_blacklist_filename}")
