import logging
from typing import Dict

import mlflow

from claim_modelling_kedro.pipelines.p01_init.config import Config, log_config_to_mlflow, add_sub_runs_ids_to_config

logger = logging.getLogger(__name__)


def process_parameters(parameters: Dict) -> Config:
    # Save parameters to MLFlow as an artifact
    logger.info("Saving parameters.yml to MLFlow as an artifact...")
    mlflow.log_artifact("conf/base/parameters.yml")
    logger.info("Saved parameters.yml to MLFlow as an artifact.")
    # Process parameters
    logger.info("Processing parameters...")
    config = Config(parameters)
    logger.info("Processed. Created config classes.")
    # Set sub runs ids
    if config.data.cv_enabled:
        logger.info("Adding sub runs ids to config...")
        config = add_sub_runs_ids_to_config(config)
        logger.info("Added sub runs ids to config.")
    # Save config to MLFlow
    logger.info("Saving config to MLFlow...")
    log_config_to_mlflow(config)
    logger.info("Saved config to MLFlow.")
    # Save description to MLFlow
    logger.info("Saving description to MLFlow...")
    mlflow.set_tag("mlflow.note.content", config.exprmnt.description)
    logger.info("Saved description to MLFlow...")
    # Sving tags
    logger.info("Saving tags to MLFlow...")
    tags = {
        "author": config.exprmnt.author,
        "target": config.exprmnt.target.value,
        "model": config.ds.model.value,
        "fs": config.ds.fs_method.value,
        "clb": config.clb.method.value,
    }
    for key, val in tags.items():
        mlflow.set_tag(key, val)
    logger.info("Saved tags to MLFlow.")
    return config
