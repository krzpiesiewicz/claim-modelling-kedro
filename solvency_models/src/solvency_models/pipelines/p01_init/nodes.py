import logging
from typing import Dict

import mlflow

from solvency_models.pipelines.p01_init.config import Config, log_config_to_mlflow, add_sub_runs_ids_to_config

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
    return config
