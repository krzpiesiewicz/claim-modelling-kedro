import logging
from typing import Dict

import mlflow

from solvency_models.pipelines.p01_init.config import Config, log_config_to_mlflow

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
    logger.info("Saving config to MLFlow...")
    log_config_to_mlflow(config)
    logger.info("Saved config to MLFlow.")
    return config
