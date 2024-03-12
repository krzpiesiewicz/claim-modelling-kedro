import logging
from typing import Dict

from solvency_models.pipelines.p01_init.config import Config

logger = logging.getLogger(__name__)


def process_parameters(parameters: Dict) -> Config:
    logger.info("Processing parameters...")
    config = Config(parameters)
    logger.info("...processed. Created config classes.")
    return config
