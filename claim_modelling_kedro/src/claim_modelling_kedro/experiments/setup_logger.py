import logging
import logging.config
import yaml

def setup_logger():
    with open("conf/logging.yml", "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
