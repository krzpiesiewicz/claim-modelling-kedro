import os
import logging

from claim_modelling_kedro.experiments import restore_default_config_files
from claim_modelling_kedro.experiments.setup_logger import setup_logger

# Configure the logger
setup_logger()
logger = logging.getLogger(__name__)

def main():
    # Get the current working directory (solvency-modelling)
    base_dir = os.path.abspath(os.getcwd())
    restore_default_config_files(base_dir=base_dir)

if __name__ == "__main__":
    main()
