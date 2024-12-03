import os
import argparse
from claim_modelling_kedro.experiments import copy_experiment_run_configs

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Copy experiment run configuration.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("run_name", type=str, help="Specific run name to copy")

    # Parse arguments
    args = parser.parse_args()
    experiment_name = args.experiment_name
    run_name = args.run_name

    # Get the current working directory (solvency-modelling)
    base_dir = os.path.abspath(os.getcwd())

    # Copy experiment run config
    copy_experiment_run_configs(base_dir, experiment_name, run_name)

if __name__ == "__main__":
    main()
