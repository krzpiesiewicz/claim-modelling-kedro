import os
import argparse
import logging
import time

from solvency_models.experiments import copy_experiment_run_configs, run_kedro_pipeline

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Run an experiment for different pipelines.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("first_pipeline", type=str, help="Pipeline to run for the first run")
    parser.add_argument("other_pipeline", type=str, help="Pipeline to run for all subsequent runs")

    # Parse arguments
    args = parser.parse_args()
    experiment_name = args.experiment_name
    first_pipeline = args.first_pipeline
    other_pipeline = args.other_pipeline

    # Get the current working directory (solvency-modelling)
    base_dir = os.path.abspath(os.getcwd())

    # Experiment directory path
    experiment_dir = os.path.join(base_dir, "experiments", experiment_name)

    # Runs directory path
    runs_dir = os.path.join(experiment_dir, "runs")

    # Get all run directories in alphabetical order
    run_names = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    dummy_run_name = "dummy"
    if dummy_run_name in run_names:
        run_names.remove(dummy_run_name)

    if not run_names:
        logger.error(f"No run directories found in {runs_dir}")
        return

    total_runs = len(run_names)
    completed_runs = 0
    start_time = time.time()
    run_times = []

    # Iterate through the run directories in alphabetical order
    for idx, run_name in enumerate(run_names):
        # Determine the pipeline to use: first for the first run, others for the rest
        if idx == 0:
            pipeline = first_pipeline
        else:
            pipeline = other_pipeline

        # Log start of the run
        logger.info(f"Starting run {idx + 1}/{total_runs} - '{run_name}' using pipeline: {pipeline}")

        # Call the function to copy experiment run configs
        copy_experiment_run_configs(experiment_dir, run_name)

        # Run the Kedro pipeline
        os.chdir(base_dir)
        run_start_time = time.time()
        run_kedro_pipeline(experiment_name, run_name, pipeline)
        run_end_time = time.time()

        # Log run completion
        run_duration = run_end_time - run_start_time
        run_times.append(run_duration)
        completed_runs += 1
        elapsed_time = time.time() - start_time
        mean_run_time = sum(run_times) / len(run_times)

        remaining_runs = total_runs - completed_runs
        expected_remaining_time = mean_run_time * remaining_runs if remaining_runs > 0 else 0
        percentage_complete = (completed_runs / total_runs) * 100

        logger.info(f"Completed run {idx + 1}/{total_runs} ({percentage_complete:.2f}%)")
        logger.info(f"Mean time of past runs: {mean_run_time:.2f} seconds")
        logger.info(f"Expected time for remaining runs: {expected_remaining_time:.2f} seconds")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    logger.info("All experiment runs completed.")

if __name__ == "__main__":
    main()
