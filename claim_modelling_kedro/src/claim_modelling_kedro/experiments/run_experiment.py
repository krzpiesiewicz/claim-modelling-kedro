import argparse
import logging
import os
import subprocess
import time

from claim_modelling_kedro.experiments import copy_experiment_run_configs, run_kedro_pipeline
from claim_modelling_kedro.experiments.setup_logger import setup_logger

# Configure the logger
setup_logger()
logger = logging.getLogger(__name__)


def copy_config_and_run_experiment(base_dir: str, experiment_dir: str, experiment_name: str, run_name: str,
                                   pipeline: str) -> float:
    copy_experiment_run_configs(experiment_dir, run_name)
    os.chdir(base_dir)
    run_start_time = time.time()
    run_kedro_pipeline(experiment_name, run_name, pipeline)
    run_end_time = time.time()
    return run_end_time - run_start_time


def main():
    parser = argparse.ArgumentParser(description="Run an experiment for different pipelines.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("first_pipeline", type=str, help="Pipeline to run for the first run")
    parser.add_argument("--other-pipeline", type=str, help="Pipeline for all other runs", default=None)

    # Mutually exclusive group: only one of these can be provided
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-name", type=str, nargs="+", help="Run names to execute")
    group.add_argument("--from-run-name", type=str, help="Start from this run (inclusive)")

    args = parser.parse_args()
    experiment_name = args.experiment_name
    first_pipeline = args.first_pipeline
    other_pipeline = args.other_pipeline or first_pipeline

    base_dir = os.path.abspath(os.getcwd())
    experiment_dir = os.path.join(base_dir, "experiments", experiment_name)
    runs_dir = os.path.join(experiment_dir, "runs")

    run_names = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    if "dummy" in run_names:
        run_names.remove("dummy")

    if not run_names:
        logger.error(f"No run directories found in {runs_dir}")
        return

    if args.run_name:
        invalid_runs = [r for r in args.run_name if r not in run_names]
        if invalid_runs:
            logger.error(f"The following run names were not found in {runs_dir}: {', '.join(invalid_runs)}")
            return
        selected_run_names = args.run_name
    elif args.from_run_name:
        if args.from_run_name not in run_names:
            logger.error(f"Run '{args.from_run_name}' not found in {runs_dir}")
            return
        idx = run_names.index(args.from_run_name)
        selected_run_names = run_names[idx:]
    else:
        selected_run_names = run_names

    logger.info(f"Running {len(selected_run_names)} run(s) for experiment '{experiment_name}'")
    logger.info(f"Runs:\n   - " + ",\n   - ".join(selected_run_names) + ".")

    total_runs = len(selected_run_names)
    run_times = []
    start_time = time.time()

    for idx, run_name in enumerate(selected_run_names):
        pipeline = first_pipeline if idx == 0 else other_pipeline
        logger.info(f"Starting run {idx + 1}/{total_runs} - '{run_name}' using pipeline: {pipeline}")

        run_duration = copy_config_and_run_experiment(base_dir, experiment_dir, experiment_name, run_name, pipeline)

        run_times.append(run_duration)
        elapsed_time = time.time() - start_time
        mean_time = sum(run_times) / len(run_times)
        remaining = total_runs - (idx + 1)
        eta = mean_time * remaining

        logger.info(f"Completed run {idx + 1}/{total_runs} ({(idx + 1) / total_runs:.2%})")
        logger.info(f"Mean time: {mean_time:.2f}s | ETA: {eta:.2f}s | Elapsed: {elapsed_time:.2f}s")

        # Notification via notify-send
        title = f"Completed run no {idx + 1} from {experiment_name}"
        message = (
            f"Completed run {run_name} from {experiment_name}\n"
            f"Completed: {idx + 1}/{total_runs} ({(idx + 1) / total_runs:.2%})\n"
            f"Mean time: {mean_time:.2f}s | ETA: {eta:.2f}s | Elapsed: {elapsed_time:.2f}s"
        )
        subprocess.run([
            "notify-send",
            "-a", "run_experiment.sh",
            "-u", "normal",
            "-t", "0",
            title,
            message
        ])

    logger.info("Completed all experiment runs.")
    # Final notification via notify-send
    if args.run_name:
        k = len(args.run_name)
        if k > 6:
            abbreviated_list = ", ".join(args.run_name[:3] + ["..."] + args.run_name[-3:])
            run_list_str = f"{abbreviated_list} ({k} runs)"
        else:
            run_list_str = ", ".join(args.run_name)
        summary_line = f"Completed selected {k} run(s) from '{experiment_name}': {run_list_str}"
    elif args.from_run_name:
        idx_first = run_names.index(selected_run_names[0]) + 1  # 1-based
        idx_last = run_names.index(selected_run_names[-1]) + 1
        summary_line = f"Completed experiment runs {idx_first} to {idx_last} from '{experiment_name}'"
    else:
        summary_line = f"Completed all {total_runs} experiment runs from '{experiment_name}'"

    title = f"Finished experiment: {experiment_name}"
    message = (
        f"{summary_line}\n"
        f"Total time: {time.time() - start_time:.2f}s"
    )
    subprocess.run([
        "notify-send",
        "-a", "run_experiment.sh",
        "-u", "normal",
        "-t", "0",
        title,
        message
    ])


if __name__ == "__main__":
    main()
