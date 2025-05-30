import logging
import os
import shutil
import re
from typing import Dict, Any
import yaml

from claim_modelling_kedro.experiments.setup_logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def fill_templates_and_copy(template_dir: str, target_dir: str, template_parameters: Dict[str, str]) -> None:
    """
    Copies the files from template_dir to the target_dir.
    In the target_dir, the placeholders in the copied files are replaced with the values from template_parameters dict.
    The placeholders are in the format of <PLACEHOLDER>.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over files in the template directory
    for root, dirs, files in os.walk(template_dir):
        for file_name in files:
            template_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(template_file_path, template_dir)
            target_file_path = os.path.join(target_dir, relative_path)

            # Create the target directories if not exist
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

            # Read the template file and replace placeholders
            with open(template_file_path, 'r') as template_file:
                content = template_file.read()

            for placeholder, value in template_parameters.items():
                content = re.sub(f"<{placeholder}>", value, content)

            # Write the replaced content to the target directory
            with open(target_file_path, 'w') as target_file:
                target_file.write(content)

    logger.info(f"Filled templates copied to {target_dir}.")


def default_run_name_from_run_no(experiment_name: str, run_no: int, suffix: str = None) -> str:
    return f"{experiment_name}_{str(run_no).zfill(5)}" + (f"_{suffix}" if suffix is not None else "")


def create_experiment_run(experiment_name: str, run_name: str, template_parameters: Dict[str, str]) -> None:
    """
    Creates a new run directory for a specific experiment.
    """
    experiment_dir = os.path.join("experiments", "experiment_name")
    templates_dir = os.path.join(experiment_dir, "templates")
    runs_dir = os.path.join(experiment_dir, "runs")

    # Format run number with leading zeros
    run_dir = os.path.join(runs_dir, run_name)

    # Create the run directory if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)

    if not os.path.exists(templates_dir):
        msg = f"Templates directory '{templates_dir}' does not exist. No templates to fill."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Update the placeholder for the name of the mlflow experiment
    template_parameters.update(MLFLOW_EXPERIMENT_NAME=experiment_name, MLFLOW_RUN_NAME=run_name)
    # Save the template parameters in the run directory
    template_parameters_file = os.path.join(run_dir, "template_parameters.yaml")
    with open(template_parameters_file, 'w') as file:
        yaml.dump(template_parameters, file)

    # Copy and fill templates into filled_templates directory
    filled_templates_dir = os.path.join(run_dir, "filled_templates")
    fill_templates_and_copy(templates_dir, filled_templates_dir, template_parameters)

    logger.info(f"Created experiment run directory: {run_dir}.")


def copy_experiment_run_configs(experiment_dir: str, run_name: str, target_dir_path: str = None,
                                target_rel_paths: Dict[str, str] = None) -> None:
    """
    Copies the filled templates from the run directory to the target_dir_path with subdirectories/files specified in target_rel_paths.

    Args:
        experiment_dir (str): The directory of the experiment.
        run_dir (str): The directory of the run (relative to experiment_dir)
        target_dir_path (str, optional): The root directory where files will be copied to. Defaults to ""conf".
        target_rel_paths (Dict[str, str], optional): A dictionary mapping the source file names to their target relative paths.
            Example: {"parameters.yml": "base/parameters.yml", "mlflow.yml": "local/mlflow.yml"}.
            Defaults to copying `parameters.yml` and `mlflow.yml` to standard locations in `base/` and `local/`.
    """
    # Default values for target_rel_paths and target_dir_path
    if target_rel_paths is None:
        target_rel_paths = {
            "parameters.yml": "base/parameters.yml",
            "mlflow.yml": "local/mlflow.yml"
        }
    if target_dir_path is None:
        target_dir_path = "conf"

    # Construct the run directory path (run number with leading zeros)
    run_dir = os.path.join(experiment_dir, "runs", run_name,
                           "filled_templates")

    # Check if the run directory exists
    if not os.path.exists(run_dir):
        logger.error(f"Run directory '{run_dir}' does not exist.")
        return

    # Copy files from the run directory to their respective target locations
    for source_file, target_rel_path in target_rel_paths.items():
        source_file_path = os.path.join(run_dir, source_file)
        target_file_path = os.path.join(target_dir_path, target_rel_path)

        # Ensure the source file exists
        if not os.path.exists(source_file_path):
            msg = f"Source file '{source_file_path}' does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Ensure the target directory exists
        target_dir = os.path.dirname(target_file_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # Copy the file
        shutil.copyfile(source_file_path, target_file_path)
        logger.info(f"Copied '{source_file_path}' to '{target_file_path}'.")

    logger.info(f"Finished copying filled templates for run {run_name}.")


def restore_default_config_files(base_dir: str) -> None:
    for source_file, target_rel_path in {
        "parameters.yml": "base/parameters.yml",
        "mlflow.yml": "local/mlflow.yml"
    }.items():
        source_file_path = os.path.join(base_dir, "conf", "default", source_file)
        target_file_path = os.path.join(base_dir, "conf", target_rel_path)

        # Ensure the source file exists
        if not os.path.exists(source_file_path):
            logger.error(f"Source file '{source_file_path}' does not exist. Skipping.")
            continue

        # Ensure the target directory exists
        target_dir = os.path.dirname(target_file_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # Copy the file
        shutil.copyfile(source_file_path, target_file_path)
        logger.info(f"Restored default config files: '{source_file_path}' to '{target_file_path}'.")


def run_kedro_pipeline(experiment_name: str, run_name: str, pipeline: str) -> None:
    """
    Runs the Kedro pipeline for the run in command line.
    `./kedro_run.sh --pipeline {pipeline} --experiment_name {experiment_name} --experiment_run_name {run_name}`
    """
    run_dir = f"experiments/{experiment_name}/runs/{run_name}"
    command = f"cd .. && ./kedro_run.sh --pipeline {pipeline} --experiment-and-run-names {experiment_name} {run_name}"
    os.system(command)
    logger.info(f"Ran Kedro pipeline: {command}.")


def get_run_mlflow_id(experiment_name: str, run_name: str, base_dir: str = None) -> str:
    """
    Reads the mlflow_run_id from the file.
    """
    run_dir = f"experiments/{experiment_name}/runs/{run_name}"
    run_dir = os.path.join(base_dir, run_dir) if base_dir is not None else run_dir
    mlflow_id_file = os.path.join(run_dir, "mlflow_run_id.txt")

    if os.path.exists(mlflow_id_file):
        with open(mlflow_id_file, 'r') as file:
            mlflow_run_id = file.read().strip()
        logger.info(f"MLflow Run ID for {experiment_name}, Run {run_name}: {mlflow_run_id}")
        return mlflow_run_id
    else:
        logger.info(f"MLflow run ID file {mlflow_id_file} not found.")
        return None


def set_run_mlflow_id(experiment_name: str, run_name: str, mlflow_run_id: str) -> None:
    """
    Writes the mlflow_run_id to the file.
    """
    run_dir = f"experiments/{experiment_name}/runs/{run_name}"
    mlflow_id_file = os.path.join(run_dir, "mlflow_run_id.txt")

    with open(mlflow_id_file, 'w') as file:
        file.write(mlflow_run_id)
    logger.info(f"MLflow Run ID {mlflow_run_id} written to {mlflow_id_file}.")
