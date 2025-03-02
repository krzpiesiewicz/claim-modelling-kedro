# claim-modelling-kedro

Project for my master thesis

## Dataset

The dataset used in this project can be found at:
[Dataset Link](https://students.mimuw.edu.pl/~kp385996/ai-insurance/data.zip)

## Scripts

### create_venv.sh

This script creates a Python virtual environment and installs the required dependencies.

**Usage:**
```sh
./create_venv.sh
```

### run_mlflow.sh

This script starts the MLflow server for tracking experiments.

**Usage:**
```sh
./run_mlflow.sh
```

### run_jupyterlab.sh

This script starts a JupyterLab server for interactive data analysis and development.

**Usage:**
```sh
./run_jupyterlab.sh
```

### kedro_run.sh

This script runs the Kedro pipeline defined in the project.

**Usage:**
```sh
./kedro_run.sh [OPTIONS]
```

**Options:**
- `--pipeline`, `-p`: Specify the pipeline to run (default: `__default__`).
- `--mlflow-run-id`: Continue the MLflow run with the given run ID.

### run_experiment.sh

This script runs an experiment for different pipelines. It takes the experiment name and the first pipeline to run as required arguments. Optionally, you can specify another pipeline to run for all subsequent runs and a specific run name.

**Usage:**
```sh
./run_experiment.sh <experiment_name> <first_pipeline> [--other-pipeline <other_pipeline>] [--run-name <run_name>]
```

### restore_default_config.sh

This script restores the default configuration files for the project from `claim_modelling_kedro/conf/default/`.

**Usage:**
```sh
./restore_default_config.sh
```

