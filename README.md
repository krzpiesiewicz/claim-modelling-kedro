# claim-modelling-kedro

Project for my master thesis.

Automates insurance claim modeling in Kedro: data processing, sampling, feature engineering, feature selection, GLM/LightGBM tuning, recalibration, and validation. Configurable via YAML. Supports generating, scheduling, and tracking hundreds of experiments. Outputs: metrics and charts. Uses Kedro, MLflow, Python.

## Dataset

The dataset used in this project can be found at:
[Dataset Link](https://students.mimuw.edu.pl/~kp385996/ai-insurance/data.zip)

## Basic scripts

### create_venv.sh

This script creates a Python virtual environment and installs the required dependencies.

**Usage:**
```sh
./create_venv.sh
```

### setup_postgres.sh

This script sets up a PostgreSQL database for a mlflow server. It creates a database named `mlflow_db` and a user named `mlflow_user` with the specified password. The script also grants all privileges on the database to the user.

**Usage:**
```sh
./setup_postgres.sh
```
The script will prompt you for the PostgreSQL password for mlflow user.

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

---

## How to Create and Run a Batch of Experiments

### Creating a Batch of Experiments

1. In the `experiments` directory, copy the `experiments_dir_template` directory and rename it to your desired experiment name, e.g., `my_experiment_name`.

2. Edit the files in `experiments/my_experiment_name/templates/` (`parameters.yml` and `mlflow.yml`) to define the parameters for your experiment.

3. In any notebook in the project (e.g., `notebooks/my_experiment_analysis.ipynb`), run the following method to create a new experiment run:

    ```python
    create_experiment_run(
        experiment_name=experiment_name,
        run_name=run_name,
        template_parameters=template_parameters
    )
    ```

    where:
   - `experiment_name` is the name of your experiment directory,
   - `run_name` is the name of the new MLflow run,
   - `template_parameters` is a dictionary of parameters whose values will replace the template tags in the files located in `experiments/<experiment_name>/templates/`.

    You can call this method multiple times with different pairs of `run_name` and `template_parameters` to create multiple runs for the same experiment.

    Import the method in your notebook as follows:

    ```python
    from claim_modelling_kedro.experiments.experiment import (
        create_experiment_run,
        default_run_name_from_run_no
    )
    ```

---

### Running a Batch of Experiments

1. Run the experiment using the provided `run_experiment.sh` script. See usage below.

2. Restore the default configuration files using the `restore_default_config.sh` script. See usage below.

---

### Viewing Experiment Results in a Notebook

Useful methods for viewing experiment results in your notebook:

- From `claim_modelling_kedro.experiments.experiment`:
  - `get_run_mlflow_id`

- From `claim_modelling_kedro.pipelines.utils.dataframes`:
  - `load_metrics_table_from_mlflow`
  - `load_predictions_and_target_from_mlflow`
  - `load_metrics_cv_stats_from_mlflow`

- From `claim_modelling_kedro.pipelines.utils.datasets`:
  - `get_partition`
  - `get_mlflow_run_id_for_partition`

---

## Scripts for Running Experiments

### run_experiment.sh

This script runs an experiment for different pipelines.  
It requires the experiment name and the name of the first pipeline to run as positional arguments.

Optionally, you can specify:
- another pipeline to run for all subsequent runs, and
- a specific run name.

The script copies the rendered templates from `experiments/<experiment_name>/templates/` to the Kedro configuration directory.

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

---

## Scripts for Managing MLflow Experiments

### manage_mlflow_experiment.sh

This script allows you to delete, restore, or permanently delete (purge) an MLflow experiment.  
It reads the tracking URI from: `claim_modelling_kedro/conf/local/mlflow.yml`.

Supported actions:
- delete  – soft-deletes the experiment (marks it as *deleted*)
- restore – restores a soft-deleted experiment
- purge   – permanently deletes the experiment  
  ⚠️ Requires MLflow ≥ 2.7 and SQL backend.

**Usage:**
```sh
./manage_mlflow_experiment.sh <delete|restore|purge> (--name <experiment_name> | --id <experiment_id>)
```

**Examples:**
- Soft-delete by name
    ```sh
    ./manage_mlflow_experiment.sh delete --name sev_001_dummy_mean_regressor
    ```
- Restore by ID
    ```sh
    ./manage_mlflow_experiment.sh restore --id 12
    ```
- Permanently delete by name
    ```sh
    ./manage_mlflow_experiment.sh purge --name sev_001_dummy_mean_regressor
    ```

### list_mlflow_experiments.sh

This script lists all MLflow experiments along with their name, ID, and lifecycle stage.  
It uses the tracking URI configured in `claim_modelling_kedro/conf/local/mlflow.yml`.

**Usage:**
```sh
./list_mlflow_experiments.sh
```

Output includes:
- experiment name
- experiment ID
- status (active or deleted)
