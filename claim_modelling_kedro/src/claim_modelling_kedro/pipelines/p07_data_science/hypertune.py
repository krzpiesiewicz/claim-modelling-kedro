import copy
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event, Manager
from datetime import timedelta
from functools import partial
from typing import Dict, Any, Tuple

import hyperopt
import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, Trials, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss
from mlflow import MlflowClient
from tabulate import tabulate

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.ds_config import HyperoptAlgoEnum, HypertuneValidationEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel, save_ds_model_in_mlflow
from claim_modelling_kedro.pipelines.utils.dataframes import save_pd_dataframe_as_csv_in_mlflow
from claim_modelling_kedro.pipelines.utils.datasets import get_partition, get_mlflow_run_id_for_partition
from claim_modelling_kedro.pipelines.utils.metrics import get_metric_from_enum, Metric
from claim_modelling_kedro.pipelines.utils.stratified_cv_split import get_stratified_train_test_cv
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path, \
    round_decimal

logger = logging.getLogger(__name__)

# Define the artifact path for hyperparameters tuning
_hypertune_artifact_path = "hyperopt"


def get_hparams_from_trial(trial: Dict[str, Any], space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hyperparameters from a trial.
    """
    return space_eval(space, {k: v[0] if isinstance(v, list) else v for k, v in trial["misc"]["vals"].items()})


def get_best_trial(trials: Trials) -> Dict[str, Any]:
    """
    Get the best trial from the trials.
    """
    trials_losses = [(trial["result"]["loss"] if trial["result"]["status"] == STATUS_OK else np.inf) for trial in trials.trials ]
    best_index = np.argmin(trials_losses)
    return trials.trials[best_index]


def log_best_trials_info_to_mlflow(
        best_trials: Dict[str, Dict[str, Any]],
        best_hparams: Dict[str, Dict[str, Any]],
        log_folds_metrics: bool,
        artifact_path: str
) -> None:
    best_metrics = {part: copy.deepcopy(best_trial.get("attachments", {}).get("metrics", {})) for part, best_trial in best_trials.items()}
    for part, best_metrics_part in best_metrics.items():
        for metric_name in best_metrics_part:
            best_metrics_part[metric_name] = round_decimal(best_metrics_part[metric_name], significant_digits=4)
    best_trials_dct = {part: best_metrics[part] | best_hparams[part] for part in best_hparams.keys()}
    columns = list(best_trials_dct[next(iter(best_trials_dct))].keys())
    best_trials_df = pd.DataFrame.from_dict(best_trials_dct, orient="index", columns=columns).sort_index()
    best_trials_df.index.name = "part"
    save_pd_dataframe_as_csv_in_mlflow(best_trials_df, artifact_path, "best_trials.csv", index=True)
    logger.info("Best hypertune results for each partition:\n" +
                tabulate(best_trials_df, headers="keys", tablefmt="psql", stralign="right", showindex=True) + "\n")

    if log_folds_metrics:
        for dataset in ["valid", "train"]:
            rows = []
            for part, trial in best_trials.items():
                folds_scores = (trial.get("attachments") or {}).get(f"{dataset}_scores") or []
                row_dct = {
                    "part": part,
                    "mean": round_decimal(np.mean(folds_scores), significant_digits=4),
                    "std": round_decimal(np.std(folds_scores), significant_digits=4) if len(folds_scores) > 1 else "",
                    **{fold: round_decimal(score, significant_digits=4) for fold, score in
                       enumerate(folds_scores)},
                }
                rows.append(row_dct)
            best_trials_scores_df = pd.DataFrame(rows).set_index("part").sort_index()
            file_name = f"best_trials_{dataset}_scores.csv"
            save_pd_dataframe_as_csv_in_mlflow(best_trials_scores_df, artifact_path, file_name, index=True)
            logger.info(f"Best hypertune trials folds {dataset.upper()} scores for each partition:\n" +
                        tabulate(best_trials_scores_df, headers="keys", tablefmt="psql", stralign="right", showindex=True) + "\n")


def log_trials_info_to_mlflow(
        trials: Trials,
        space: Dict[str, Any],
        log_folds_metrics: bool,
        artifact_path: str
) -> None:
    """
    Create a DataFrame from the trials and log it to MLFlow.
    """
    trials_df = pd.DataFrame([{
        "trial_no": trial["tid"] + 1,
        "book_time": trial["book_time"],
        "eval_time": trial["result"].get("eval_time"),
        "status": trial["result"].get("status"),
        "loss": round_decimal(trial["result"].get("loss"), 3)
    } for trial in trials.trials])
    trials_df = pd.concat([trials_df, pd.DataFrame([{
        **{k: round_decimal(v, 3) for k, v in ((trial.get("attachments") or {}).get("metrics") or {}).items()}
    } for trial in trials.trials])], axis=1)
    trials_df = pd.concat([trials_df,
                           pd.DataFrame([get_hparams_from_trial(trial, space) for trial in trials.trials])], axis=1)
    save_pd_dataframe_as_csv_in_mlflow(trials_df, artifact_path, "trials.csv", index=False)

    if log_folds_metrics:
        rows = []
        for trial in trials.trials:
            for dataset in ["valid", "train"]:
                rows.append({
                    "trial_no": trial["tid"] + 1,
                    "dataset": dataset,
                    **{fold: round_decimal(score, 4) for fold, score in
                       enumerate((trial.get("attachments") or {}).get(f"{dataset}_scores") or [])},
                })
        trials_scores_df = pd.DataFrame(rows)
        save_pd_dataframe_as_csv_in_mlflow(trials_scores_df, artifact_path, "trials_folds_scores.csv", index=False)


def create_hyperopt_result_message(trials: Trials, current_trial: Dict[str, Any], part: str, max_evals: int, is_best: bool = False) -> str:
    """
    Create a message summarizing the results of a hyperopt trial.

    Args:
        current_trial (Dict[str, Any]): A dictionary containing trial information.
        part (str): The partition of the dataset for which the trial was run.
        is_best (bool): A flag indicating if this is the best trial.

    Returns:
        str: A formatted message summarizing the trial results.
    """
    trial_no = current_trial["tid"] + 1
    eval_time = current_trial["result"].get("eval_time")
    status = current_trial["result"].get("status")
    loss = current_trial["result"].get("loss")
    attachments = current_trial.get("attachments", {})
    metrics = attachments.get("metrics", {})
    metric_name = attachments.get("metric_name")
    train_scores = attachments.get("train_scores", [])
    val_scores = attachments.get("valid_scores", [])

    if is_best:
        n_trials = len(trials.trials)
        msg = (f"Final hyperopt results for partition: '{part}':\nbest trial: {trial_no}/{n_trials} - evaluation time: {eval_time}, status: {status}, loss: {loss}\nmetrics:\n")
    else:
        best_trial = get_best_trial(trials)
        best_trial_no = best_trial["tid"] + 1
        best_loss = best_trial["result"].get("loss")
        msg = (f"Hyperopt trial {trial_no}/{max_evals} for partition '{part}' - evaluation time: {eval_time}, status: {status}\n" +
               f"loss: {loss} \[best trial {best_trial_no}/{max_evals} - best loss: {best_loss}]\n" +
               f"metrics:\n")

    if metrics:
        train_mean = round_decimal(metrics.get(f"{metric_name}_train"), significant_digits=4)
        valid_mean = round_decimal(metrics.get(f"{metric_name}_valid"), significant_digits=4)
        train_std = round_decimal(metrics.get(f"{metric_name}_std_train"), significant_digits=4)
        valid_std = round_decimal(metrics.get(f"{metric_name}_std_valid"), significant_digits=4)
        scores_df = pd.DataFrame({
            f"valid": map(partial(round_decimal, significant_digits=4), val_scores),
            f"train": map(partial(round_decimal, significant_digits=4), train_scores)
        })
        scores_df.index.name = "fold"

        if train_std is not None and valid_std is not None:
            msg += (
                f"\n{metric_name}_valid: {valid_mean}  (std: {valid_std})\n"
                f"{metric_name}_train: {train_mean}  (std: {train_std})\n"
                f"{metric_name}_scores:\n{scores_df}"
            )
        else:
            msg += (
                f"\n{metric_name}_valid: {valid_mean}\n"
                f"{metric_name}_train: {train_mean}"
            )
    return msg


def process_fold(config: Config, fold: str, train_keys_cv: Dict[str, pd.Index], val_keys_cv: Dict[str, pd.Index],
                 selected_sample_features_df: pd.DataFrame, sample_target_df: pd.DataFrame,
                 hparams: Dict[str, Any], model: PredictiveModel, metric: Metric) -> Tuple[float, float, PredictiveModel]:
    train_keys = train_keys_cv[fold]
    val_keys = val_keys_cv[fold]

    model = get_class_from_path(config.ds.model_class)(config=config, target_col=config.mdl_task.target_col,
                                                       pred_col=config.mdl_task.prediction_col)
    model.update_hparams(config.ds.model_const_hparams)
    model.update_hparams(hparams)
    model.fit(selected_sample_features_df, sample_target_df, sample_train_keys=train_keys, sample_val_keys=val_keys)

    train_predictions_df = model.predict(selected_sample_features_df.loc[train_keys, :])
    val_predictions_df = model.predict(selected_sample_features_df.loc[val_keys, :])

    train_score = metric.eval(sample_target_df.loc[train_keys, :], train_predictions_df)
    val_score = metric.eval(sample_target_df.loc[val_keys, :], val_predictions_df)

    return train_score, val_score, model


class TaskCancelledError(Exception):
    """
    Raised when a task is cancelled due to a global failure in another parallel task.
    """
    def __init__(self, message="Task was cancelled due to another failure."):
        super().__init__(message)


def fit_model(hparams: Dict[str, any],
              model: PredictiveModel,
              config: Config,
              trials: Trials,
              space: Dict[str, Any],
              metric: Metric,
              selected_sample_features_df: pd.DataFrame,
              sample_target_df: pd.DataFrame,
              train_keys_cv: Dict[str, pd.Index],
              val_keys_cv: Dict[str, pd.Index],
              hyperopt_artifact_path: str,
              part: str,
              cancel_event: Event) -> float:
    if cancel_event.is_set():
        raise TaskCancelledError(f"Task for part {part} was cancelled due to earlier failure.")
    trial = trials.trials[-1]
    trial_no = len(trials.trials)
    max_evals = config.ds.hopt_max_evals
    msg = f"Hyperopt trial {trial_no}/{max_evals} for partition '{part}'. Hyperparameters:\n"
    for param, value in hparams.items():
        msg += f"    - {param}: {value}\n"
    msg += f"Fitting the predictive model(s) for hyperopt trial {trial_no}..."
    logger.info(msg)

    log_trials_info_to_mlflow(trials, space, log_folds_metrics=(len(train_keys_cv) > 1),
                              artifact_path=hyperopt_artifact_path)

    train_scores = []
    val_scores = []
    metric_name = metric.get_short_name()
    start_time = time.time()

    fitted_model = None
    try:
        with ProcessPoolExecutor(max_workers=5) as executor:  # Ustaw odpowiednią liczbę procesów
            futures = {
                executor.submit(process_fold, config, fold, train_keys_cv, val_keys_cv,
                                selected_sample_features_df, sample_target_df,
                                hparams, model, metric): fold
                for fold in train_keys_cv.keys()
            }
            for future in as_completed(futures):
                train_score, val_score, fitted_model = future.result()
                train_scores.append(train_score)
                val_scores.append(val_score)
                logger.debug(
                    f"Part '{part}'. Hyperopt fold: {futures[future]}. Train score ({metric_name}): {train_score}, Validation score ({metric_name}): {val_score}.")

        train_score_mean = np.mean(train_scores)
        val_score_mean = np.mean(val_scores)
        loss_val = -val_score_mean if metric.is_larger_better() else val_score_mean
        loss_train = -train_score_mean if metric.is_larger_better() else train_score_mean
        loss = loss_val + config.ds.hopt_overfit_penalty * max(0, loss_val - loss_train)
        if len(train_keys_cv) > 1:
            train_score_std = np.std(train_scores)
            val_score_std = np.std(val_scores)
        status = STATUS_OK
    except Exception as e:
        logger.error(f"Error during hyperopt trial {trial_no}: {e}")
        logger.debug(traceback.format_exc())
        train_score_mean = np.nan
        val_score_mean = np.nan
        train_score_std = np.nan
        val_score_std = np.nan
        loss = np.nan
        status = STATUS_FAIL

    eval_time = time.time() - start_time
    formatted_time = str(timedelta(seconds=eval_time))
    metrics_results = {
        f"{metric_name}_valid": val_score_mean,
        f"{metric_name}_train": train_score_mean,
    }
    if len(train_keys_cv) > 1:
        metrics_results.update({
            f"{metric_name}_std_valid": val_score_std,
            f"{metric_name}_std_train": train_score_std,
        })

    attachments = {
        "metric_name": metric_name,
        "metrics": metrics_results,
        "train_scores": train_scores,
        "valid_scores": val_scores,
        "model": fitted_model
    }
    trial_result = {
        "trial_no": trial_no,
        "book_time": time.gmtime(start_time),
        "eval_time": formatted_time,
        "status": status,
        "loss": loss,
    }
    trial["attachments"] = attachments
    trial["result"] = trial_result
    msg = create_hyperopt_result_message(trials, trial, part=part, max_evals=config.ds.hopt_max_evals)
    logger.info(msg)
    log_trials_info_to_mlflow(trials, space, log_folds_metrics=(len(train_keys_cv) > 1),
                              artifact_path=hyperopt_artifact_path)
    return {
        **trial_result,
        "attachments": attachments
    }


def hypertune_part(config: Config, selected_sample_features_df: pd.DataFrame,
                   sample_target_df: pd.DataFrame, sample_train_keys: pd.Index,
                   sample_val_keys: pd.Index, part: str, hyperopt_artifact_path: str,
                   save_best_hparams_in_mlfow: bool = True, cancel_event: Event = None
) -> Tuple[Trials, Dict[str, Any], Dict[str, Any]]:
    match config.ds.hopt_algo:
        case HyperoptAlgoEnum.TPE:
            hopt_algo = hyperopt.tpe.suggest
        case HyperoptAlgoEnum.RANDOM:
            hopt_algo = hyperopt.random.suggest
        case _:
            raise ValueError(f"Hyperopt algorithm {config.ds.hopt_algo} not supported.")
    model = get_class_from_path(config.ds.model_class)(config=config, target_col=config.mdl_task.target_col,
                                                       pred_col=config.mdl_task.prediction_col)
    model.update_hparams(config.ds.model_const_hparams)
    metric = model.metric() if config.ds.hopt_metric is None else get_metric_from_enum(config, config.ds.hopt_metric,
                                                                                       pred_col=config.mdl_task.prediction_col)
    hparam_space = model.get_hparams_space()
    for hparam in config.ds.hopt_excluded_params + list(config.ds.model_const_hparams.keys()):
        if hparam in hparam_space:
            hparam_space.pop(hparam)
    if hparam_space is None or len(hparam_space) == 0:
        raise Exception(f"No hyperparameter space defined for {config.ds.hopt_algo}.")

    # Set up the hyperopt validation method - using the sample train and validation keys
    # sample_val_set - use a sample validation set already defined in the sampling process
    # cross_validation - use cross-validation with the sample train and validation keys
    # repeated_split - use repeated stratified split with the sample train and validation keys
    match config.ds.hopt_validation_method:
        case HypertuneValidationEnum.SAMPLE_VAL_SET:
            train_keys_cv = {"0": sample_train_keys}
            val_keys_cv = {"0": sample_val_keys}
        case HypertuneValidationEnum.CROSS_VALIDATION:
            train_keys_cv, val_keys_cv = get_stratified_train_test_cv(sample_target_df,
                                                                      stratify_target_col=config.mdl_task.target_col,
                                                                      cv_folds=config.ds.hopt_cv_folds, shuffle=True,
                                                                      random_seed=config.ds.hopt_cv_random_seed,
                                                                      verbose=False)
        case HypertuneValidationEnum.REPEATED_SPLIT:
            train_keys_cv = {}
            val_keys_cv = {}
            for fold in range(config.ds.hopt_repeated_split_n_repeats):
                train_keys, val_keys = get_stratified_train_test_split_keys(
                    sample_target_df,
                    stratify_target_col=config.mdl_task.target_col,
                    test_size=config.ds.hopt_repeated_split_val_size,
                    shuffle=True,
                    random_seed=max(config.ds.hopt_repeated_split_random_seed + 100, 1) * (100 * fold + 1),
                    verbose=False)
                train_keys_cv[str(fold)] = train_keys
                val_keys_cv[str(fold)] = val_keys

    trials = Trials()
    objective = partial(fit_model, config=config, model=model, trials=trials, space=hparam_space, metric=metric,
                        selected_sample_features_df=selected_sample_features_df,
                        sample_target_df=sample_target_df, train_keys_cv=train_keys_cv,
                        val_keys_cv=val_keys_cv, part=part, hyperopt_artifact_path=hyperopt_artifact_path,
                        cancel_event=cancel_event)
    if config.ds.hopt_early_stop_enabled:
        hp_early_stop = no_progress_loss(iteration_stop_count=config.ds.hopt_early_iteration_stop_count,
                                         percent_increase=config.ds.hopt_early_stop_percent_increase)
    else:
        hp_early_stop = None
    try:
        hp_assignment = fmin(
            fn=objective,
            space=hparam_space,
            algo=hopt_algo,
            max_evals=config.ds.hopt_max_evals,
            trials=trials,
            rstate=np.random.default_rng(config.ds.hopt_fmin_random_seed),
            early_stop_fn=hp_early_stop
        )
    except TaskCancelledError as e:
        logger.error(f"Task for partition '{part}' was cancelled due to earlier failure.")
        mlflow.set_tag("task_status", "cancelled")
        mlflow.end_run(status="FAILED")
        raise e
    except Exception as e:
        logger.error(f"Error during hyperopt tuning for partition '{part}': {e}\n"
                     f"{traceback.format_exc()}")
        mlflow.set_tag("task_status", "failed")
        mlflow.end_run(status="FAILED")
        cancel_event.set()
        raise
    best_trial = get_best_trial(trials)
    best_hparams = get_hparams_from_trial(best_trial, hparam_space)
    logger.info(f"Best hyperparameters for partition '{part}':\n{best_hparams=}")

    if save_best_hparams_in_mlfow:
        mlflow.log_dict(best_hparams, f"{_hypertune_artifact_path}/best_hparams.yml")
    return trials, best_trial, best_hparams


def process_hypertune_part(config: Config, part: str, selected_sample_features_df: Dict[str, pd.DataFrame],
                           sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                           sample_val_keys: Dict[str, pd.Index], save_best_hparams_in_mlflow: bool,
                           parent_mlflow_run_id: str = None, cancel_event: Event = None
                           ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    try:
        selected_sample_features_part_df = get_partition(selected_sample_features_df, part)
        sample_target_part_df = get_partition(sample_target_df, part)
        sample_train_keys_part = get_partition(sample_train_keys, part)
        sample_val_keys_part = get_partition(sample_val_keys, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mlflow_run_id=parent_mlflow_run_id)
        logger.info(f"Tuning the hyper parameters of the predictive model on partition '{part}' of the sample dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            trials, best_trial, best_hparams_part = hypertune_part(
                config, selected_sample_features_part_df, sample_target_part_df,
                sample_train_keys_part, sample_val_keys_part, part=part,
                hyperopt_artifact_path=f"{_hypertune_artifact_path}",
                cancel_event=cancel_event
            )
            msg = create_hyperopt_result_message(trials, best_trial, part=part, is_best=True, max_evals=config.ds.hopt_max_evals)
            msg += f"\nThe best hyperparameters for partition '{part}':\n"
            for param, value in best_hparams_part.items():
                msg += f"    - {param}: {value}\n"
            logger.info(msg)
            if config.ds.hopt_enabled and config.ds.hopt_validation_method == HypertuneValidationEnum.SAMPLE_VAL_SET:
                fitted_model = best_trial["attachments"].pop("model")
                save_ds_model_in_mlflow(fitted_model)
    except Exception as e:
        cancel_event.set()
        raise
    return part, best_hparams_part, best_trial


def hypertune(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
              sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
              sample_val_keys: Dict[str, pd.Index], save_best_hparams_in_mlflow: bool = True
              ) -> Dict[str, Dict[str, Any]]:
    logger.info(f"Tuning the hyper parameters of the predictive model {config.ds.model_class}...")
    parts_cnt = len(selected_sample_features_df)
    best_hparams = {}
    best_trials = {}
    manager = Manager()
    cancel_event = manager.Event()
    mlflow_run_id = mlflow.active_run().info.run_id
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_hypertune_part, config, part, selected_sample_features_df, sample_target_df,
                            sample_train_keys, sample_val_keys, save_best_hparams_in_mlflow, mlflow_run_id, cancel_event): part
            for part in selected_sample_features_df.keys()
        }
        try:
            for future in as_completed(futures):
                part = futures[future]
                try:
                    part1, best_hparams_part, best_trial = future.result()
                    assert part1 == part, "The part returned from the future does not match the key."
                    best_hparams[part] = best_hparams_part
                    best_trials[part] = best_trial
                    logger.info(f"The best hyperparameters for partition '{part}' of the sample dataset:\n"
                                f"{best_hparams_part=}")
                except Exception as e:
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(f"Hypertuning failed on part '{part}'") from e
        except Exception:
            logger.error(f"Task hypertune was cancelled due to failure in a subprocess.")
            mlflow.set_tag("task_status", "failed")
            raise

    if save_best_hparams_in_mlflow:
        mlflow.log_dict(best_hparams, f"{_hypertune_artifact_path}/best_trials_hparams.yml")
    log_best_trials_info_to_mlflow(best_trials, best_hparams, artifact_path=_hypertune_artifact_path,
                                   log_folds_metrics=(config.ds.hopt_validation_method != HypertuneValidationEnum.SAMPLE_VAL_SET))
    return best_hparams
