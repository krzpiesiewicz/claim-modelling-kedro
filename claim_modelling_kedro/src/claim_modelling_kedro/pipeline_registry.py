"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from claim_modelling_kedro.pipelines import (
    p01_init, p02_data_preprocessing, p03_define_modeling_task, p04_data_splitting, p05_sampling, p06_data_engineering,
    p_dummy, p07_data_science, p08_model_calibration, p09_test, p10_load_pred_and_trg, p11_charts
, p11_recompute_metrics
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dummy_pipeline = p_dummy.create_pipeline()
    init_pipeline = p01_init.create_pipeline()
    data_processing_pipeline = p02_data_preprocessing.create_pipeline()
    def_modeling_task_pipeline = p03_define_modeling_task.create_pipeline()
    data_splitting_pipeline = p04_data_splitting.create_pipeline()
    sampling_pipeline = p05_sampling.create_pipeline()
    data_engineering_pipeline = p06_data_engineering.create_pipeline()
    data_science_pipeline = p07_data_science.create_pipeline()
    calibration_pipeline = p08_model_calibration.create_pipeline()
    test_pipeline = p09_test.create_pipeline()
    load_pred_and_trg_pipeline = p10_load_pred_and_trg.create_pipeline()
    charts_pipeline = p11_charts.create_pipeline()
    recompute_metrics_pipeline = p11_recompute_metrics.create_pipeline()
    pipelines = dict()
    pipelines["dummy"] = dummy_pipeline
    pipelines["init"] = init_pipeline
    pipelines["dp"] = init_pipeline + data_processing_pipeline
    pipelines["deftask"] = init_pipeline + def_modeling_task_pipeline
    pipelines["split"] = init_pipeline + def_modeling_task_pipeline + data_splitting_pipeline
    pipelines[
        "dp_split"] = init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + data_splitting_pipeline
    pipelines["smpl"] = init_pipeline + def_modeling_task_pipeline + sampling_pipeline
    pipelines["split_smpl"] = init_pipeline + def_modeling_task_pipeline + data_splitting_pipeline + sampling_pipeline
    pipelines["de"] = init_pipeline + data_engineering_pipeline
    pipelines["smpl_de"] = init_pipeline + def_modeling_task_pipeline + sampling_pipeline + data_engineering_pipeline
    pipelines["ds"] = init_pipeline + data_science_pipeline
    pipelines["de_ds"] = init_pipeline + data_engineering_pipeline + data_science_pipeline
    pipelines["clb"] = init_pipeline + calibration_pipeline
    pipelines["de_ds_clb"] = init_pipeline + data_engineering_pipeline + data_science_pipeline + calibration_pipeline
    pipelines["ds_clb"] = init_pipeline + data_science_pipeline + calibration_pipeline
    pipelines["smpl_de_ds"] = (init_pipeline + def_modeling_task_pipeline + sampling_pipeline +
                               data_engineering_pipeline + data_science_pipeline)
    pipelines["test"] = init_pipeline + test_pipeline
    pipelines["clb_test"] = init_pipeline + calibration_pipeline + test_pipeline
    pipelines["ds_clb_test"] = init_pipeline + data_science_pipeline + calibration_pipeline + test_pipeline
    pipelines[
        "de_ds_clb_test"] = (init_pipeline + data_engineering_pipeline + data_science_pipeline + calibration_pipeline +
                             test_pipeline)
    pipelines[
        "smpl_de_ds_clb_test"] = (init_pipeline + def_modeling_task_pipeline + sampling_pipeline +
                                  data_engineering_pipeline + data_science_pipeline + calibration_pipeline +
                                  test_pipeline)
    pipelines["smpl_to_test"] = pipelines["smpl_de_ds_clb_test"]
    pipelines["split_to_test"] = (pipelines["split_smpl"] + data_engineering_pipeline + data_science_pipeline +
                                  calibration_pipeline + test_pipeline)
    pipelines["de_to_test"] = pipelines["de_ds_clb_test"]
    pipelines["ds_to_test"] = pipelines["ds_clb_test"]
    pipelines["charts"] = init_pipeline + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["test_charts"] = init_pipeline + test_pipeline + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["clb_test_charts"] = pipelines["clb_test"] + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["clb_to_charts"] = pipelines["clb_test_charts"]
    pipelines["split_to_charts"] = pipelines["split_to_test"] + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["ds_to_charts"] = pipelines["ds_to_test"] + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["de_to_charts"] = pipelines["de_to_test"] + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["smpl_to_charts"] = pipelines["smpl_to_test"] + load_pred_and_trg_pipeline + charts_pipeline
    pipelines["all_to_split"] = pipelines["dp_split"]
    pipelines["all_to_smpl"] = (init_pipeline + data_processing_pipeline + def_modeling_task_pipeline +
                                data_splitting_pipeline + sampling_pipeline)
    pipelines["all_to_de"] = (
            init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + data_splitting_pipeline +
            sampling_pipeline + data_engineering_pipeline)
    pipelines["all_to_ds"] = (
            init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + data_splitting_pipeline +
            sampling_pipeline + data_engineering_pipeline + data_science_pipeline)
    pipelines["all_to_clb"] = (
            init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + data_splitting_pipeline +
            sampling_pipeline + data_engineering_pipeline + data_science_pipeline + calibration_pipeline)
    pipelines["all_to_test"] = (
            init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + data_splitting_pipeline +
            sampling_pipeline + data_engineering_pipeline + data_science_pipeline + calibration_pipeline +
            test_pipeline)
    pipelines["all_to_charts"] = (
            init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + data_splitting_pipeline +
            sampling_pipeline + data_engineering_pipeline + data_science_pipeline + calibration_pipeline +
            test_pipeline + load_pred_and_trg_pipeline + charts_pipeline)
    pipelines["recompute_metrics"] = init_pipeline + load_pred_and_trg_pipeline + recompute_metrics_pipeline
    return pipelines
