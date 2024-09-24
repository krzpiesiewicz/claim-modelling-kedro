from dataclasses import dataclass
from typing import List

from solvency_models.pipelines.p01_init.data_config import DataConfig
from solvency_models.pipelines.p01_init.mdl_info_config import ModelInfo, Target
from solvency_models.pipelines.p01_init.metric_config import MetricType, MetricEnum, TWEEDIE_DEV, WEIGHTED_TWEEDIE_DEV


@dataclass
class ModelTask:
    target_col: str
    prediction_col: str
    evaluation_metrics: List[MetricType]

    def __init__(self, mdl_info: ModelInfo, data: DataConfig):
        match mdl_info.target:
            case Target.CLAIMS_NUMBER:
                self.target_col = data.claims_number_target_col
                self.prediction_col = data.claims_number_pred_col
                self.evaluation_metrics = [MetricEnum.POISSON_DEV, MetricEnum.WEIGHTED_POISSON_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK,
                                           MetricEnum.GINI, MetricEnum.WEIGHTED_GINI]
            case Target.FREQUENCY:
                self.target_col = data.claims_freq_target_col
                self.prediction_col = data.claims_freq_pred_col
                self.evaluation_metrics = [MetricEnum.POISSON_DEV, MetricEnum.WEIGHTED_POISSON_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK,
                                           MetricEnum.GINI, MetricEnum.WEIGHTED_GINI]
            case Target.TOTAL_AMOUNT:
                self.target_col = data.claims_total_amount_target_col
                self.prediction_col = data.claims_total_amount_pred_col
                self.evaluation_metrics = [MetricEnum.RMSE, MetricEnum.WEIGHTED_RMSE, MetricEnum.R2,
                                           MetricEnum.MBD, MetricEnum.WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK,
                                           MetricEnum.GINI, MetricEnum.WEIGHTED_GINI] + \
                                          [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]] +\
                                          [WEIGHTED_TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
            case Target.POSITIVE_AMOUNT:
                self.target_col = data.claims_total_amount_target_col
                self.prediction_col = data.claims_total_amount_pred_col
                self.evaluation_metrics = [MetricEnum.GAMMA_DEV, MetricEnum.WEIGHTED_GAMMA_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK,
                                           MetricEnum.GINI, MetricEnum.WEIGHTED_GINI] + \
                                          [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]] +\
                                          [WEIGHTED_TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
            case Target.SEVERITY:
                self.target_col = data.claims_severity_target_col
                self.prediction_col = data.claims_severity_pred_col
                self.evaluation_metrics = [MetricEnum.RMSE, MetricEnum.WEIGHTED_RMSE, MetricEnum.R2,
                                           MetricEnum.MBD, MetricEnum.WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.WEIGHTED_R2, MetricEnum.SPEARMAN_RANK,
                                           MetricEnum.GINI, MetricEnum.WEIGHTED_GINI] + \
                                          [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]] +\
                                          [WEIGHTED_TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
            case Target.POSITIVE_SEVERITY:
                self.target_col = data.claims_severity_target_col
                self.prediction_col = data.claims_severity_pred_col
                self.evaluation_metrics = [MetricEnum.GAMMA_DEV, MetricEnum.WEIGHTED_GAMMA_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK,
                                           MetricEnum.GINI, MetricEnum.WEIGHTED_GINI] + \
                                          [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]] +\
                                          [WEIGHTED_TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
