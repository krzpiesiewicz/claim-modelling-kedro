from dataclasses import dataclass
from typing import List

from claim_modelling_kedro.pipelines.p01_init.data_config import DataConfig
from claim_modelling_kedro.pipelines.p01_init.exprmnt import ExperimentInfo, Target, TargetWeight
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricType, MetricEnum, TWEEDIE_DEV, get_weighted_metric_enum


@dataclass
class ModelTask:
    target_col: Target
    prediction_col: str
    exposure_weighted: bool
    claim_nb_weighted: bool
    evaluation_metrics: List[MetricType]

    def __init__(self, exprmnt: ExperimentInfo, data: DataConfig):
        self.exposure_weighted = (exprmnt.target_weight == TargetWeight.EXPOSURE)
        self.claim_nb_weighted = (exprmnt.target_weight == TargetWeight.CLAIMS_NUMBER)
        # Add common metrics
        self.evaluation_metrics = [
            MetricEnum.MAE,
            MetricEnum.RMSE,
            MetricEnum.R2,
            MetricEnum.MBD,
            MetricEnum.SPEARMAN,
            MetricEnum.ABC,
            MetricEnum.SUP_CL_DIFF,
            MetricEnum.CCI,
            MetricEnum.COI,
            MetricEnum.CUI,
            MetricEnum.LC_GINI,
            MetricEnum.CC_GINI,
            MetricEnum.NORMALIZED_CC_GINI,
        ]
        # Add metrics depending on the target
        match exprmnt.target:
            case Target.CLAIMS_NUMBER:
                self.target_col = data.claims_number_target_col
                self.prediction_col = data.claims_number_pred_col
                self.evaluation_metrics.append(MetricEnum.POISSON_DEV)
            case Target.FREQUENCY:
                self.target_col = data.claims_freq_target_col
                self.prediction_col = data.claims_freq_pred_col
                self.evaluation_metrics.append(MetricEnum.POISSON_DEV)
            case Target.TOTAL_AMOUNT:
                self.target_col = data.claims_total_amount_target_col
                self.prediction_col = data.claims_total_amount_pred_col
                self.evaluation_metrics += [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
            case Target.AVG_CLAIM_AMOUNT:
                self.target_col = data.claims_avg_amount_target_col
                self.prediction_col = data.claims_avg_amount_pred_col
                self.evaluation_metrics.append(MetricEnum.GAMMA_DEV)
            case Target.PURE_PREMIUM:
                self.target_col = data.claims_pure_premium_target_col
                self.prediction_col = data.claims_pure_premium_pred_col
                self.evaluation_metrics += [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
        # Convert to weighted metrics if needed
        self.evaluation_metrics = [
            get_weighted_metric_enum(metric_enum, exprmnt.target_weight) for metric_enum in self.evaluation_metrics
        ]
