from dataclasses import dataclass
from typing import List

from claim_modelling_kedro.pipelines.p01_init.data_config import DataConfig
from claim_modelling_kedro.pipelines.p01_init.mdl_info_config import ModelInfo, Target
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricType, MetricEnum, TWEEDIE_DEV, \
    EXP_WEIGHTED_TWEEDIE_DEV


@dataclass
class ModelTask:
    target_col: str
    prediction_col: str
    exposure_weighted: bool
    claim_nb_weighted: bool
    evaluation_metrics: List[MetricType]

    def __init__(self, mdl_info: ModelInfo, data: DataConfig):
        match mdl_info.target:
            case Target.CLAIMS_NUMBER:
                self.target_col = data.claims_number_target_col
                self.prediction_col = data.claims_number_pred_col
                self.exposure_weighted = True
                self.claim_nb_weighted = False
                self.evaluation_metrics = [MetricEnum.POISSON_DEV, MetricEnum.EXP_WEIGHTED_POISSON_DEV,
                                           MetricEnum.MAE, MetricEnum.EXP_WEIGHTED_MAE, MetricEnum.RMSE,
                                           MetricEnum.EXP_WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.EXP_WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.EXP_WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.EXP_WEIGHTED_SPEARMAN,
                                           MetricEnum.CI, MetricEnum.EXP_WEIGHTED_CI,
                                           MetricEnum.ABC, MetricEnum.EXP_WEIGHTED_ABC,
                                           MetricEnum.CCI, MetricEnum.EXP_WEIGHTED_CCI,
                                           MetricEnum.COI, MetricEnum.EXP_WEIGHTED_COI,
                                           MetricEnum.CUI, MetricEnum.EXP_WEIGHTED_CUI]
            case Target.FREQUENCY:
                self.target_col = data.claims_freq_target_col
                self.prediction_col = data.claims_freq_pred_col
                self.exposure_weighted = True
                self.claim_nb_weighted = False
                self.evaluation_metrics = [MetricEnum.POISSON_DEV, MetricEnum.EXP_WEIGHTED_POISSON_DEV,
                                           MetricEnum.MAE, MetricEnum.EXP_WEIGHTED_MAE, MetricEnum.RMSE,
                                           MetricEnum.EXP_WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.EXP_WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.EXP_WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.EXP_WEIGHTED_SPEARMAN,
                                           MetricEnum.CI, MetricEnum.EXP_WEIGHTED_CI,
                                           MetricEnum.ABC, MetricEnum.EXP_WEIGHTED_ABC,
                                           MetricEnum.CCI, MetricEnum.EXP_WEIGHTED_CCI,
                                           MetricEnum.COI, MetricEnum.EXP_WEIGHTED_COI,
                                           MetricEnum.CUI, MetricEnum.EXP_WEIGHTED_CUI]
            case Target.TOTAL_AMOUNT:
                self.target_col = data.claims_total_amount_target_col
                self.prediction_col = data.claims_total_amount_pred_col
                self.exposure_weighted = True
                self.claim_nb_weighted = False
                self.evaluation_metrics = [MetricEnum.MAE, MetricEnum.EXP_WEIGHTED_MAE,
                                           MetricEnum.RMSE, MetricEnum.EXP_WEIGHTED_RMSE, MetricEnum.R2,
                                           MetricEnum.MBD, MetricEnum.EXP_WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.EXP_WEIGHTED_SPEARMAN,
                                           MetricEnum.CI, MetricEnum.EXP_WEIGHTED_CI,
                                           MetricEnum.ABC, MetricEnum.EXP_WEIGHTED_ABC,
                                           MetricEnum.CCI, MetricEnum.EXP_WEIGHTED_CCI,
                                           MetricEnum.COI, MetricEnum.EXP_WEIGHTED_COI,
                                           MetricEnum.CUI, MetricEnum.EXP_WEIGHTED_CUI] + \
                                          [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]] + \
                                          [EXP_WEIGHTED_TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
            case Target.AVG_CLAIM_AMOUNT:
                self.target_col = data.claims_avg_amount_target_col
                self.prediction_col = data.claims_avg_amount_pred_col
                self.exposure_weighted = False
                self.claim_nb_weighted = True
                self.evaluation_metrics = [MetricEnum.GAMMA_DEV, MetricEnum.CLNB_WEIGHTED_GAMMA_DEV,
                                           MetricEnum.MAE, MetricEnum.CLNB_WEIGHTED_MAE, MetricEnum.RMSE,
                                           MetricEnum.CLNB_WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.CLNB_WEIGHTED_R2,
                                           MetricEnum.MBD, MetricEnum.CLNB_WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.CLNB_WEIGHTED_SPEARMAN,
                                           MetricEnum.CI, MetricEnum.CLNB_WEIGHTED_CI,
                                           MetricEnum.ABC, MetricEnum.CLNB_WEIGHTED_ABC,
                                           MetricEnum.CCI, MetricEnum.CLNB_WEIGHTED_CCI,
                                           MetricEnum.COI, MetricEnum.CLNB_WEIGHTED_COI,
                                           MetricEnum.CUI, MetricEnum.CLNB_WEIGHTED_CUI]
            case Target.PURE_PREMIUM:
                self.target_col = data.claims_pure_premium_target_col
                self.prediction_col = data.claims_pure_premium_pred_col
                self.exposure_weighted = True
                self.claim_nb_weighted = False
                self.evaluation_metrics = [MetricEnum.MAE, MetricEnum.EXP_WEIGHTED_MAE,
                                           MetricEnum.RMSE, MetricEnum.EXP_WEIGHTED_RMSE, MetricEnum.R2,
                                           MetricEnum.MBD, MetricEnum.EXP_WEIGHTED_MBD,
                                           MetricEnum.SPEARMAN, MetricEnum.EXP_WEIGHTED_SPEARMAN,
                                           MetricEnum.CI, MetricEnum.EXP_WEIGHTED_CI,
                                           MetricEnum.ABC, MetricEnum.EXP_WEIGHTED_ABC,
                                           MetricEnum.CCI, MetricEnum.EXP_WEIGHTED_CCI,
                                           MetricEnum.COI, MetricEnum.EXP_WEIGHTED_COI,
                                           MetricEnum.CUI, MetricEnum.EXP_WEIGHTED_CUI] + \
                                          [TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]] + \
                                          [EXP_WEIGHTED_TWEEDIE_DEV(p) for p in [1.2, 1.5, 1.7, 1.9]]
