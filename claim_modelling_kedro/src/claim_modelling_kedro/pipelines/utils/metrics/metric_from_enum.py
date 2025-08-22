from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum, TWEEDIE_DEV, EXP_WEIGHTED_TWEEDIE_DEV, \
    CLNB_WEIGHTED_TWEEDIE_DEV, MAX_ABS_BIAS_BINS
from claim_modelling_kedro.pipelines.utils.metrics.cumulative_calibration_index import CumulativeCalibrationIndex, \
    CumulativeOverpricingIndex, CumulativeUnderpricingIndex
from claim_modelling_kedro.pipelines.utils.metrics.area_between_cc_and_lc import AreaBetweenCCAndLC
from claim_modelling_kedro.pipelines.utils.metrics.bins_metrics import MaxAbsBiasBinsMetric
from claim_modelling_kedro.pipelines.utils.metrics.sklearn_like_metric import MeanAbsoluteError, RootMeanSquaredError, R2, \
    MeanBiasDeviation, MeanPoissonDeviance, MeanGammaDeviance, SpearmanCorrelation, MeanTweedieDeviance, SklearnLikeMetric
from claim_modelling_kedro.pipelines.utils.metrics.gini import LorenzGiniIndex, ConcentrationCurveGiniIndex, NormalizedConcentrationCurveGiniIndex
from claim_modelling_kedro.pipelines.utils.metrics.sup_diff_cc_and_lc import SupremumDiffBetweenCCAndLC


def get_metric_from_enum(config: Config, enum: MetricEnum, pred_col: str = None) -> SklearnLikeMetric:
    match enum:
        case MetricEnum.MAE:
            return MeanAbsoluteError(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_MAE:
            return MeanAbsoluteError(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_MAE:
            return MeanAbsoluteError(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.RMSE:
            return RootMeanSquaredError(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_RMSE:
            return RootMeanSquaredError(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_RMSE:
            return RootMeanSquaredError(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.R2:
            return R2(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_R2:
            return R2(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_R2:
            return R2(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.MBD:
            return MeanBiasDeviation(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_MBD:
            return MeanBiasDeviation(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_MBD:
            return MeanBiasDeviation(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.POISSON_DEV:
            return MeanPoissonDeviance(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_POISSON_DEV:
            return MeanPoissonDeviance(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_POISSON_DEV:
            return MeanPoissonDeviance(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.GAMMA_DEV:
            return MeanGammaDeviance(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_GAMMA_DEV:
            return MeanGammaDeviance(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_GAMMA_DEV:
            return MeanGammaDeviance(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.SPEARMAN:
            return SpearmanCorrelation(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_SPEARMAN:
            return SpearmanCorrelation(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_SPEARMAN:
            return SpearmanCorrelation(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.ABC:
            return AreaBetweenCCAndLC(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_ABC:
            return AreaBetweenCCAndLC(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_ABC:
            return AreaBetweenCCAndLC(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.SUP_CL_DIFF:
            return SupremumDiffBetweenCCAndLC(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_SUP_CL_DIFF:
            return SupremumDiffBetweenCCAndLC(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_SUP_CL_DIFF:
            return SupremumDiffBetweenCCAndLC(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.CCI:
            return CumulativeCalibrationIndex(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_CCI:
            return CumulativeCalibrationIndex(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_CCI:
            return CumulativeCalibrationIndex(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.COI:
            return CumulativeOverpricingIndex(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_COI:
            return CumulativeOverpricingIndex(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_COI:
            return CumulativeOverpricingIndex(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.CUI:
            return CumulativeUnderpricingIndex(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_CUI:
            return CumulativeUnderpricingIndex(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_CUI:
            return CumulativeUnderpricingIndex(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.LC_GINI:
            return LorenzGiniIndex(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_LC_GINI:
            return LorenzGiniIndex(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_LC_GINI:
            return LorenzGiniIndex(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.CC_GINI:
            return ConcentrationCurveGiniIndex(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_CC_GINI:
            return ConcentrationCurveGiniIndex(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_CC_GINI:
            return ConcentrationCurveGiniIndex(config, pred_col=pred_col, claim_nb_weighted=True)
        case MetricEnum.NORMALIZED_CC_GINI:
            return NormalizedConcentrationCurveGiniIndex(config, pred_col=pred_col)
        case MetricEnum.EXP_WEIGHTED_NORMALIZED_CC_GINI:
            return NormalizedConcentrationCurveGiniIndex(config, pred_col=pred_col, exposure_weighted=True)
        case MetricEnum.CLNB_WEIGHTED_NORMALIZED_CC_GINI:
            return NormalizedConcentrationCurveGiniIndex(config, pred_col=pred_col, claim_nb_weighted=True)
        case TWEEDIE_DEV(p):
            return MeanTweedieDeviance(config, pred_col=pred_col, power=p)
        case EXP_WEIGHTED_TWEEDIE_DEV(p):
            return MeanTweedieDeviance(config, pred_col=pred_col, exposure_weighted=True, power=p)
        case CLNB_WEIGHTED_TWEEDIE_DEV(p):
            return MeanTweedieDeviance(config, pred_col=pred_col, claim_nb_weighted=True, power=p)
        case MAX_ABS_BIAS_BINS(n_bins):
            return MaxAbsBiasBinsMetric(config, n_bins=n_bins)
        case _:
            raise ValueError(f"The metric enum: {enum} is not supported by the Metric class.")
