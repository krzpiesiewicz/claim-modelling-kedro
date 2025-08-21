from abc import ABC, abstractmethod

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum


class Metric(ABC):
    def __init__(self, config: Config, exposure_weighted=False, claim_nb_weighted=False):
        self.config = config
        if exposure_weighted and claim_nb_weighted:
            raise ValueError("Only one of exposure_weighted and claim_nb_weighted can be True.")
        self.exposure_weighted = exposure_weighted
        self.claim_nb_weighted = claim_nb_weighted

    def get_name(self) -> str:
        name = self._get_name()
        if self.exposure_weighted:
            name = "Exposure-Weighted " + name
        if self.claim_nb_weighted:
            name = "Claim-Number-Weighted " + name
        return name

    def get_short_name(self) -> str:
        name = self._get_short_name()
        if self.exposure_weighted:
            name = "ew" + name
        if self.claim_nb_weighted:
            name = "nw" + name
        return name

    @abstractmethod
    def _get_name(self) -> str:
        pass

    @abstractmethod
    def _get_short_name(self) -> str:
        pass

    @abstractmethod
    def get_enum(self) -> MetricEnum:
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        pass
