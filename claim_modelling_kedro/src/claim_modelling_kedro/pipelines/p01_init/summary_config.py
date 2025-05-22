import logging
from dataclasses import dataclass
from typing import Dict, Any



logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    lift_chart_params: Dict[str, Any] = None

    def __init__(self, parameters: Dict):
        params = parameters["summary"]
        self.lift_chart_params = params["lift_chart"]
