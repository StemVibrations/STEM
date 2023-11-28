from typing import List, Union
from dataclasses import dataclass, field
from abc import ABC
class WaterBoundaryParametersABC(ABC):
    """
    Class which contains the parameters for a water boundary.
    """


@dataclass
class UniformWaterBoundary(WaterBoundaryParametersABC):
    """
    Class which contains the parameters for a uniform water boundary.
    """
    WATER_PRESSURE: float

