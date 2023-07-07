from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class AdditionalProcessesParametersABC(ABC):
    """
    Abstract base class to describe the parameters required for additional processes (e.g. excavations)
    """
    pass


@dataclass
class Excavation(AdditionalProcessesParametersABC):
    """
    Class containing the parameters for an excavation process

    Inheritance:
        - :class:`AdditionalProcessesParametersABC`

    Attributes:
        - deactivate_soil_part (bool): Deactivate or not the soil part
    """

    deactivate_soil_part: bool
