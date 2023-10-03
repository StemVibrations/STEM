from dataclasses import dataclass
from abc import ABC


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
        - deactivate_body_model_part (bool): Deactivate or not the body model part
    """

    deactivate_body_model_part: bool