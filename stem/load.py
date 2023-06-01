from typing import List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC


@dataclass
class LoadParametersABC(ABC):
    """
    Abstract base class for load parameters
    """
    pass


@dataclass
class PointLoad(LoadParametersABC):
    """
    Class containing the load parameters for a point load
    :Attributes:
        active (List[bool]): Activate/deactivate load for each direction.
        value (List[float]): Entity of the load in the 3 directions [N].
    """

    active: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class Load:
    """
    Class containing load information acting on a body part
    Attributes:
        name (str): name of the load
        load_parameters (LoadParametersABC): class containing load parameters
    """

    def __init__(self, name: str, load_parameters: LoadParametersABC):
        """
        Constructor of the load class
        Args:
            name (str): name of the load
            load_parameters (LoadParametersABC): class containing load parameters
        """

        self.name: str = name
        self.load_parameters: LoadParametersABC = load_parameters
