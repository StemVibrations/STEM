from typing import List, Dict, Any, Union, Optional
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

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction.
        - value (List[float]): Entity of the load in the 3 directions [N].
    """

    active: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class LineLoad(LoadParametersABC):
    """
    Class containing the load parameters for a line load

    Attributes:
        active (List[bool]): Activate/deactivate load for each direction.
        value (List[float]): Entity of the load in the 3 directions [N].
    """
    active: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class SurfaceLoad(LoadParametersABC):
    """
    Class containing the load parameters for a surface load

    Attributes:
        active (List[bool]): Activate/deactivate load for each direction.
        value (List[float]): Entity of the load in the 3 directions [N].
    """
    active: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MovingLoad(LoadParametersABC):
    """
    Class containing the load parameters for a moving load.

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - load (Union[List[float], List[str]]): Entity of the load [N] in the 3 \
               directions. Can be defined as strings (when function of time) or as float. \
               Mixed types are not accepted.
        - direction (List[int]):  Direction of the moving load (-1 or +1 in x, y, z direction) [-].
        - velocity (Union[float, str]): Velocity of the moving load [m/s].
        - origin (List[float]): Starting coordinates of the moving load [m].
        - offset (float): Offset of the moving load [m].
    """

    load: Union[List[float], List[str]] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    direction: List[float] = field(default_factory=lambda: [1, 1, 1])
    velocity: Union[float, str] = 0.0
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    offset: float = 0.0


class Load:
    """
    Class containing load information acting on a body part

    Attributes:
        - part_name (str): name of the load
        - load_parameters (:class:`LoadParametersABC`): class containing load parameters
    """

    def __init__(self, part_name: str, load_parameters: LoadParametersABC):
        """
        Constructor of the load class

        Args:
            - part_name (str): name of the load
            - load_parameters (:class:`LoadParametersABC`): class containing load parameters
        """

        self.part_name: str = part_name
        self.load_parameters: LoadParametersABC = load_parameters
