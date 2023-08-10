from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC
from stem.table import Table

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
        - value (List[Union[float, :class:`stem.table.Table`]]): Entity of the load in the 3 directions [N]. \
            It should be a list of either float or table for each load. If a float is specified, the \
            load is time-independent, otherwise the table specifies the amplitude of the \
            load [N] over time [s] for each direction.
    """

    active: List[bool]
    value: List[Union[float, Table]]


@dataclass
class LineLoad(LoadParametersABC):
    """
    Class containing the load parameters for a line load

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction.
        - value (List[Union[float, :class:`stem.table.Table`]]): Entity of the load in the 3 directions [N/m]. \
            It should be a list of either float or table for each load. If a float is specified, the \
            load is time-independent, otherwise the table specifies the amplitude of the \
            load [N/m] over time [s] for each direction.
    """
    active: List[bool]
    value: List[Union[float, Table]]


@dataclass
class SurfaceLoad(LoadParametersABC):
    """
    Class containing the load parameters for a surface load

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction.
        - value (List[Union[float, :class:`stem.table.Table`]]): Entity of the load in the 3 directions [Pa]. \
            It should be a list of either float or table for each load. If a float is specified, the \
            load is time-independent, otherwise the table specifies the amplitude of the \
            load [Pa] over time [s] for each direction.
    """
    active: List[bool]
    value: Union[List[float], List[Table]]


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

    load: Union[List[float], List[str]]
    direction: List[float]
    velocity: Union[float, str]
    origin: List[float]
    offset: float = 0.0


@dataclass
class GravityLoad(LoadParametersABC):
    """
    Class containing the load parameters for a gravity load.

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction. Input True only in the vertical direction.
        - value (List[float]): Entity of the gravity acceleration in the 3 directions [m/s^2]. Should be -9.81 only in
            the vertical direction
    """
    active: List[bool]
    value: List[float]
