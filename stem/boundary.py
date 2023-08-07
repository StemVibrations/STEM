from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from stem.table import Table


@dataclass
class BoundaryParametersABC(ABC):
    """
    Abstract base class for boundary parameters
    """

    pass

    @property
    @abstractmethod
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint.
        
        Returns:
            - bool
        """
        raise Exception("abstract method 'is_constraint' of boundary parameters class is called")


@dataclass
class DisplacementConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for displacement constraint

    Inheritance:
        - :class:`BoundaryParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate constraint for each direction.
        - is_fixed (List[bool]): Specify if constraint is fixed for each direction.
        - value (Union[List[float], List[:class:`stem.table.Table`]]): Displacement value for direction [m]. \
            It can be either a list of floats, if the load is time-independent, or a list of table which specify the \
            amplitude of the displacement [m] over time [s] for each direction.
    """

    active: List[bool]
    is_fixed: List[bool]
    value: Union[List[float], List[Table]]

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. True for DisplacementConstraint.
        
        Returns:
            - bool
        """
        return True


@dataclass
class RotationConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for rotation constraint

    Inheritance:
        - :class:`BoundaryParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate constraint for each direction.
        - is_fixed (List[bool]): Specify if constraint is fixed around each axis.
        - value (List[float]): Rotation constraint
        - value (Union[List[float], List[:class:`stem.table.Table`]]): Rotation value around x, y and z axis [Rad]. \
            It can be either a list of floats, if the load is time-independent, or a list of table which specify the \
            amplitude of the rotation [Rad] over time [s] around each axis.
    """

    active: List[bool]
    is_fixed: List[bool]
    value: Union[List[float], List[Table]]

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. True for RotationConstraint.
        
        Returns:
            - bool
        """
        return True


@dataclass
class AbsorbingBoundary(BoundaryParametersABC):
    """
    Class containing the boundary parameters for a point boundary

    Inheritance:
        - :class:`BoundaryParametersABC`

    Attributes:
        - absorbing_factors (List[float]): Indicated how much of the P-wave \
            and S-wave should be damped from the boundaries and is comprised between \
            0 (no damping) and 1 (full damping).
        - virtual_thickness (float): Entity of the virtual thickness [m].
    """

    absorbing_factors: List[float]
    virtual_thickness: float

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. False for AbsorbingBoundary.
        
        Returns:
            - bool
        """
        return False
