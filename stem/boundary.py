from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


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
        Helping function to determine whether the boundary should be in the list of
        constraint (True) or in the list of loads (False).
        Returns:
            bool
        """
        pass


@dataclass
class DisplacementConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for displacement constraint

    Attributes:
        active (List[bool]): Activate/deactivate constraint for each direction.
        is_fixed (List[bool]): Specify if constraint is fixed for each direction.
        value (List[float]): Displacement constraint for direction [m].
    """

    active: List[bool]
    is_fixed: List[bool]
    value: List[float]

    @property
    def is_constraint(self) -> bool:
        """
        Determines whether the boundary should be in the list of
        constraint (True) or in the list of loads (False).
        Returns:
            bool
        """
        return True


@dataclass
class RotationConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for rotation constraint

    Attributes:
        active (List[bool]): Activate/deactivate constraint for each direction.
        is_fixed (List[bool]): Specify if constraint is fixed around each axis.
        value (List[float]): Rotation constraint around x, y and axis.
    """

    active: List[bool]
    is_fixed: List[bool]
    value: List[float]

    @property
    def is_constraint(self) -> bool:
        """
        Determines whether the boundary should be in the list of
        constraint (True) or in the list of loads (False).
        Returns:
            bool
        """
        return True


@dataclass
class AbsorbingBoundary(BoundaryParametersABC):
    """
    Class containing the boundary parameters for a point boundary

    Attributes:
        absorbing_factors (List[float]): Indicated how much of the P-wave
            and S-wave should be damped from the boundaries and is comprised between
            0 (no damping) and 1 (full damping).
        virtual_thickness (float): Entity of the virtual thickness [m].
    """

    absorbing_factors: List[float]
    virtual_thickness: float

    @property
    def is_constraint(self) -> bool:
        """
        Determines whether the boundary should be in the list of
        constraint (True) or in the list of loads (False).
        Returns:
            bool
        """
        return False


class Boundary:
    """
    Class containing boundary information acting on a body part

    Attributes:
        part_name (str): name of the boundary
        boundary_parameters (BoundaryParametersABC): class containing boundary parameters
    """

    def __init__(self, part_name: str, boundary_parameters: BoundaryParametersABC):
        """
        Constructor of the boundary class

        Args:
            part_name (str): name of the boundary
            boundary_parameters (BoundaryParametersABC): class containing boundary parameters
        """

        self.part_name: str = part_name
        self.boundary_parameters: BoundaryParametersABC = boundary_parameters
