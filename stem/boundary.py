from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class BoundaryParametersABC(ABC):
    """
    Abstract base class for boundary parameters
    """
    pass

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
        is_fixed (List[bool]): Specify if constraint is fixed.
        value (List[float]):
    """

    active: List[bool] = field(default_factory=lambda: [True, True, True])
    is_fixed: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def is_constraint(self) -> bool:
        return True


@dataclass
class RotationConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for rotation constraint

    Attributes:
        active (List[bool]): Activate/deactivate constraint for each direction.
        is_fixed (List[bool]): Specify if constraint is fixed.
        value (List[float]):
    """

    active: List[bool] = field(default_factory=lambda: [True, True, True])
    is_fixed: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def is_constraint(self) -> bool:
        return True

@dataclass
class AbsorbingBoundary(BoundaryParametersABC):
    """
    Class containing the boundary parameters for a point boundary

    Attributes:
        absorbing_factors (List[float]): Activate/deactivate boundary for each direction.
        virtual_thickness (float): Entity of the boundary in the 3 directions [N].
    """
    absorbing_factors: List[float] = field(default_factory=lambda: [1.0, 1.0])
    virtual_thickness: float = 1

    def is_constraint(self) -> bool:
        return False


class Boundary:
    """
    Class containing boundary information acting on a body part

    Attributes:
        name (str): name of the boundary
        boundary_parameters (BoundaryParametersABC): class containing boundary parameters
    """

    def __init__(self, name: str, boundary_parameters: BoundaryParametersABC):
        """
        Constructor of the boundary class

        Args:
            name (str): name of the boundary
            boundary_parameters (BoundaryParametersABC): class containing boundary parameters
        """

        self.name: str = name
        self.boundary_parameters: BoundaryParametersABC = boundary_parameters
