from typing import List, Union
from dataclasses import dataclass, field
from abc import ABC



@dataclass
class WaterBoundaryParameters(ABC):
    """
    Abstract base class for load water boundary parameters

    Attributes:
        - surfaces_assigment (List[str]): List of surfaces to which the water boundary is assigned.
        - is_fixed (bool): True if the water boundary is fixed, False otherwise.
        - gravity_direction (int): Direction of the gravity vector.
        - out_of_plane_direction (int): Direction of the out of plane vector.


    """
    surfaces_assigment: List[str] = field(default_factory=lambda: [""])
    is_fixed: bool = True
    gravity_direction: int = 1
    out_of_plane_direction: int = 2


@dataclass
class PhreaticMultiLineBoundary(WaterBoundaryParameters):
    """
    Class containing the load parameters for a phreatic line boundary condition

    Attributes:
        - x_coordinates (List[float]): X coordinates of the phreatic line [m].
        - y_coordinates (List[float]): Y coordinates of the phreatic line [m].
        - z_coordinates (List[float]): Z coordinates of the phreatic line [m].
        - specific_weight (float): Specific weight of the water [kN/m3].


    """
    x_coordinates: List[float] = field(default_factory=lambda: [0.0])
    y_coordinates: List[float] = field(default_factory=lambda: [0.0])
    z_coordinates: List[float] = field(default_factory=lambda: [0.0])
    specific_weight: float = 9.81
    water_pressure: float = 0.0

    def __post_init__(self):
        """
        Post initialization method of the class. It checks that the coordinates are of the same length.

        Returns: None

        """

        # Check that the coordinates are of the same length
        if len(self.x_coordinates) != len(self.y_coordinates):
            raise ValueError("The x and y coordinates must be of the same length")
        # check if coordinate z is defined
        if len(self.z_coordinates) > 1:
            if len(self.x_coordinates) != len(self.z_coordinates):
                raise ValueError("The x/y and z coordinates must be of the same length")
        else:
            # define default z coordinates
            self.z_coordinates = [0.0] * len(self.x_coordinates)


    @property
    def type(self):
        return "Phreatic_Multi_Line"

@dataclass
class InterpolateLineBoundary(WaterBoundaryParameters):
    """
    Class containing the boundary parameters for a interpolate line boundary condition.


    """
    pass

    @property
    def type(self):
        return "Interpolate_Line"


@dataclass
class PhreaticLine(WaterBoundaryParameters):
    """
    Class containing the boundary parameters for phreatic line boundary condition. This condition is should only contain
    two points.

    Attributes:
        - first_reference_coordinate (List[float]): First reference coordinate of the phreatic line [m].
        - second_reference_coordinate (List[float]): Second reference coordinate of the phreatic line [m].
        - specific_weight (float): Specific weight of the water .
        - value (float): Value of the water pressure .


    """
    first_reference_coordinate: List[float] = field(default_factory=lambda: [0.0])
    second_reference_coordinate: List[float] = field(default_factory=lambda: [0.0])
    specific_weight: float = 9.81
    value: float = 0.0

    @property
    def type(self):
        return "Phreatic_Line"


class WaterBoundary:
    """
    Class containing water boundary information acting on a body part

    Attributes:
        - water_boundary (WaterBoundaryParameters): Water boundary parameters
        - type (str): Type of water boundary

    """

    def __init__(self, water_boundary: Union[InterpolateLineBoundary, PhreaticMultiLineBoundary, PhreaticLine], name: str):
        """
        Constructor of the class

        Attributes:
            - water_boundary (WaterBoundaryParameters): Water boundary parameters

        """

        self.water_boundary: Union[InterpolateLineBoundary, PhreaticMultiLineBoundary, PhreaticLine] = water_boundary
        self.type: str = self.water_boundary.type
        self.name: str = name

