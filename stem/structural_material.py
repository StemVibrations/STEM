from typing import List, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC

@dataclass
class StructuralParametersABC(ABC):
    """
    Abstract base class for structural material parameters
    """
    pass


@dataclass
class EulerBeam(StructuralParametersABC):
    """
    Class containing the material parameters for beam material

    Attributes:
        ndim (int): The number of dimensions of the beam formulation (2 or 3)
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].
        DENSITY (float): The density [kg/m3].
        CROSS_AREA (float): The cross-sectional area [m2].
        I33 (float): The second moment of area around the z-axis [m4].
        I22 (float): The second moment of area around the y-axis [m4].
        TORSIONAL_INERTIA (float): The torsional inertia [m4].
    """

    ndim: int
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    DENSITY: float
    CROSS_AREA: float
    I33: float

    # Euler beam parameters for 3D
    I22: Optional[float] = None
    TORSIONAL_INERTIA: Optional[float] = None

    def __post_init__(self):
        """
        Check if the second moment of area about the y-axis and the torsional inertia are defined for 3D

        Returns:

        """
        if self.ndim == 3:
            if self.I22 is None:
                raise ValueError("The second moment of area around the y-axis (I22) is not defined.")
            if self.TORSIONAL_INERTIA is None:
                raise ValueError("The torsional inertia (TORSIONAL_INERTIA) is not defined.")


@dataclass
class ElasticSpringDamper(StructuralParametersABC):
    """
    Class containing the constitutive parameters for an elastic spring-damper

    Attributes:
        NODAL_DISPLACEMENT_STIFFNESS (List[float]): The stiffness of the spring in x,y,z direction [N/m].
        NODAL_ROTATIONAL_STIFFNESS (List[float]): The stiffness of the rotational spring around x,y,z axis [Nm/rad].
        NODAL_DAMPING_COEFFICIENT (List[float]): The damping coefficient of the spring in x,y,z direction [Ns/m].
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT (List[float]): The damping coefficient of the rotational spring
            around x,y,z axis [Ns/rad].
    """
    NODAL_DISPLACEMENT_STIFFNESS: List[float]
    NODAL_ROTATIONAL_STIFFNESS: List[float]
    NODAL_DAMPING_COEFFICIENT: List[float]
    NODAL_ROTATIONAL_DAMPING_COEFFICIENT: List[float]


@dataclass
class NodalConcentrated(StructuralParametersABC):
    """
    Class containing the material parameters for a nodal concentrated element

    Attributes:
        NODAL_DISPLACEMENT_STIFFNESS (List[float]): The stiffness of the spring in x,y,z direction [N/m].
        NODAL_MASS (float): The mass of the concentrated element [kg].
        NODAL_DAMPING_COEFFICIENT (List[float]): The damping coefficient of the spring in x,y,z direction [Ns/m].
    """
    NODAL_DISPLACEMENT_STIFFNESS: List[float]
    NODAL_MASS: float
    NODAL_DAMPING_COEFFICIENT: List[float]

@dataclass
class StructuralMaterial:
    """
    Class containing material information about a body part, e.g. a soil layer or track components

    Attributes:
        name (str): name of the material
        material_parameters (StructuralParametersABC): class containing material parameters

    """

    name: str
    material_parameters: StructuralParametersABC
