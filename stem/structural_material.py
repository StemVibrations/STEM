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
class EulerBeam2D(StructuralParametersABC):
    """
    Class containing the material parameters for beam material

    Attributes:
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].
        DENSITY (float): The density [kg/m3].
        CROSS_AREA (float): The cross-sectional area [m2].
        I33 (float): The second moment of area about the z-axis [m4].
    """

    YOUNG_MODULUS: float
    POISSON_RATIO: float
    DENSITY: float
    CROSS_AREA: float
    I33: float


@dataclass
class EulerBeam3D(StructuralParametersABC):
    """
    Class containing the constitutive parameters for an euler beam

    Attributes:
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].
        DENSITY (float): The density [kg/m3].
        CROSS_AREA (float): The cross-sectional area [m2].
        I22 (float): The second moment of area about the y-axis [m4].
        I33 (float): The second moment of area about the z-axis [m4].
        TORSIONAL_INERTIA (float): The torsional inertia [m4].
    """

    YOUNG_MODULUS: float
    POISSON_RATIO: float
    DENSITY: float
    CROSS_AREA: float
    I22: float
    I33: float
    TORSIONAL_INERTIA: float


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