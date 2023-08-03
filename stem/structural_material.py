from typing import List, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from stem.solver import AnalysisType

@dataclass
class StructuralParametersABC(ABC):
    """
    Abstract base class for structural material parameters
    """
    pass


    @staticmethod
    @abstractmethod
    def get_element_name(n_dim_model, n_nodes_element, analysis_type):
        raise Exception("abstract method 'get_element_name' of structural parameters class is called")


@dataclass
class EulerBeam(StructuralParametersABC):
    """
    Class containing the material parameters for beam material

    Inheritance:
        - :class:`StructuralParametersABC`

    Attributes:
        - ndim (int): The number of dimensions of the beam formulation (2 or 3)
        - YOUNG_MODULUS (float): The Young's modulus [Pa].
        - POISSON_RATIO (float): The Poisson's ratio [-].
        - DENSITY (float): The density [kg/m3].
        - CROSS_AREA (float): The cross-sectional area [m2].
        - I33 (float): The second moment of area around the z-axis [m4].
        - I22 (float): The second moment of area around the y-axis [m4].
        - TORSIONAL_INERTIA (float): The torsional inertia [m4].
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

    @staticmethod
    def get_element_name(n_dim_model, n_nodes_element, analysis_type):
        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:

            if n_nodes_element == 2:
                element_name = f"GeoCrBeamElement{n_dim_model}D{n_nodes_element}N"
            else:
                raise ValueError(
                    f"Only 2 node Euler beam elements are supported. {n_nodes_element} nodes were provided."
                )
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented yet for soil material.")

        return element_name


@dataclass
class ElasticSpringDamper(StructuralParametersABC):
    """
    Class containing the constitutive parameters for an elastic spring-damper

    Inheritance:
        - :class:`StructuralParametersABC`

    Attributes:
        - NODAL_DISPLACEMENT_STIFFNESS (List[float]): The stiffness of the spring in x,y,z direction [N/m].
        - NODAL_ROTATIONAL_STIFFNESS (List[float]): The stiffness of the rotational spring around x,y,z axis [Nm/rad].
        - NODAL_DAMPING_COEFFICIENT (List[float]): The damping coefficient of the spring in x,y,z direction [Ns/m].
        - NODAL_ROTATIONAL_DAMPING_COEFFICIENT (List[float]): The damping coefficient of the rotational spring\
              around x,y,z axis [Ns/rad].
    """
    NODAL_DISPLACEMENT_STIFFNESS: List[float]
    NODAL_ROTATIONAL_STIFFNESS: List[float]
    NODAL_DAMPING_COEFFICIENT: List[float]
    NODAL_ROTATIONAL_DAMPING_COEFFICIENT: List[float]

    @staticmethod
    def get_element_name(n_dim_model, n_nodes_element, analysis_type):
        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:

            if n_nodes_element == 2:
                element_name = f"StructuralMechanicsApplication.SpringDamperElement{n_dim_model}D"
            else:
                raise ValueError(
                     f"Only 2 noded elastic spring damper elements are supported. {n_nodes_element} nodes were provided."
                )
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented yet for soil material.")

        return element_name

@dataclass
class NodalConcentrated(StructuralParametersABC):
    """
    Class containing the material parameters for a nodal concentrated element

    Inheritance:
        - :class:`StructuralParametersABC`

    Attributes:
        - NODAL_DISPLACEMENT_STIFFNESS (List[float]): The stiffness of the spring in x,y,z direction [N/m].
        - NODAL_MASS (float): The mass of the concentrated element [kg].
        - NODAL_DAMPING_COEFFICIENT (List[float]): The damping coefficient of the spring in x,y,z direction [Ns/m].
    """
    NODAL_DISPLACEMENT_STIFFNESS: List[float]
    NODAL_MASS: float
    NODAL_DAMPING_COEFFICIENT: List[float]

    @staticmethod
    def get_element_name(n_dim_model, n_nodes_element, analysis_type):

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:

            if n_nodes_element == 1:
                element_name = f"StructuralMechanicsApplication.NodalConcentratedElement{n_dim_model}D1N"
            else:
                raise ValueError(f"Only 1 noded nodal concentrated elements are supported. {n_nodes_element} "
                                 f"nodes were provided.")
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented yet for soil material.")

        return element_name

@dataclass
class StructuralMaterial:
    """
    Class containing material information about a body part, e.g. a soil layer or track components

    Attributes:
        - name (str): The name to describe the structural material.
        - material_parameters (:class:`StructuralParametersABC`): class containing material parameters
    """
    name: str
    material_parameters: StructuralParametersABC

    def get_element_name(self, n_dim_model, n_nodes_element, analysis_type):

        return self.material_parameters.get_element_name(n_dim_model, n_nodes_element, analysis_type)

