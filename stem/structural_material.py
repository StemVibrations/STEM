from typing import List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from stem.solver import AnalysisType
from stem.utils import Utils


@dataclass
class StructuralParametersABC(ABC):
    """
    Abstract base class for structural material parameters
    """
    pass

    @staticmethod
    @abstractmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Abstract static method to get the element name for a structural material.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - Exception: abstract method is called
        """
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
        - RAYLEIGH_M (Optional[float]): Mass proportional rayleigh damping parameter [-].
        - RAYLEIGH_K (Optional[float]): Stiffness proportional rayleigh damping parameter [-].
    """
    ndim: int
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    # THICKNESS: float
    DENSITY: float
    # THICKNESS_EFFECTIVE_Y: float
    CROSS_AREA: float
    I33: float

    # Euler beam parameters for 3D
    I22: Optional[float] = None
    TORSIONAL_INERTIA: Optional[float] = None

    RAYLEIGH_M: Optional[float] = None
    RAYLEIGH_K: Optional[float] = None

    def __post_init__(self):
        """
        Check if the second moment of area about the y-axis and the torsional inertia are defined for 3D

        Raises:
            - ValueError: If the second moment of area around the y-axis (I22) or the torsional inertia\
                    (TORSIONAL_INERTIA) is not defined for 3D.
        """
        self.AREA_EFFECTIVE_Y = self.CROSS_AREA
        if self.ndim == 3:
            if self.I22 is None:
                raise ValueError("The second moment of area around the y-axis (I22) is not defined.")
            if self.TORSIONAL_INERTIA is None:
                raise ValueError("The torsional inertia (TORSIONAL_INERTIA) is not defined.")

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for an Euler beam element.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: If the analysis type is not implemented yet for Euler beam elements.

        Returns:
            - Optional[str]: The element name

        """

        available_node_dim_combinations = {
            2: [2, 3],
            3: [2],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Euler beam")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            if n_dim_model == 2:
                element_name = f"LinearTimoshenkoBeamElement2D{n_nodes_element}N"
                # element_name = f"LinearTimoshenkoBeamElement2D{n_nodes_element}N"
                # element_name = f"CrLinearBeamElement2D{n_nodes_element}N"
            else:
                element_name = f"CrLinearBeamElement3D{n_nodes_element}N"

        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented for euler beams.")

        return element_name


@dataclass
class EmbeddedBeam(StructuralParametersABC):
    """
    Class containing the material parameters for an embedded beam, according to Sadek & Shahrour (2004), "A three
    dimensional embedded beam element for reinforced geomaterials". The embedded beam is a linear Timoshenko beam which
    is perfectly bonded to the soil in which it is located. The nodes of the beam do not need to coincide with the
    nodes of the soil mesh: the translations of a beam node are interpolated from the displacements of the soil element
    in which the node is located, whereas the rotations remain independent degrees of freedom of the beam nodes.

    Inheritance:
        - :class:`StructuralParametersABC`

    Attributes:
        - ndim (int): The number of dimensions of the beam formulation (2 or 3)
        - YOUNG_MODULUS (float): The Young's modulus [Pa].
        - POISSON_RATIO (float): The Poisson's ratio [-].
        - DENSITY (float): The density [kg/m3].
        - CROSS_AREA (float): The cross-sectional area [m2].
        - I33 (float): The second moment of area around the local z-axis [m4].
        - I22 (Optional[float]): The second moment of area around the local y-axis [m4]. Required in 3D.
        - TORSIONAL_INERTIA (Optional[float]): The torsional inertia [m4], only used in 3D. If None, I22 + I33 is \
            used.
        - AREA_EFFECTIVE_Y (Optional[float]): The effective shear area in local y-direction [m2]. If None, the \
            cross-sectional area is used.
        - AREA_EFFECTIVE_Z (Optional[float]): The effective shear area in local z-direction [m2], only used in 3D. \
            If None, the cross-sectional area is used.
        - RAYLEIGH_M (Optional[float]): Mass proportional rayleigh damping parameter [-].
        - RAYLEIGH_K (Optional[float]): Stiffness proportional rayleigh damping parameter [-].
    """
    ndim: int
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    DENSITY: float
    CROSS_AREA: float
    I33: float

    I22: Optional[float] = None
    TORSIONAL_INERTIA: Optional[float] = None
    AREA_EFFECTIVE_Y: Optional[float] = None
    AREA_EFFECTIVE_Z: Optional[float] = None

    RAYLEIGH_M: Optional[float] = None
    RAYLEIGH_K: Optional[float] = None

    def __post_init__(self):
        """
        Sets the default effective shear areas and checks if the second moment of area about the y-axis is defined
        for 3D.

        Raises:
            - ValueError: If the second moment of area around the y-axis (I22) is not defined for 3D.
        """
        if self.AREA_EFFECTIVE_Y is None:
            self.AREA_EFFECTIVE_Y = self.CROSS_AREA

        if self.ndim == 3:
            if self.I22 is None:
                raise ValueError("The second moment of area around the y-axis (I22) is not defined.")
            if self.AREA_EFFECTIVE_Z is None:
                self.AREA_EFFECTIVE_Z = self.CROSS_AREA

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for an embedded beam element.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: If the analysis type is not implemented yet for embedded beam elements.

        Returns:
            - Optional[str]: The element name
        """

        available_node_dim_combinations = {
            2: [2],
            3: [2],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Embedded beam")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"GeoEmbeddedBeamElement{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented for embedded beams.")

        return element_name


@dataclass
class EmbeddedBeamInterface(StructuralParametersABC):
    """
    Class containing the material parameters for the interface between a beam and the soil in which it is embedded,
    according to Sadek & Shahrour (2004), "A three dimensional embedded beam element for reinforced geomaterials". The
    interface connects the translations of the beam nodes to the displacement of the soil at the location of these
    nodes. The beam nodes do not need to coincide with the nodes of the soil mesh.

    The bonding force per unit length of the beam is linear elastic, with a Mohr-Coulomb limit on the tangential
    component: \
        |t_t| <= P (c + max(0, -sigma'_n) tan(phi)) \
    where P is the perimeter of the beam (pi * DIAMETER in 3D, two faces per unit width in 2D). If neither the cohesion
    nor the friction angle is defined, the interface is linear elastic.

    Inheritance:
        - :class:`StructuralParametersABC`

    Attributes:
        - INTERFACE_SHEAR_STIFFNESS (float): The tangential stiffness per unit length of the beam [N/m2].
        - INTERFACE_NORMAL_STIFFNESS (float): The normal stiffness per unit length of the beam [N/m2].
        - GEO_COHESION (Optional[float]): The cohesion of the interface [Pa].
        - GEO_FRICTION_ANGLE (Optional[float]): The friction angle of the interface [degrees].
        - DIAMETER (Optional[float]): The diameter of the beam [m], required in 3D when the cohesion or friction \
            angle is defined.
    """
    INTERFACE_SHEAR_STIFFNESS: float
    INTERFACE_NORMAL_STIFFNESS: float
    GEO_COHESION: Optional[float] = None
    GEO_FRICTION_ANGLE: Optional[float] = None
    DIAMETER: Optional[float] = None

    def __post_init__(self):
        """
        Validates the interface parameters.

        Raises:
            - ValueError: If the interface stiffnesses are not positive.
            - ValueError: If the cohesion is negative.
            - ValueError: If the friction angle is not in the range [0, 90) degrees.
            - ValueError: If the diameter is not positive.
        """
        if self.INTERFACE_SHEAR_STIFFNESS <= 0 or self.INTERFACE_NORMAL_STIFFNESS <= 0:
            raise ValueError("The interface shear and normal stiffness of an embedded beam interface must be positive.")
        if self.GEO_COHESION is not None and self.GEO_COHESION < 0:
            raise ValueError("The cohesion of an embedded beam interface must not be negative.")
        if self.GEO_FRICTION_ANGLE is not None and not 0 <= self.GEO_FRICTION_ANGLE < 90:
            raise ValueError("The friction angle of an embedded beam interface must be in the range [0, 90) degrees.")
        if self.DIAMETER is not None and self.DIAMETER <= 0:
            raise ValueError("The diameter of an embedded beam interface must be positive.")

    @property
    def has_shear_strength(self) -> bool:
        """
        Whether the tangential bonding force is limited by a Mohr-Coulomb criterion.

        Returns:
            - bool: True if the cohesion or the friction angle is defined
        """
        return self.GEO_COHESION is not None or self.GEO_FRICTION_ANGLE is not None

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for an embedded beam interface element.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: If the analysis type is not implemented yet for embedded beam interface elements.

        Returns:
            - Optional[str]: The element name
        """

        available_node_dim_combinations = {
            2: [2],
            3: [2],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Embedded beam interface")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"GeoEmbeddedBeamInterfaceElement{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented for embedded beam interfaces.")

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

    _end_coordinates: Optional[
        List[float]] = None  # only used for post-processing, not a parameter to define the material

    @staticmethod
    def get_element_name(n_dim_model, n_nodes_element, analysis_type) -> Optional[str]:
        """
        Static method to get the element name for an elastic spring damper element.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: If the analysis type is not implemented yet for elastic spring damper elements.

        Returns:
            - Optional[str]: The element name
        """

        available_node_dim_combinations = {
            2: [2],
            3: [2],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Elastic spring damper")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"SpringDamperElement{n_dim_model}D"
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented for elastic spring dampers.")

        return element_name


@dataclass
class Anchor(StructuralParametersABC):
    """
    Class containing the material parameters for an anchor element

    Inheritance:
        - :class:`StructuralParametersABC`
        """

    CROSS_AREA: float
    YOUNG_MODULUS: float
    DENSITY: float = 0.0
    PRESTRESS: float = 0.0

    _end_coordinates: Optional[
        List[float]] = None  # only used for post-processing, not a parameter to define the material

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Get the element name for the anchor element

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes of the element (2)
            - analysis_type (AnalysisType): The analysis type of the model

        Raises:
            - ValueError: If the analysis type is not implemented yet for anchor elements.

        Returns:
            - Optional[str]: The element name
        """

        available_node_dim_combinations = {
            2: [2],
            3: [2],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations, "Anchor")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"LinearTrussElement{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented for anchor elements.")

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
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Get the element name for the nodal concentrated element

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes of the element (1)
            - analysis_type (AnalysisType): The analysis type of the model

        Raises:
            - ValueError: If the analysis type is not implemented yet for nodal concentrated elements.

        Returns:
            - Optional[str]: The element name
        """

        available_node_dim_combinations = {
            2: [1],
            3: [1],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Nodal concentrated")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"NodalConcentratedElement{n_dim_model}D1N"
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented for nodal concentrated elements.")

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

    def get_element_name(self, n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Get the element name for the structural material

        Args:
            - n_dim_model (int): The dimension of the model.
            - n_nodes_element (int): The number of nodes per element.
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Returns:
            - Optional[str]: The element name.

        """

        return self.material_parameters.get_element_name(n_dim_model, n_nodes_element, analysis_type)

    def get_property_in_material(self, property_name: str) -> Any:
        """
        Function to retrieve the requested property for the structural material. The function is capital sensitive!

        Args:
            - property_name (str): The desired structural property name.

        Raises:
            - ValueError: If the property is not in not available in the structural material.

        Returns:
            - Any : The value of the structural property

        """

        property_value = self.material_parameters.__dict__.get(property_name)

        if property_value is None:
            raise ValueError(f"Property {property_name} is not one of the parameters of the structural material")

        return property_value
