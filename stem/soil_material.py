from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field
from abc import ABC

from stem.globals import FluidProperties
from stem.solver import AnalysisType
from stem.utils import Utils


@dataclass
class SoilFormulationParametersABC(ABC):
    """
    Abstract base class for soil formulation parameters

    Attributes:
        - ndim (int): The number of dimensions of the soil formulation (2 or 3)
    """
    ndim: int


@dataclass
class SoilConstitutiveLawABC(ABC):
    """
    Abstract base class for soil constitutive laws
    """
    pass


@dataclass
class RetentionLawABC(ABC):
    """
    Abstract class containing the parameters for a retention law. This class is created for type checking purposes.
    """
    pass


# @dataclass
# class FluidProperties:
#     """
#     Class containing the parameters for a fluid. Default values are for water at 12 degrees Celsius.
#
#     Attributes:
#         - DENSITY_FLUID (float): The density of fluid [kg/m^3].
#         - DYNAMIC_VISCOSITY (float): The dynamic viscosity of fluid [Pa s].
#         - BULK_MODULUS_FLUID (float): The bulk modulus of fluid [Pa].
#     """
#     DENSITY_FLUID: float = 1000
#     DYNAMIC_VISCOSITY: float = 1.3e-3
#     BULK_MODULUS_FLUID: float = 2e9


@dataclass
class OnePhaseSoil(SoilFormulationParametersABC):
    """
    Class containing the material parameters for an undrained soil material

    Inheritance:
        - :class:`SoilFormulationParametersABC`

    Attributes:
        - IS_UNDRAINED (bool): Boolean indicating if the soil is undrained.
        - DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        - POROSITY (float): The porosity of the soil [-].
        - BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        - BIOT_COEFFICIENT (Optional[float]): The Biot coefficient [-].
        - RAYLEIGH_M (Optional[float]): Mass proportional Rayleigh damping parameter [-].
        - RAYLEIGH_K (Optional[float]): Stiffness proportional Rayleigh damping parameter [-].
    """
    IS_DRAINED: bool
    DENSITY_SOLID: float
    POROSITY: float
    BULK_MODULUS_SOLID: float = 50e9
    BIOT_COEFFICIENT: Optional[float] = None

    RAYLEIGH_M: Optional[float] = None
    RAYLEIGH_K: Optional[float] = None


@dataclass
class TwoPhaseSoil(SoilFormulationParametersABC):
    """
    Class containing the material parameters for a two phase soil material

    Inheritance:
        - :class:`SoilFormulationParametersABC`

    Attributes:
        - DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        - POROSITY (float): The porosity of the soil [-].
        - PERMEABILITY_XX (float): The permeability in the x-direction [m^2].
        - PERMEABILITY_YY (float): The permeability in the y-direction [m^2].
        - PERMEABILITY_XY (float): The permeability in the xy-direction [m^2].
        - BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        - BIOT_COEFFICIENT (Optional[float]): The Biot coefficient [-].
        - RAYLEIGH_M (Optional[float]): Mass proportional rayleigh damping parameter [-].
        - RAYLEIGH_K (Optional[float]): Stiffness proportional rayleigh damping parameter [-].
        - PERMEABILITY_YZ (Optional[float]): The permeability in the yz-direction [m^2].
        - PERMEABILITY_ZX (Optional[float]): The permeability in the zx-direction [m^2].
        - PERMEABILITY_ZZ (Optional[float]): The permeability in the z-direction [m^2].
    """
    DENSITY_SOLID: float
    POROSITY: float
    PERMEABILITY_XX: float
    PERMEABILITY_YY: float
    PERMEABILITY_XY: float = 0
    BULK_MODULUS_SOLID: float = 50e9
    BIOT_COEFFICIENT: Optional[float] = None
    RAYLEIGH_M: Optional[float] = None
    RAYLEIGH_K: Optional[float] = None

    # parameters for 3D
    PERMEABILITY_YZ: Optional[float] = 0
    PERMEABILITY_ZX: Optional[float] = 0

    PERMEABILITY_ZZ: Optional[float] = None

    def __post_init__(self):
        """
        Check if the permeability parameters are defined for the correct number of dimensions

        Raises:
            - ValueError: If the permeability parameters are not defined for the correct number of dimensions
        """
        if self.ndim == 2:
            # set permeability_yz and permeability_zx to None as they are not used in 2D
            self.PERMEABILITY_YZ = None
            self.PERMEABILITY_ZX = None

        elif self.ndim == 3:
            # Check if the permeability parameters are defined for 3D
            if self.PERMEABILITY_ZZ is None:
                raise ValueError("The permeability in the z-direction (PERMEABILITY_ZZ) is not defined.")
            if self.PERMEABILITY_YZ is None:
                raise ValueError("The permeability in the yz-direction (PERMEABILITY_YZ) is not defined.")
            if self.PERMEABILITY_ZX is None:
                raise ValueError("The permeability in the zx-direction (PERMEABILITY_ZX) is not defined.")


@dataclass
class LinearElasticSoil(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for a 2D linear elastic material

    Inheritance:
        - :class:`SoilConstitutiveLawABC`

    Attributes:
        - YOUNG_MODULUS (float): The Young's modulus [Pa].
        - POISSON_RATIO (float): The Poisson's ratio [-].
        - K0_MAIN_DIRECTION (Optional[int]): The direction of the vertical stress in the K0 procedure \
            (0 = x, 1 = y, 2 = z) [-]. Only required for the K0 procedure.
        - K0_NC (Optional[float]): The ratio of the horizontal to the vertical effective stress in the K0 procedure \
            [-]. Only required for the K0 procedure.
    """
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    K0_MAIN_DIRECTION: Optional[int] = None
    K0_NC: Optional[float] = None


@dataclass
class MohrCoulombLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for a 2D Mohr-Coulomb material

    Inheritance:
        - :class:`SoilConstitutiveLawABC`
        """
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    GEO_COHESION: float
    GEO_FRICTION_ANGLE: float
    GEO_TENSILE_STRENGTH: float
    GEO_DILATANCY_ANGLE: float


@dataclass
class SmallStrainUmatLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for a 2D small strain umat material

    Inheritance:
        - :class:`SoilConstitutiveLawABC`

    Attributes:
        - UMAT_NAME (str): The name and location of the umat .dll or .so file.
        - IS_FORTRAN_UMAT (bool): A boolean to indicate whether the umat is written in Fortran.
        - UMAT_PARAMETERS (list): The parameters of the umat.
        - STATE_VARIABLES (list): The state variables of the umat.
    """
    UMAT_NAME: str
    IS_FORTRAN_UMAT: bool
    UMAT_PARAMETERS: List[Any]
    STATE_VARIABLES: List[Any]


@dataclass
class HardeningSoilLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for the Hardening Soil model (C UMAT from PlatypusBytes
    ConstitutiveModels). Stress parameters must be given in the same unit as the model (Pa).

    Inheritance:
        - :class:`SoilConstitutiveLawABC`

    Attributes:
        - UMAT_NAME (str): The name and location of the hardening soil umat .dll or .so file.
        - E50_REF (float): Reference secant stiffness at 50% strength [Pa].
        - EUR_REF (float): Reference unloading/reloading stiffness [Pa].
        - POWER_M (float): Stress exponent for stiffness [-].
        - COHESION (float): Cohesion [Pa].
        - FRICTION_ANGLE (float): Friction angle [deg].
        - DILATANCY_ANGLE (float): Dilatancy angle [deg].
        - P_REF (float): Reference pressure for stiffness [Pa].
        - FAILURE_RATIO (float): Failure ratio qf/qa, in (0, 1) [-].
        - POISSON_RATIO (float): Unloading/reloading Poisson's ratio [-].
        - M_CAP (float): Cap aspect ratio, 0 disables the cap [-].
        - K_RATIO (float): Ks/Kc, swelling over elasto-plastic bulk modulus in isotropic compression (> 1) [-].
        - INITIAL_VOID_RATIO (Optional[float]): Initial void ratio [-].
        - CRITICAL_VOID_RATIO (Optional[float]): Critical void ratio of the dilatancy cut-off, 0 disables it [-].
        - INITIAL_PLASTIC_SHEAR_STRAIN (float): Initial plastic shear strain state variable [-].
        - INITIAL_PRECONSOLIDATION_STRESS (float): Initial pre-consolidation stress state variable; 0 lets the \
            model initialise a normally consolidated state [Pa].
    """
    UMAT_NAME: str
    E50_REF: float
    EUR_REF: float
    POWER_M: float
    COHESION: float
    FRICTION_ANGLE: float
    DILATANCY_ANGLE: float
    P_REF: float
    FAILURE_RATIO: float
    POISSON_RATIO: float
    M_CAP: float
    K_RATIO: float
    INITIAL_VOID_RATIO: Optional[float] = None
    CRITICAL_VOID_RATIO: Optional[float] = None
    INITIAL_PLASTIC_SHEAR_STRAIN: float = 0.0
    INITIAL_PRECONSOLIDATION_STRESS: float = 0.0

    def __post_init__(self):
        """
        Checks the validity of the material parameters.

        Raises:
            - ValueError: If a parameter is outside its valid range.
        """
        if self.E50_REF <= 0 or self.EUR_REF <= 0:
            raise ValueError("E50_REF and EUR_REF must be positive.")
        if self.COHESION < 0:
            raise ValueError("COHESION must be non-negative.")
        if not 0 < self.FRICTION_ANGLE < 90:
            raise ValueError("FRICTION_ANGLE must be in (0, 90) degrees.")
        if self.DILATANCY_ANGLE > self.FRICTION_ANGLE:
            raise ValueError("DILATANCY_ANGLE must not exceed FRICTION_ANGLE.")
        if self.P_REF <= 0:
            raise ValueError("P_REF must be positive.")
        if not 0 < self.FAILURE_RATIO < 1:
            raise ValueError("FAILURE_RATIO must be in (0, 1).")
        if self.EUR_REF <= 2 * self.E50_REF / (2 - self.FAILURE_RATIO):
            raise ValueError("EUR_REF must be larger than 2 * E50_REF / (2 - FAILURE_RATIO).")
        if not 0 <= self.POISSON_RATIO < 0.5:
            raise ValueError("POISSON_RATIO must be in [0, 0.5).")
        if self.M_CAP < 0:
            raise ValueError("M_CAP must be non-negative (0 disables the cap).")
        if self.M_CAP > 0 and self.K_RATIO <= 1:
            raise ValueError("K_RATIO must be larger than 1.")
        if self.CRITICAL_VOID_RATIO is not None and self.CRITICAL_VOID_RATIO > 0 and \
                (self.INITIAL_VOID_RATIO is None or self.INITIAL_VOID_RATIO <= 0):
            raise ValueError("INITIAL_VOID_RATIO must be positive when the dilatancy cut-off is used.")

    def get_umat_parameters(self) -> List[float]:
        """
        Returns the UMAT PROPS array in the order expected by the hardening soil UMAT.

        Returns:
            - List[float]: UMAT parameters.
        """
        parameters = [
            self.E50_REF, self.EUR_REF, self.POWER_M, self.COHESION, self.FRICTION_ANGLE, self.DILATANCY_ANGLE,
            self.P_REF, self.FAILURE_RATIO, self.POISSON_RATIO, self.M_CAP, self.K_RATIO
        ]
        if self.INITIAL_VOID_RATIO is not None or self.CRITICAL_VOID_RATIO is not None:
            parameters.append(self.INITIAL_VOID_RATIO if self.INITIAL_VOID_RATIO is not None else 0.0)
        if self.CRITICAL_VOID_RATIO is not None:
            parameters.append(self.CRITICAL_VOID_RATIO)
        return parameters

    def get_state_variables(self) -> List[float]:
        """
        Returns the initial UMAT STATEV array: [plastic shear strain, pre-consolidation stress, at failure].

        Returns:
            - List[float]: initial state variables.
        """
        return [self.INITIAL_PLASTIC_SHEAR_STRAIN, self.INITIAL_PRECONSOLIDATION_STRESS, 0.0]


@dataclass
class SoftSoilCreepLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for the Soft Soil Creep model (C UMAT from PlatypusBytes
    ConstitutiveModels).

    Inheritance:
        - :class:`SoilConstitutiveLawABC`

    Attributes:
        - UMAT_NAME (str): The name and location of the soft soil creep umat .dll or .so file.
        - MODIFIED_SWELLING_INDEX (float): kappa* [-].
        - MODIFIED_COMPRESSION_INDEX (float): lambda* [-] (must be larger than kappa*).
        - MODIFIED_CREEP_INDEX (float): mu* [-].
        - POISSON_RATIO (float): Poisson's ratio [-].
        - REFERENCE_TIME (float): Reference time tau, in the time unit of the analysis [s].
        - CRITICAL_STATE_LINE_SLOPE (float): Slope of the critical state line M [-].
        - FRICTION_ANGLE (float): Mohr-Coulomb friction angle, 0 disables the failure surface [deg].
        - OCR (float): Initial isotropic over-consolidation ratio [-].
        - MINIMUM_PRESSURE (Optional[float]): Lower limit of the mean stress for the stiffness [Pa].
        - INITIAL_CREEP_STRAIN (float): Initial volumetric creep strain state variable [-].
        - INITIAL_PRECONSOLIDATION_PRESSURE (float): Initial isotropic pre-consolidation pressure; 0 lets the \
            model initialise it as OCR * p_eq [Pa].
    """
    UMAT_NAME: str
    MODIFIED_SWELLING_INDEX: float
    MODIFIED_COMPRESSION_INDEX: float
    MODIFIED_CREEP_INDEX: float
    POISSON_RATIO: float
    REFERENCE_TIME: float
    CRITICAL_STATE_LINE_SLOPE: float
    FRICTION_ANGLE: float
    OCR: float = 1.0
    MINIMUM_PRESSURE: Optional[float] = None
    INITIAL_CREEP_STRAIN: float = 0.0
    INITIAL_PRECONSOLIDATION_PRESSURE: float = 0.0

    def __post_init__(self):
        """
        Checks the validity of the material parameters.

        Raises:
            - ValueError: If a parameter is outside its valid range.
        """
        if self.MODIFIED_SWELLING_INDEX <= 0:
            raise ValueError("MODIFIED_SWELLING_INDEX must be positive.")
        if self.MODIFIED_COMPRESSION_INDEX <= self.MODIFIED_SWELLING_INDEX:
            raise ValueError("MODIFIED_COMPRESSION_INDEX must be larger than MODIFIED_SWELLING_INDEX.")
        if self.MODIFIED_CREEP_INDEX <= 0:
            raise ValueError("MODIFIED_CREEP_INDEX must be positive.")
        if not 0 <= self.POISSON_RATIO < 0.5:
            raise ValueError("POISSON_RATIO must be in [0, 0.5).")
        if self.REFERENCE_TIME <= 0:
            raise ValueError("REFERENCE_TIME must be positive.")
        if self.CRITICAL_STATE_LINE_SLOPE <= 0:
            raise ValueError("CRITICAL_STATE_LINE_SLOPE must be positive.")
        if not 0 <= self.FRICTION_ANGLE < 90:
            raise ValueError("FRICTION_ANGLE must be in [0, 90) degrees.")
        if self.OCR <= 0:
            raise ValueError("OCR must be positive.")
        if self.MINIMUM_PRESSURE is not None and self.MINIMUM_PRESSURE <= 0:
            raise ValueError("MINIMUM_PRESSURE must be positive.")

    def get_umat_parameters(self) -> List[float]:
        """
        Returns the UMAT PROPS array in the order expected by the soft soil creep UMAT.

        Returns:
            - List[float]: UMAT parameters.
        """
        parameters = [
            self.MODIFIED_SWELLING_INDEX, self.MODIFIED_COMPRESSION_INDEX - self.MODIFIED_SWELLING_INDEX,
            self.MODIFIED_CREEP_INDEX, self.POISSON_RATIO, self.REFERENCE_TIME, self.CRITICAL_STATE_LINE_SLOPE,
            self.FRICTION_ANGLE, self.OCR
        ]
        if self.MINIMUM_PRESSURE is not None:
            parameters.append(self.MINIMUM_PRESSURE)
        return parameters

    def get_state_variables(self) -> List[float]:
        """
        Returns the initial UMAT STATEV array: [creep strain, pre-consolidation pressure, p_eq, OCR, at failure].

        Returns:
            - List[float]: initial state variables.
        """
        return [self.INITIAL_CREEP_STRAIN, self.INITIAL_PRECONSOLIDATION_PRESSURE, 0.0, 0.0, 0.0]


@dataclass
class SmallStrainUdsmLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for small strain udsm material

    Inheritance:
        - :class:`SoilConstitutiveLawABC`

    Attributes:
        - UDSM_NAME (str): The name and location of the udsm .dll or .so file.
        - UDSM_NUMBER (int): The model number within the udsm.
        - IS_FORTRAN_UDSM (bool): A boolean to indicate whether the udsm is written in Fortran.
        - UDSM_PARAMETERS (list): The parameters of the udsm.
    """
    UDSM_NAME: str
    UDSM_NUMBER: int
    IS_FORTRAN_UDSM: bool
    UDSM_PARAMETERS: List[Any]


@dataclass
class SaturatedBelowPhreaticLevelLaw(RetentionLawABC):
    """
    Class containing the parameters for the retention law: saturated below phreatic level

    Inheritance:
        - :class:`RetentionLawABC`

    Attributes:
        - SATURATED_SATURATION (float): The saturation ratio below phreatic level [-].
        - RESIDUAL_SATURATION (float): The residual saturation ratio [-].
    """
    SATURATED_SATURATION: float = 1.0
    RESIDUAL_SATURATION: float = 1e-10


@dataclass
class SaturatedLaw(RetentionLawABC):
    """
    Class containing the parameters for the retention law: saturated

    Inheritance:
        - :class:`RetentionLawABC`

    Attributes:
        - SATURATED_SATURATION (float): The saturation ratio [-].
    """
    SATURATED_SATURATION: float = 1.0


@dataclass
class VanGenuchtenLaw(RetentionLawABC):
    """
    Class containing the parameters for a retention law

    Inheritance:
        - :class:`RetentionLawABC`

    Attributes:
        - VAN_GENUCHTEN_AIR_ENTRY_PRESSURE (float): The air entry pressure [Pa].
        - VAN_GENUCHTEN_GN (float): The pore size distribution index [-].
        - VAN_GENUCHTEN_GL (float): exponent for calculating relative permeability [-].
        - SATURATED_SATURATION (float): The maximum saturation ratio [-].
        - RESIDUAL_SATURATION (float): The minumum saturation ratio [-].
        - MINIMUM_RELATIVE_PERMEABILITY (float): The minimum relative permeability [-].
    """
    VAN_GENUCHTEN_AIR_ENTRY_PRESSURE: float
    VAN_GENUCHTEN_GN: float
    VAN_GENUCHTEN_GL: float
    SATURATED_SATURATION: float = 1.0
    RESIDUAL_SATURATION: float = 1e-10
    MINIMUM_RELATIVE_PERMEABILITY: float = 0.0001


@dataclass
class SoilMaterial:
    """
    Class containing the parameters for a soil material

    Attributes:
        - name (str): The name to describe the soil material.
        - soil_formulation (:class:`SoilFormulationParametersABC`): The soil formulation parameters.
        - constitutive_law (:class:`SoilConstitutiveLawABC`): The soil constitutive law parameters.
        - retention_parameters (:class:`RetentionLawABC`): The retention law parameters.
        - fluid_properties (:class:`FluidProperties`): The fluid properties.
    """
    name: str
    soil_formulation: SoilFormulationParametersABC
    constitutive_law: SoilConstitutiveLawABC
    retention_parameters: RetentionLawABC
    fluid_properties: FluidProperties = field(default_factory=FluidProperties)

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> str:
        """
        Function to get the element name based on the number of dimensions, the number of nodes and the analysis type.

        Args:
            - n_dim_model (int): The number of dimensions of the model.
            - n_nodes_element (int): The number of nodes per element.
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: If the analysis type is not implemented yet for nodal concentrated elements.

        Returns:
            - element_name (str): The name of the element.

        """

        available_node_dim_combinations = {
            2: [3, 4, 6, 8],
            3: [4, 8, 10, 20],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations, "Soil")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            # for second order elements, displacements are calculated on second order elements and pore pressure on
            # first order elements
            if (n_dim_model == 2 and n_nodes_element > 4) or (n_dim_model == 3 and n_nodes_element > 8):
                element_name = f"SmallStrainUPwDiffOrderElement{n_dim_model}D{n_nodes_element}N"
            else:
                element_name = f"UPwSmallStrainElement{n_dim_model}D{n_nodes_element}N"

        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented yet for soil material.")

        return element_name

    def to_dict(self) -> Dict[str, Any]:
        """
        Function to convert the soil material to a dictionary. For serialization purposes.

        Returns:
            - Dict[str, Any]: The soil material as a dictionary.
        """

        # add class name to each sub-dictionary
        self.soil_formulation.__dict__.update({"type": self.soil_formulation.__class__.__name__})
        self.constitutive_law.__dict__.update({"type": self.constitutive_law.__class__.__name__})
        self.retention_parameters.__dict__.update({"type": self.retention_parameters.__class__.__name__})
        self.fluid_properties.__dict__.update({"type": self.fluid_properties.__class__.__name__})

        return {
            "name": self.name,
            "soil_formulation": self.soil_formulation.__dict__,
            "constitutive_law": self.constitutive_law.__dict__,
            "retention_parameters": self.retention_parameters.__dict__,
            "fluid_properties": self.fluid_properties.__dict__,
        }

    def get_property_in_material(self, property_name: str) -> Any:
        """
        Function to retrieve the requested property for the soil material. The function is capital sensitive!

        Args:
            - property_name (str): The desired soil property name.

        Raises:
            - ValueError: If the property is not in not available in the soil material.

        Returns:
            - Any : The value of the soil property.

        """

        all_properties = {}

        all_properties.update(self.soil_formulation.__dict__)
        all_properties.update(self.constitutive_law.__dict__)
        all_properties.update(self.retention_parameters.__dict__)
        all_properties.update(self.fluid_properties.__dict__)

        property_value = all_properties.get(property_name)

        if property_value is None:
            raise ValueError(f"Property {property_name} is not one of the parameters of the soil material")

        return property_value


@dataclass
class InterfaceMaterial:
    """
    Class containing the parameters for an interface material

    Attributes:
        - name (str): The name to describe the interface material.
        - constitutive_law (:class:`SoilConstitutiveLawABC`): The constitutive law parameters for
        the interface material.
        - soil_formulation (:class:`SoilFormulationParametersABC`): The soil formulation parameters
        for the interface material.
        - retention_parameters (:class:`RetentionLawABC`): The retention parameters for the interface material.
        - fluid_properties (:class:`FluidProperties`): The fluid properties for the interface material.
        - consider_gap_closure (bool): If True, a gap that opened when the tensile strength of the interface was
        reached must close before compression can develop again (as the Plaxis option "Consider gap closure").
        Only supported by the Mohr-Coulomb interface law.
    """
    name: str
    constitutive_law: SoilConstitutiveLawABC
    soil_formulation: SoilFormulationParametersABC
    retention_parameters: RetentionLawABC
    fluid_properties: FluidProperties = field(default_factory=FluidProperties)
    consider_gap_closure: bool = False

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> str:
        """
        Function to get the element name based on the number of dimensions, the number of nodes and the analysis type.

        Args:
            - n_dim_model (int): The number of dimensions of the model.
            - n_nodes_element (int): The number of nodes per element.
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: If the analysis type is not implemented yet for nodal concentrated elements.

        Returns:
            - element_name (str): The name of the element.

        """

        available_node_dim_combinations = {
            2: [4, 6],
            3: [6, 8, 12, 16],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations, "Soil")
        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            n_nodes_name = n_nodes_element // 2
            if n_dim_model == 2:
                element_name = f"Geo_ULineInterfacePlaneStrainElement{n_nodes_name}Plus{n_nodes_name}N"
            else:
                element_name = f"Geo_USurfaceInterfaceElement{n_nodes_name}Plus{n_nodes_name}N"
        else:
            raise ValueError(f"Analysis type {analysis_type} is not implemented yet for interface material.")

        return element_name

    def get_property_in_material(self, property_name: str) -> Any:
        """
        Function to retrieve the requested property for the interface material. The function is capital sensitive!

        Args:
            - property_name (str): The desired interface property name.

        Raises:
            - ValueError: If the property is not in not available in the interface material.

        Returns:
            - Any : The value of the interface property.

        """

        all_properties = {}

        all_properties.update(self.soil_formulation.__dict__)
        all_properties.update(self.constitutive_law.__dict__)
        all_properties.update(self.retention_parameters.__dict__)
        all_properties.update(self.fluid_properties.__dict__)

        property_value = all_properties.get(property_name)

        if property_value is None:
            raise ValueError(f"Property {property_name} is not one of the parameters of the interface material")

        return property_value
