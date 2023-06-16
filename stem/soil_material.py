from typing import List, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC


@dataclass
class SoilFormulationParametersABC(ABC):
    """
    Abstract base class for soil formulation parameters

    Attributes:
        ndim (int): The number of dimensions of the soil formulation (2 or 3)
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


@dataclass
class FluidProperties:
    """
    Class containing the parameters for a fluid. Default values are for water at 12 degrees Celsius.

    Attributes:
        DENSITY_FLUID (float): The density of fluid [kg/m^3].
        DYNAMIC_VISCOSITY (float): The dynamic viscosity of fluid [Pa s].
        BULK_MODULUS_FLUID (float): The bulk modulus of fluid [Pa].
    """
    DENSITY_FLUID: float = 1000
    DYNAMIC_VISCOSITY: float = 1.3e-3
    BULK_MODULUS_FLUID: float = 2e9

@dataclass
class OnePhaseSoil(SoilFormulationParametersABC):
    """
    Class containing the material parameters for an undrained soil material

    Attributes:
        IS_UNDRAINED (bool): Boolean indicating if the soil is undrained.
        DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        POROSITY (float): The porosity of the soil [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        BIOT_COEFFICIENT (float): The Biot coefficient [-].

    """
    IS_DRAINED: bool
    DENSITY_SOLID: float
    POROSITY: float
    BULK_MODULUS_SOLID: float = 50e9
    BIOT_COEFFICIENT: Optional[float] = None


@dataclass
class TwoPhaseSoil(SoilFormulationParametersABC):
    """
    Class containing the material parameters for a two phase soil material

    Attributes:
        DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        POROSITY (float): The porosity of the soil [-].
        PERMEABILITY_XX (float): The permeability in the x-direction [m^2].
        PERMEABILITY_YY (float): The permeability in the y-direction [m^2].
        PERMEABILITY_XY (float): The permeability in the xy-direction [m^2].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        BIOT_COEFFICIENT (float): The Biot coefficient [-].
        PERMEABILITY_YZ (float): The permeability in the yz-direction [m^2].
        PERMEABILITY_ZX (float): The permeability in the zx-direction [m^2].
        PERMEABILITY_ZZ (float): The permeability in the z-direction [m^2].

    """

    DENSITY_SOLID: float
    POROSITY: float
    PERMEABILITY_XX: float
    PERMEABILITY_YY: float
    PERMEABILITY_XY: float = 0
    BULK_MODULUS_SOLID: float = 50e9
    BIOT_COEFFICIENT: Optional[float] = None

    # parameters for 3D
    PERMEABILITY_YZ: Optional[float] = 0
    PERMEABILITY_ZX: Optional[float] = 0

    PERMEABILITY_ZZ: Optional[float] = None

    def __post_init__(self):
        """
        Check if the permeability parameters are defined for the correct number of dimensions

        Returns:

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

    Attributes:
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].

    """
    YOUNG_MODULUS: float
    POISSON_RATIO: float


@dataclass
class SmallStrainUmatLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for a 2D small strain umat material

    Attributes:
        UMAT_NAME (str): The name and location of the umat .dll or .so file.
        IS_FORTRAN_UMAT (bool): A boolean to indicate whether the umat is written in Fortran.
        UMAT_PARAMETERS (list): The parameters of the umat.
        STATE_VARIABLES (list): The state variables of the umat.

    """
    UMAT_NAME: str
    IS_FORTRAN_UMAT: bool
    UMAT_PARAMETERS: List[Any]
    STATE_VARIABLES: List[Any]


@dataclass
class SmallStrainUdsmLaw(SoilConstitutiveLawABC):
    """
    Class containing the material parameters for small strain udsm material

    Attributes:
        UDSM_NAME (str): The name and location of the udsm .dll or .so file.
        UDSM_NUMBER (int): The model number within the udsm.
        IS_FORTRAN_UDSM (bool): A boolean to indicate whether the udsm is written in Fortran.
        UDSM_PARAMETERS (list): The parameters of the udsm.

    """
    UDSM_NAME: str
    UDSM_NUMBER: int
    IS_FORTRAN_UDSM: bool
    UDSM_PARAMETERS: List[Any]


@dataclass
class SaturatedBelowPhreaticLevelLaw(RetentionLawABC):
    """
    Class containing the parameters for the retention law: saturated below phreatic level

    Inheritance: RetentionLawABC

    Attributes:
        SATURATED_SATURATION (float): The saturation ratio below phreatic level [-].
        RESIDUAL_SATURATION (float): The residual saturation ratio [-].

    """
    SATURATED_SATURATION: float = 1.0
    RESIDUAL_SATURATION: float = 1e-10


@dataclass
class SaturatedLaw(RetentionLawABC):
    """
    Class containing the parameters for the retention law: saturated

    Inheritance: RetentionLawABC

    Attributes:
        SATURATED_SATURATION (float): The saturation ratio [-].

    """
    SATURATED_SATURATION: float = 1.0


@dataclass
class VanGenuchtenLaw(RetentionLawABC):
    """
    Class containing the parameters for a retention law

    Inheritance: RetentionLawABC

    Attributes:
        VAN_GENUCHTEN_AIR_ENTRY_PRESSURE (float): The air entry pressure [Pa].
        VAN_GENUCHTEN_GN (float): The pore size distribution index [-].
        VAN_GENUCHTEN_GL (float): exponent for calculating relative permeability [-].
        SATURATED_SATURATION (float): The maximum saturation ratio [-].
        RESIDUAL_SATURATION (float): The minum saturation ratio [-].
        MINIMUM_RELATIVE_PERMEABILITY (float): The minimum relative permeability [-].

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
        name (str): The name of the material.
        soil_formulation (SoilFormulationParametersABC): The soil formulation parameters.
        constitutive_law (SoilConstitutiveLawABC): The soil constitutive law parameters.
        retention_parameters (RetentionLawABC): The retention law parameters.
        fluid_properties (FluidProperties): The fluid properties.
    """
    name: str
    soil_formulation: SoilFormulationParametersABC
    constitutive_law: SoilConstitutiveLawABC
    retention_parameters: RetentionLawABC
    fluid_properties: FluidProperties = field(default_factory=FluidProperties)
