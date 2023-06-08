from typing import List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC

from stem.retention_law import RetentionLawABC, SaturatedBelowPhreaticLevelLaw




@dataclass
class SoilTypeParametersABC(ABC):
    """
    Abstract base class for material parameters
    """
    pass


@dataclass
class StructuralParametersABC(ABC):
    """
    Abstract base class for material parameters
    """
    pass


@dataclass
class FluidProperties:
    """
    Class containing the parameters for a fluid

    Attributes:
        DENSITY_WATER (float): The density of water [kg/m^3].
        DYNAMIC_VISCOSITY (float): The dynamic viscosity of water [Pa s].
        BULK_MODULUS_FLUID (float): The bulk modulus of water [Pa].
    """
    DENSITY_WATER: float = 1000
    DYNAMIC_VISCOSITY: float = 1.3e-3
    BULK_MODULUS_FLUID: float = 2e-30


@dataclass
class DrainedSoil(SoilTypeParametersABC):
    """
    Class containing the material parameters for a drained soil material

    Attributes:
        DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        POROSITY (float): The porosity of the soil [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        BIOT_COEFFICIENT (float): The Biot coefficient [-].
    """
    DENSITY_SOLID: float
    POROSITY: float
    BULK_MODULUS_SOLID: float
    BIOT_COEFFICIENT: Optional[float] = None


@dataclass
class UndrainedSoil(SoilTypeParametersABC):
    """
    Class containing the material parameters for an undrained soil material

    Attributes:
        DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        POROSITY (float): The porosity of the soil [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        BIOT_COEFFICIENT (float): The Biot coefficient [-].

    """
    DENSITY_SOLID: float
    POROSITY: float
    BULK_MODULUS_SOLID: float
    BIOT_COEFFICIENT: Optional[float] = None


@dataclass
class TwoPhaseSoil2D(SoilTypeParametersABC):
    """
    Class containing the material parameters for a two phase 2D soil material

    Attributes:
        DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        POROSITY (float): The porosity of the soil [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        PERMEABILITY_XX (float): The permeability in the x-direction [m^2].
        PERMEABILITY_YY (float): The permeability in the y-direction [m^2].
        PERMEABILITY_XY (float): The permeability in the xy-direction [m^2].
        BIOT_COEFFICIENT (float): The Biot coefficient [-].
    """

    DENSITY_SOLID: float
    POROSITY: float
    BULK_MODULUS_SOLID: float
    PERMEABILITY_XX: float
    PERMEABILITY_YY: float
    PERMEABILITY_XY: float
    BIOT_COEFFICIENT: Optional[float] = None


@dataclass
class TwoPhaseSoil3D(SoilTypeParametersABC):
    """
    Class containing the material parameters for a two phase 3D soil material

    Attributes:
        DENSITY_SOLID (float): The density of the solid material [kg/m^3].
        POROSITY (float): The porosity of the soil [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid material [Pa].
        PERMEABILITY_XX (float): The permeability in the x-direction [m^2].
        PERMEABILITY_YY (float): The permeability in the y-direction [m^2].
        PERMEABILITY_ZZ (float): The permeability in the z-direction [m^2].
        PERMEABILITY_XY (float): The permeability in the xy-direction [m^2].
        PERMEABILITY_YZ (float): The permeability in the yz-direction [m^2].
        PERMEABILITY_ZX (float): The permeability in the zx-direction [m^2].
        BIOT_COEFFICIENT (float): The Biot coefficient [-].
    """
    DENSITY_SOLID: float
    POROSITY: float
    BULK_MODULUS_SOLID: float
    PERMEABILITY_XX: float
    PERMEABILITY_YY: float
    PERMEABILITY_ZZ: float
    PERMEABILITY_XY: float = 0
    PERMEABILITY_YZ: float = 0
    PERMEABILITY_ZX: float = 0
    BIOT_COEFFICIENT: Optional[float] = None

@dataclass
class SoilParametersABC(ABC):
    """
    Abstract base class for material parameters
    """
    SOIL_TYPE: SoilTypeParametersABC
    RETENTION_PARAMETERS: RetentionLawABC

@dataclass
class LinearElasticSoil(SoilParametersABC):
    """
    Class containing the material parameters for a 2D linear elastic material

    Attributes:
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].
        SOIL_TYPE (Union[DrainedSoil, UndrainedSoil, TwoPhaseSoil2D, TwoPhaseSoil3D]): The soil type.

    """
    YOUNG_MODULUS: float
    POISSON_RATIO: float


@dataclass
class SmallStrainUmatLaw(SoilParametersABC):
    """
    Class containing the material parameters for a 2D small strain umat material

    Attributes:
        UMAT_NAME (str): The name and location of the umat .dll or .so file.
        IS_FORTRAN_UMAT (bool): A boolean to indicate whether the umat is written in Fortran.
        UMAT_PARAMETERS (list): The parameters of the umat.
        STATE_VARIABLES (list): The state variables of the umat.
        SOIL_TYPE (Union[DrainedSoil, UndrainedSoil, TwoPhaseSoil2D, TwoPhaseSoil3D]): The soil type.

    """
    UMAT_NAME: str
    IS_FORTRAN_UMAT: bool
    UMAT_PARAMETERS: List[Any]
    STATE_VARIABLES: List[Any]


@dataclass
class SmallStrainUdsmLaw(SoilParametersABC):
    """
    Class containing the material parameters for a 2D small strain udsm material

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

    :Attributes:
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
        NODAL_DISPLACEMENT_STIFFNESS (float): The stiffness of the spring [N/m].
        NODAL_ROTATIONAL_STIFFNESS (float): The stiffness of the rotational spring [Nm/rad].
        NODAL_DAMPING_COEFFICIENT (float): The damping coefficient of the spring [Ns/m].
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT (float): The damping coefficient of the rotational spring [Ns/rad].
    """
    NODAL_DISPLACEMENT_STIFFNESS: float
    NODAL_ROTATIONAL_STIFFNESS: float
    NODAL_DAMPING_COEFFICIENT: float
    NODAL_ROTATIONAL_DAMPING_COEFFICIENT: float


@dataclass
class NodalConcentrated(StructuralParametersABC):
    """
    Class containing the material parameters for a nodal concentrated element

    Attributes:
        NODAL_DISPLACEMENT_STIFFNESS (float): The stiffness of the spring [N/m].
        NODAL_MASS (float): The mass of the concentrated element [kg].
        NODAL_DAMPING_COEFFICIENT (float): The damping coefficient of the spring [Ns/m].
    """
    NODAL_DISPLACEMENT_STIFFNESS: float
    NODAL_MASS: float
    NODAL_DAMPING_COEFFICIENT: float


@dataclass
class Material:
    """
    Class containing material information about a body part, e.g. a soil layer or track components

    Attributes:
        id (int): unique id of the material
        name (str): name of the material
        material_parameters (MaterialParametersABC): class containing material parameters

    """

    id: int
    name: str
    material_parameters: Union[SoilParametersABC, StructuralParametersABC]
    fluid_properties: FluidProperties = FluidProperties()

