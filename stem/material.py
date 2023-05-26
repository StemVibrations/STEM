from typing import Union, List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC

from stem.retention_law import RetentionLawABC, SaturatedBelowPhreaticLevelLaw


@dataclass
class MaterialParametersABC(ABC):
    pass

@dataclass
class SoilMaterial2D(MaterialParametersABC):
    """
    Class containing the material parameters for a 2D soil material

    :Attributes:
        DENSITY_SOLID (float): The density of the solid [kg/m3].
        DENSITY_WATER (float): The density of the water [kg/m3].
        POROSITY (float): The porosity [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid [Pa].
        BULK_MODULUS_FLUID (float): The bulk modulus of the fluid [Pa].
        PERMEABILITY_XX (float): The intrinsic permeability in the x-direction [m2].
        PERMEABILITY_YY (float): The intrinsic permeability in the y-direction [m2].
        PERMEABILITY_XY (float): The intrinsic permeability in the xy-direction [m2].
        DYNAMIC_VISCOSITY (float): The dynamic viscosity [Pa s].
        IGNORE_UNDRAINED (bool): A boolean to indicate whether undrained behaviour should be ignored.
        BIOT_COEFFICIENT (float): The Biot coefficient [-].

    """
    DENSITY_SOLID: float = 2650
    DENSITY_WATER: float = 1000
    POROSITY: float = 0.3
    BULK_MODULUS_SOLID: float = 1e16
    BULK_MODULUS_FLUID: float = 2e-30
    PERMEABILITY_XX: float = 4.5e-30
    PERMEABILITY_YY: float = 4.5e-30
    PERMEABILITY_XY: float = 0
    DYNAMIC_VISCOSITY: float = 1e-3
    THICKNESS: float = 1.0
    IGNORE_UNDRAINED: bool = True
    BIOT_COEFFICIENT: float = 1.0

@dataclass
class SoilMaterial3D(MaterialParametersABC):
    """
    Class containing the material parameters for a 3D soil material

    :Attributes:
        DENSITY_SOLID (float): The density of the solid [kg/m3].
        DENSITY_WATER (float): The density of the water [kg/m3].
        POROSITY (float): The porosity [-].
        BULK_MODULUS_SOLID (float): The bulk modulus of the solid [Pa].
        BULK_MODULUS_FLUID (float): The bulk modulus of the fluid [Pa].
        PERMEABILITY_XX (float): The intrinsic permeability in the x-direction [m2].
        PERMEABILITY_YY (float): The intrinsic permeability in the y-direction [m2].
        PERMEABILITY_ZZ (float): The intrinsic permeability in the z-direction [m2].
        PERMEABILITY_XY (float): The intrinsic permeability in the xy-direction [m2].
        PERMEABILITY_YZ (float): The intrinsic permeability in the yz-direction [m2].
        PERMEABILITY_ZX (float): The intrinsic permeability in the zx-direction [m2].
        DYNAMIC_VISCOSITY (float): The dynamic viscosity [Pa s].
        IGNORE_UNDRAINED (bool): A boolean to indicate whether undrained behaviour should be ignored.
        BIOT_COEFFICIENT (float): The Biot coefficient [-].
    """

    DENSITY_SOLID: float = 2650
    DENSITY_WATER: float = 1000
    POROSITY: float = 0.3
    BULK_MODULUS_SOLID: float = 1e16
    BULK_MODULUS_FLUID: float = 2e-30
    PERMEABILITY_XX: float = 4.5e-30
    PERMEABILITY_YY: float = 4.5e-30
    PERMEABILITY_ZZ: float = 4.5e-30
    PERMEABILITY_XY: float = 0
    PERMEABILITY_YZ: float = 0
    PERMEABILITY_ZX: float = 0
    DYNAMIC_VISCOSITY: float = 1e-3
    IGNORE_UNDRAINED: bool = True
    BIOT_COEFFICIENT: float = 1.0


@dataclass
class LinearElastic2D(SoilMaterial2D):
    """
    Class containing the material parameters for a 2D linear elastic material

    :Attributes:
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].

    """
    YOUNG_MODULUS: float = 1e9
    POISSON_RATIO: float = 0.0

@dataclass
class LinearElastic3D(SoilMaterial3D):
    """
    Class containing the material parameters for a 3D linear elastic material

    :Attributes:
        YOUNG_MODULUS (float): The Young's modulus [Pa].
        POISSON_RATIO (float): The Poisson's ratio [-].


    """
    YOUNG_MODULUS: float = 1e9
    POISSON_RATIO: float = 0.0

@dataclass
class SmallStrainUmat2DLaw(SoilMaterial2D):
    """
    Class containing the material parameters for a 2D small strain umat material

    :Attributes:
        UMAT_NAME (str): The name and location of the umat .dll or .so file.
        IS_FORTRAN_UMAT (bool): A boolean to indicate whether the umat is written in Fortran.
        NUMBER_OF_UMAT_PARAMETERS (int): The number of parameters in the umat.
        UMAT_PARAMETERS (list): The parameters of the umat.
        STATE_VARIABLES (list): The state variables of the umat.

    """
    UMAT_NAME: str = ""
    IS_FORTRAN_UMAT: bool = False
    UMAT_PARAMETERS: list = field(default_factory=list)
    STATE_VARIABLES: list = field(default_factory=list)

@dataclass
class SmallStrainUmat3DLaw(SoilMaterial3D):
    """
    Class containing the material parameters for a 3D small strain umat material

    :Attributes:
        UMAT_NAME (str): The name and location of the umat .dll or .so file.
        IS_FORTRAN_UMAT (bool): A boolean to indicate whether the umat is written in Fortran.
        UMAT_PARAMETERS (list): The parameters of the umat.
        STATE_VARIABLES (list): The state variables of the umat.

    """
    UMAT_NAME: str = ""
    IS_FORTRAN_UMAT: bool = False
    UMAT_PARAMETERS: list = field(default_factory=list)
    STATE_VARIABLES: list = field(default_factory=list)

@dataclass
class SmallStrainUdsm2DLaw(SoilMaterial2D):
    """
    Class containing the material parameters for a 2D small strain udsm material

    :Attributes:
        UDSM_NAME (str): The name and location of the udsm .dll or .so file.
        UDSM_NUMBER (int): The model number within the udsm.
        IS_FORTRAN_UDSM (bool): A boolean to indicate whether the udsm is written in Fortran.
        UDSM_PARAMETERS (list): The parameters of the udsm.


    """
    UDSM_NAME: str = ""
    UDSM_NUMBER: int = 0
    IS_FORTRAN_UDSM: bool = False
    UDSM_PARAMETERS: list = field(default_factory=list)


@dataclass
class SmallStrainUdsm3DLaw(SoilMaterial3D):
    """
    Class containing the material parameters for a 3D small strain udsm material

    :Attributes:
        UDSM_NAME (str): The name and location of the udsm .dll or .so file.
        UDSM_NUMBER (int): The model number within the udsm.
        IS_FORTRAN_UDSM (bool): A boolean to indicate whether the udsm is written in Fortran.
        UDSM_PARAMETERS (list): The parameters of the udsm.

    """
    UDSM_NAME: str = ""
    UDSM_NUMBER: int = 0
    IS_FORTRAN_UDSM: bool = False
    UDSM_PARAMETERS: list = field(default_factory=list)


class Material:
    """
    Class containing material information about a body part, e.g. a soil layer or track components

    Attributes:
        id (int): unique id of the material
        name (str): name of the material
        material_parameters (MaterialParametersABC): class containing material parameters
        retention_parameters (RetentionLawABC): class containing the retention parameters

    """

    def __init__(self, name: str, material_parameters: MaterialParametersABC, id: int = 0):

        self.id = id
        self.name: str = name
        self.material_parameters: MaterialParametersABC = material_parameters
        self.retention_parameters: RetentionLawABC = SaturatedBelowPhreaticLevelLaw()

