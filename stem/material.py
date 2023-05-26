from dataclasses import dataclass

from retention_law import SaturatedBelowPhreaticLevelLaw, SaturatedLaw, VanGenuchtenLaw

@dataclass
class LinearElastic2D:
    """
    Class containing the material parameters for a 2D linear elastic material
    """
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    DENSITY_SOLID: float
    DENSITY_WATER: float
    POROSITY: float
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
class LinearElastic3D:
    """
    Class containing the material parameters for a 3D linear elastic material
    """
    YOUNG_MODULUS: float
    POISSON_RATIO: float
    DENSITY_SOLID: float
    DENSITY_WATER: float
    POROSITY: float
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


@dataclass
class SmallStrainUmat2DLaw:
    """
    Class containing the material parameters for a 2D small strain umat material
    """
    UMAT_NAME: str # UDSM name
    IS_FORTRAN_UMAT: bool
    NUMBER_OF_UMAT_PARAMETERS: int
    UMAT_PARAMETERS: list
    STATE_VARIABLES: list
    DENSITY_SOLID: float
    DENSITY_WATER: float
    POROSITY: float
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
class SmallStrainUmat3DLaw:
    """
    Class containing the material parameters for a 3D small strain umat material
    """
    UMAT_NAME: str
    IS_FORTRAN_UMAT: bool
    NUMBER_OF_UMAT_PARAMETERS: int
    UMAT_PARAMETERS: list
    STATE_VARIABLES: list
    DENSITY_SOLID: float
    DENSITY_WATER: float
    POROSITY: float
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
class SmallStrainUdsm2DLaw:
    """
    Class containing the material parameters for a 2D small strain udsm material
    """
    UDSM_NAME: str
    UDSM_NUMBER: int
    IS_FORTRAN_UDSM: bool
    UDSM_PARAMETERS: list
    DENSITY_SOLID: float
    DENSITY_WATER: float
    POROSITY: float
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
class SmallStrainUdsm3DLaw:
    """
    Class containing the material parameters for a 3D small strain udsm material
    """
    UDSM_NAME: str
    UDSM_NUMBER: int
    IS_FORTRAN_UDSM: bool
    UDSM_PARAMETERS: list
    DENSITY_SOLID: float
    DENSITY_WATER: float
    POROSITY: float
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


class Material:
    """
    Class containing material information about a body part, e.g. a soil layer or track components

    Attributes:
        name (str): name of the material
        parameters (dict): dictionary containing the material parameters

    """

    def __init__(self, name, material_parameters, retention_parameters=SaturatedBelowPhreaticLevelLaw()):

        self.id = 0
        self.name = name
        self.material_parameters = material_parameters
        self.retention_parameters = retention_parameters