from dataclasses import dataclass
from abc import ABC

@dataclass
class RetentionLawABC(ABC):
    """
    Abstract class containing the parameters for a retention law. This class is created for type checking purposes.

    """
    pass

@dataclass
class SaturatedBelowPhreaticLevelLaw(RetentionLawABC):
    """
    Class containing the parameters for the retention law: saturated below phreatic level

    :Inheritance: RetentionLaw

    :Attributes:
        SATURATED_SATURATION (float): The saturation ratio below phreatic level [-].
        RESIDUAL_SATURATION (float): The residual saturation ratio [-].

    """
    SATURATED_SATURATION: float = 1.0
    RESIDUAL_SATURATION: float = 1e-10

@dataclass
class SaturatedLaw(RetentionLawABC):
    """
    Class containing the parameters for the retention law: saturated

    :Inheritance: RetentionLaw

    :Attributes:
        SATURATED_SATURATION (float): The saturation ratio [-].

    """
    SATURATED_SATURATION: float = 1.0

@dataclass
class VanGenuchtenLaw(RetentionLawABC):
    """
    Class containing the parameters for a retention law

    :Inheritance: RetentionLaw

    :Attributes:
        SATURATED_SATURATION (float): The maximum saturation ratio [-].
        RESIDUAL_SATURATION (float): The minum saturation ratio [-].
        VAN_GENUCHTEN_AIR_ENTRY_PRESSURE (float): The air entry pressure [Pa].
        VAN_GENUCHTEN_GN (float): The pore size distribution index [-].
        VAN_GENUCHTEN_GL (float): exponent for calculating relative permeability [-].
        MINIMUM_RELATIVE_PERMEABILITY (float): The minimum relative permeability [-].

    """
    SATURATED_SATURATION = 1.0
    RESIDUAL_SATURATION = 1e-10
    VAN_GENUCHTEN_AIR_ENTRY_PRESSURE = 2.561
    VAN_GENUCHTEN_GN = 1.377
    VAN_GENUCHTEN_GL = 1.25
    MINIMUM_RELATIVE_PERMEABILITY = 0.0001

