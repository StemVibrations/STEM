from dataclasses import dataclass
from enum import Enum

@dataclass
class SaturatedBelowPhreaticLevelLaw:
    """
    Class containing the parameters for a retention law

    """
    SATURATED_SATURATION: float = 1.0
    RESIDUAL_SATURATION: float = 1e-10

@dataclass
class SaturatedLaw:
    """
    Class containing the parameters for a retention law

    """
    SATURATED_SATURATION: float = 1.0

@dataclass
class VanGenuchtenLaw:
    """
    Class containing the parameters for a retention law

    """
    SATURATED_SATURATION = 1.0
    RESIDUAL_SATURATION = 1e-10
    VAN_GENUCHTEN_AIR_ENTRY_PRESSURE = 2.561
    VAN_GENUCHTEN_GN = 1.377
    VAN_GENUCHTEN_GL = 1.25
    MINIMUM_RELATIVE_PERMEABILITY = 0.0001

