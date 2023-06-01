from typing import List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC

@dataclass
class LoadParametersABC(ABC):
    """
    Abstract base class for load parameters
    """
    pass


@dataclass
class PointLoad(LoadParametersABC):
    """
    Class containing the load parameters for a point load
    :Attributes:
        DENSITY_SOLID (float): The density of the solid [kg/m3].
        DENSITY_WATER (float): The density of the water [kg/m3].
        POROSITY (float): The porosity [-].
    """

    active: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class Load:
    """
    Class containing load information acting on a body part, e.g. a soil layer or track
    components
    Attributes:
        name (str): name of the load
        material_parameters (MaterialParametersABC): class containing load parameters
    """

    def __init__(self, name: str, load_parameters: LoadParametersABC):
        """
        Constructor of the material class
        Args:
            name (str): name of the material
            load_parameters (MaterialParametersABC): class containing load
            parameters
        """

        # self.id: int = id
        self.name: str = name
        self.load_parameters: LoadParametersABC = load_parameters