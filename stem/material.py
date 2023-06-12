from typing import List, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC


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


