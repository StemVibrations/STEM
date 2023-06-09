from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


@dataclass
class SchemeABC(ABC):
    """
    Abstract class for the scheme
    """
    pass


@dataclass
class NewmarkScheme(SchemeABC):
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
@dataclass
class BackwardEulerScheme(SchemeABC):
    pass

class SolutionType(Enum):
    """
    Enum class containing the solution types
    """
    QUASI_STATIC = 1
    K0_PROCEDURE = 2
    DYNAMIC = 3

@dataclass
class SolverSettings:
    """
    Class containing information about the solver settings

    Attributes:
        settings (dict): dictionary containing the solver settings

    """

    start_time: float
    end_time: float
    delta_time: float
    solution_type: SolutionType
    reset_displacements: bool


@dataclass
class Solver:
    """
    Class containing information about the builder, solver and strategy

    Attributes:
        settings (dict): dictionary containing the solver settings

    """

    name: str = ""
    echo_level: int = 1
    number_of_threads: int = 1
    settings: dict = None




