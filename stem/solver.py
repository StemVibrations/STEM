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
class ConvergenceCriteriaABC(ABC):
    """
    Abstract class for the convergence criteria
    """
    pass

class DisplacementConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the displacement convergence criteria
    """
    pass

class ResidualConvergenceCriteria(ConvergenceCriteriaABC):

    pass

class WaterPressureConvergenceCriteria(ConvergenceCriteriaABC):

    pass

class DisplacementAndWaterPressureConvergenceCriteria(ConvergenceCriteriaABC):

    pass

@dataclass
class NewmarkScheme(SchemeABC):
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    newmark_theta: float = 0.0
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
class StrategyTypeABC(ABC):
    max_iterations: int = 15
    min_iterations: int = 2
    number_cycles: int = 5

@dataclass
class NewtonRaphsonStrategy(StrategyTypeABC):
    """
    Class containing information about the Newton-Raphson strategy
    """
    pass

@dataclass
class LineSearchStrategy(StrategyTypeABC):
    """
    Class containing information about the line search strategy
    """
    max_iterations: int = 15
    min_iterations: int = 2
    max_line_search_iterations: int = 10
    first_alpha_value: float = 1.0
    second_alpha_value: float = 0.5
    min_alpha: float = 1e-4
    max_alpha: float = 1e4
    line_search_tolerance: float = 1e-4
    echo_level: int = 0

@dataclass
class ArcLengthStrategy(StrategyTypeABC):
    """
    Class containing information about the arc length strategy
    """
    max_iterations: int = 15
    min_iterations: int = 2
    desired_iterations: int = 10
    max_radius_factor: float = 1.0
    min_radius_factor: float = 0.1


@dataclass
class LinearSolverSettingsABC(ABC):
    """
    Class containing information about the linear solver settings
    """
    pass

class amgcl(LinearSolverSettingsABC):
    """
    Class containing information about the amgcl linear solver settings
    """
    scaling: bool = False



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
    max_delta_time_factor: float
    reduction_factor: float
    increase_factor: float

    solution_type: SolutionType
    reset_displacements: bool
    rayleigh_m: float
    rayleigh_k: float
    strategy_type: StrategyTypeABC
    convergence_criteria: ConvergenceCriteriaABC
    scheme: SchemeABC
    linear_solver_settings: LinearSolverSettingsABC


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




