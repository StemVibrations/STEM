from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from typing import Optional


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
    displacement_relative_tolerance: float = 1e-4
    displacement_absolute_tolerance: float = 1e-9

class ResidualConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the residual convergence criteria
    """

    residual_relative_tolerance: float = 1e-4
    residual_absolute_tolerance: float = 1e-9

class WaterPressureConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the water pressure convergence criteria
    """

    water_pressure_relative_tolerance: float = 1e-4
    water_pressure_absolute_tolerance: float = 1e-9

class DisplacementAndWaterPressureConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the displacement and water pressure convergence criteria
    """

    displacement_relative_tolerance: float = 1e-4
    displacement_absolute_tolerance: float = 1e-9
    water_pressure_relative_tolerance: float = 1e-4
    water_pressure_absolute_tolerance: float = 1e-9



@dataclass
class NewmarkScheme(SchemeABC):
    """
    Class containing information about the Newmark scheme
    """
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    newmark_theta: float = 0.5


@dataclass
class BackwardEulerScheme(SchemeABC):
    """
    Class containing information about the backward Euler scheme
    """
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
    desired_iterations: int = 10
    max_radius_factor: float = 1.0
    min_radius_factor: float = 0.1


@dataclass
class LinearSolverSettingsABC(ABC):
    """
    Class containing information about the linear solver settings
    """
    pass

@dataclass
class amgcl(LinearSolverSettingsABC):
    """
    Class containing information about the amgcl linear solver settings
    """
    scaling: bool = False
    tolerance: float = 1e-6
    max_iterations: int = 1000


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

    rebuild_level: int
    solution_type: SolutionType
    reset_displacements: bool
    strategy_type: StrategyTypeABC
    convergence_criteria: ConvergenceCriteriaABC
    scheme: SchemeABC
    linear_solver_settings: LinearSolverSettingsABC
    rayleigh_m:Optional[float] = None
    rayleigh_k: Optional[float] = None

    def __post_init__(self):
        """
        Post initialization method

        Raises:
            ValueError: if the rayleigh damping parameters are not provided for dynamic analysis
        """
        if self.solution_type == SolutionType.DYNAMIC:
            if self.rayleigh_m is None or self.rayleigh_k is None:
                raise ValueError("Rayleigh damping parameters must be provided for dynamic analysis")


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




