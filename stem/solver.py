from dataclasses import dataclass
from enum import Enum
from abc import ABC

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

    Attributes:
        displacement_relative_tolerance (float): The relative tolerance for the displacement.
        displacement_absolute_tolerance (float): The absolute tolerance for the displacement.

    """
    displacement_relative_tolerance: float = 1e-4
    displacement_absolute_tolerance: float = 1e-9

class ResidualConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the residual convergence criteria

    Attributes:
        residual_relative_tolerance (float): The relative tolerance for the residual.
        residual_absolute_tolerance (float): The absolute tolerance for the residual.

    """

    residual_relative_tolerance: float = 1e-4
    residual_absolute_tolerance: float = 1e-9

class WaterPressureConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the water pressure convergence criteria

    Attributes:
        water_pressure_relative_tolerance (float): The relative tolerance for the water pressure.
        water_pressure_absolute_tolerance (float): The absolute tolerance for the water pressure.

    """

    water_pressure_relative_tolerance: float = 1e-4
    water_pressure_absolute_tolerance: float = 1e-9

class DisplacementAndWaterPressureConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the displacement and water pressure convergence criteria

    Attributes:
        displacement_relative_tolerance (float): The relative tolerance for the displacement.
        displacement_absolute_tolerance (float): The absolute tolerance for the displacement.
        water_pressure_relative_tolerance (float): The relative tolerance for the water pressure.
        water_pressure_absolute_tolerance (float): The absolute tolerance for the water pressure.

    """

    displacement_relative_tolerance: float = 1e-4
    displacement_absolute_tolerance: float = 1e-9
    water_pressure_relative_tolerance: float = 1e-4
    water_pressure_absolute_tolerance: float = 1e-9



@dataclass
class NewmarkScheme(SchemeABC):
    """
    Class containing information about the Newmark scheme

    Attributes:
        newmark_beta (float): The beta parameter of the Newmark scheme.
        newmark_gamma (float): The gamma parameter of the Newmark scheme.
        newmark_theta (float): The theta parameter of the Newmark scheme, which is used for water pressure.
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

    Attributes:
        QUASI_STATIC (int): quasi-static solution type
        K0_PROCEDURE (int): K0-procedure solution type
        DYNAMIC (int): dynamic solution type

    """
    QUASI_STATIC = 1
    K0_PROCEDURE = 2
    DYNAMIC = 3

@dataclass
class StrategyTypeABC(ABC):
    """
    Abstract class for the strategy type

    Attributes:
        max_iterations (int): maximum number of iterations allowed, if this number is reached, the time step size is
            decreased and the algorithm is restarted
        min_iterations (int): minimum number of iterations, below this number, the time step size is increased
        number_cycles (int): number of allowed cycles of decreasing the time step size until the algorithm is stopped.

    """
    max_iterations: int = 15
    min_iterations: int = 6
    number_cycles: int = 100

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

    Attributes:
        max_line_search_iterations (int): maximum number of line search iterations
        first_alpha_value (float): first alpha guess value used for the first iteration
        second_alpha_value (float): second alpha guess value used for the first iteration
        min_alpha (float): minimum possible alpha value at the end of the algorithm
        max_alpha (float): maximum possible alpha value at the end of the algorithm
        line_search_tolerance (float): Tolerance of the line search algorithm, defined as the ratio between maximum
            residual*alpha*dx and current iteration residual*alpha*dx
        echo_level (int): echo level
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

    Attributes:
        desired_iterations (int): This is used to calculate the radius of the next step
        max_radius_factor (float): maximum radius factor of the arc
        min_radius_factor (float): minimum radius factor of the arc

    """
    desired_iterations: int = 10
    max_radius_factor: float = 1.0
    min_radius_factor: float = 0.1


@dataclass
class LinearSolverSettingsABC(ABC):
    """
    Class containing information about the linear solver settings

    Attributes:
        scaling (bool): if true, the system matrix will be scaled before solving the linear system of equations

    """
    scaling: bool = False


@dataclass
class Amgcl(LinearSolverSettingsABC):
    """
    Class containing information about the amgcl linear solver settings

    Attributes:
        tolerance (float): tolerance for the linear solver convergence criteria
        max_iterations (int): maximum number of iterations for the linear solver

    """
    tolerance: float = 1e-6
    max_iterations: int = 1000

@dataclass
class TimeIntegration:
    """
    Class containing information about the time integration

    Attributes:
        start_time (float): start time of the analysis
        end_time (float): end time of the analysis
        delta_time (float): initial time step
        reduction_factor (float): factor used to reduce the time step when the solution diverges
        increase_factor (float): factor used to increase the time step when the solution converges within the minimum
            number of iterations
        max_delta_time_factor (float): maximum time step factor, used to limit the time step increase

    """

    start_time: float
    end_time: float
    delta_time: float

    reduction_factor: float
    increase_factor: float
    max_delta_time_factor: float = 1000

@dataclass
class SolverSettings:
    """
    Class containing information about the time integration, builder, strategy, scheme and linear solver.

    Attributes:
        solution_type (SolutionType): solution type, quasi-static, K0-procedure or dynamic
        time_integration (TimeIntegration): time integration settings
        rebuild_level (int): 2 if the lhs matrix is rebuilt at each non-linear iteration, 1 if the lhs matrix
            is rebuilt at each time step, 0 if the lhs matrix is only built once
        prebuild_dynamics (bool): if true, the mass and damping matrices are prebuilt and directly used to calculate the
            rhs. If false, the mass and damping matrices are built at each non linear iteration for calculating the rhs
        convergence_criteria (ConvergenceCriteriaABC): convergence criteria, displacement, residual, water pressure or
            displacement and water pressure
        reset_displacements (bool): if true, the displacements are reset at the beginning of the phase
        strategy_type (StrategyTypeABC): strategy type, Newton-Raphson, line search or arc length

        scheme (SchemeABC): scheme, Newmark or backward Euler
        linear_solver_settings (LinearSolverSettingsABC): linear solver settings, currently only AMGCL is supported
        rayleigh_m (Optional[float]): mass proportional damping parameter
        rayleigh_k (Optional[float]): stiffness proportional damping parameter

    """

    solution_type: SolutionType
    time_integration: TimeIntegration

    rebuild_level: int
    prebuild_dynamics: bool

    convergence_criteria: ConvergenceCriteriaABC
    reset_displacements: bool = False
    strategy_type: StrategyTypeABC = NewtonRaphsonStrategy()
    scheme: SchemeABC = NewmarkScheme()
    linear_solver_settings: LinearSolverSettingsABC = Amgcl()
    rayleigh_m: Optional[float] = None
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
class Problem:
    """
    Class containing information about the builder, strategy, scheme and linear solver.

    Attributes:
        problem_name (str): name of the problem
        number_of_threads (int): number of threads used for the analysis
        settings (SolverSettings): dictionary containing the solver settings
        echo_level (int): echo level

    """

    problem_name: str
    number_of_threads: int
    settings: SolverSettings
    echo_level: int = 1




