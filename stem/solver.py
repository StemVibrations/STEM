import abc
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC

from typing import Optional


@dataclass
class SchemeABC(ABC):
    """
    Abstract class for the scheme
    """

    @property
    @abc.abstractmethod
    def scheme_type(self):
        """
        Abstract property for returning the type of the scheme

        Raises:
            - Exception: abstract class of scheme is called

        """
        Exception("abstract class of scheme is called")


@dataclass
class ConvergenceCriteriaABC(ABC):
    """
    Abstract class for the convergence criteria
    """

    @property
    @abc.abstractmethod
    def convergence_criterion(self):
        """
        Abstract property for returning the type of the convergence criterion

        Raises:
            - Exception: abstract class of convergence criteria is called
        """
        raise Exception("abstract class of convergence criteria is called")


@dataclass
class DisplacementConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the displacement convergence criteria

    Inheritance:
        - :class:`ConvergenceCriteriaABC`

    Attributes:
        - displacement_relative_tolerance (float): The relative tolerance for the displacement. Default value is 1e-4.
        - displacement_absolute_tolerance (float): The absolute tolerance for the displacement. Default values is 1e-9.

    """
    displacement_relative_tolerance: float = 1e-4
    displacement_absolute_tolerance: float = 1e-9

    @property
    def convergence_criterion(self):
        """
        Property for returning the type of the displacement convergence criterion

        Returns:
            - str: The type of the displacement convergence criterion

        """
        return "displacement_criterion"

@dataclass
class ResidualConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the residual convergence criteria

    Inheritance:
        - :class:`ConvergenceCriteriaABC`

    Attributes:
        - residual_relative_tolerance (float): The relative tolerance for the residual. Default value is 1e-4.
        - residual_absolute_tolerance (float): The absolute tolerance for the residual. Default value is 1e-9.

    """

    residual_relative_tolerance: float = 1e-4
    residual_absolute_tolerance: float = 1e-9

    @property
    def convergence_criterion(self):
        """
        Property for returning the type of the residual convergence criterion

        Returns:
            - str: The type of the residual convergence criterion

        """
        return "residual_criterion"


@dataclass
class WaterPressureConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the water pressure convergence criteria

    Inheritance:
        - :class:`ConvergenceCriteriaABC`

    Attributes:
        - water_pressure_relative_tolerance (float): The relative tolerance for the water pressure. Default value \
            is 1e-4.
        - water_pressure_absolute_tolerance (float): The absolute tolerance for the water pressure. Default value \
            is 1e-9.

    """

    water_pressure_relative_tolerance: float = 1e-4
    water_pressure_absolute_tolerance: float = 1e-9

    @property
    def convergence_criterion(self):
        """
        Property for returning the type of the water pressure convergence criterion

        Returns:
            - str: The type of the water pressure convergence criterion
        """
        return "water_pressure_criterion"

@dataclass
class DisplacementAndWaterPressureConvergenceCriteria(ConvergenceCriteriaABC):
    """
    Class containing information about the displacement and water pressure convergence criteria

    Inheritance:
        - :class:`ConvergenceCriteriaABC`

    Attributes:
        - displacement_relative_tolerance (float): The relative tolerance for the displacement. Default value is 1e-4.
        - displacement_absolute_tolerance (float): The absolute tolerance for the displacement. Default values is 1e-9.
        - water_pressure_relative_tolerance (float): The relative tolerance for the water pressure. Default value \
            is 1e-4.
        - water_pressure_absolute_tolerance (float): The absolute tolerance for the water pressure. Default value \
            is 1e-9.

    """

    displacement_relative_tolerance: float = 1e-4
    displacement_absolute_tolerance: float = 1e-9
    water_pressure_relative_tolerance: float = 1e-4
    water_pressure_absolute_tolerance: float = 1e-9

    @property
    def convergence_criterion(self):
        """
        Property for returning the type of the displacement and water pressure convergence criterion

        Returns:
            - str: The type of the displacement and water pressure convergence criterion
        """
        return "displacement_and_water_pressure_criterion"


@dataclass
class NewmarkScheme(SchemeABC):
    """
    Class containing information about the Newmark scheme

    Inheritance:
        - :class:`SchemeABC`

    Attributes:
        - newmark_beta (float): The beta parameter of the Newmark scheme. Default value is 0.25.
        - newmark_gamma (float): The gamma parameter of the Newmark scheme. Default value is 0.5.
        - newmark_theta (float): The theta parameter of the Newmark scheme, which is used for water pressure. Default\
            value is 0.5.
    """
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    newmark_theta: float = 0.5

    @property
    def scheme_type(self):
        """
        Property for returning the type of the Newmark scheme

        Returns:
            - str: The type of the newmark scheme
        """
        return "newmark"


@dataclass
class BackwardEulerScheme(SchemeABC):
    """
    Class containing information about the backward Euler scheme

    Inheritance:
        - :class:`SchemeABC`
    """

    @property
    def scheme_type(self):
        """
        Property for returning the type of the backward Euler scheme

        Returns:
            - str: The type of the backward Euler scheme
        """
        return "backward_euler"


class SolutionType(Enum):
    """
    Enum class containing the solution types

    Attributes:
        - QUASI_STATIC (int): quasi-static solution type
        - DYNAMIC (int): dynamic solution type
    """
    QUASI_STATIC = 1
    DYNAMIC = 2


class StressInitialisationType(Enum):
    """
    Enum class containing the stress initialisation types

    Attributes:
        - NONE (int): no stress initialisation
        - GRAVITY_LOADING (int): gravity loading stress initialisation
        - K0_PROCEDURE (int): K0-procedure stress initialisation
    """
    NONE = 1
    GRAVITY_LOADING = 2
    K0_PROCEDURE = 3


@dataclass
class StrategyTypeABC(ABC):
    """
    Abstract class for the strategy type
    """

    @property
    @abc.abstractmethod
    def strategy_type(self):
        """
        Abstract property for returning the type of the strategy

        Raises:
            - Exception: abstract class of strategy type is called
        """
        raise Exception("abstract class of strategy type is called")


@dataclass
class NewtonRaphsonStrategy(StrategyTypeABC):
    """
    Class containing information about the Newton-Raphson strategy

    Attributes:
        - max_iterations (int): maximum number of iterations allowed, if this number is reached, the time step size is\
            decreased and the algorithm is restarted. Default value is 15.
        - min_iterations (int): minimum number of iterations, below this number, the time step size is increased.\
            Default value is 6.
        - number_cycles (int): number of allowed cycles of decreasing the time step size until the algorithm is stopped.\
            Default value is 100.

    Inheritance:
        - :class:`StrategyTypeABC`
    """
    max_iterations: int = 15
    min_iterations: int = 6
    number_cycles: int = 100

    @property
    def strategy_type(self):
        """
        Returns the strategy type name of the Newton-Raphson strategy

        Returns:
            - str: strategy type name
        """
        return "newton_raphson"


@dataclass
class LineSearchStrategy(StrategyTypeABC):
    """
    Class containing information about the line search strategy

    Inheritance:
        - :class:`StrategyTypeABC`

    Attributes:
        - max_iterations (int): maximum number of iterations allowed, if this number is reached, the time step size is\
            decreased and the algorithm is restarted. Default value is 15.
        - min_iterations (int): minimum number of iterations, below this number, the time step size is increased.\
            Default value is 6.
        - number_cycles (int): number of allowed cycles of decreasing the time step size until the algorithm is stopped.\
            Default value is 100.
        - max_line_search_iterations (int): maximum number of line search iterations. Default value is 10.
        - first_alpha_value (float): first alpha guess value used for the first iteration. Default value is 1.0.
        - second_alpha_value (float): second alpha guess value used for the first iteration. Default value is 0.5.
        - min_alpha (float): minimum possible alpha value at the end of the algorithm. Default value is 1e-4.
        - max_alpha (float): maximum possible alpha value at the end of the algorithm. Default value is 1e4.
        - line_search_tolerance (float): Tolerance of the line search algorithm, defined as the ratio between maximum\
            residual*alpha*dx and current iteration residual*alpha*dx. Default value is 1e-4.
        - echo_level (int): echo level. Default value is 0.
    """
    max_iterations: int = 15
    min_iterations: int = 6
    number_cycles: int = 100
    max_line_search_iterations: int = 10
    first_alpha_value: float = 1.0
    second_alpha_value: float = 0.5
    min_alpha: float = 1e-4
    max_alpha: float = 1e4
    line_search_tolerance: float = 1e-4
    echo_level: int = 0

    @property
    def strategy_type(self):
        """
        Returns the strategy type name of the line search strategy

        Returns:
            - str: strategy type name
        """
        return "line_search"


@dataclass
class ArcLengthStrategy(StrategyTypeABC):
    """
    Class containing information about the arc length strategy

    Inheritance:
        - :class:`StrategyTypeABC`

    Attributes:
        - max_iterations (int): maximum number of iterations allowed, if this number is reached, the time step size is\
            decreased and the algorithm is restarted. Default value is 15.
        - min_iterations (int): minimum number of iterations, below this number, the time step size is increased.\
            Default value is 6.
        - number_cycles (int): number of allowed cycles of decreasing the time step size until the algorithm is stopped.\
            Default value is 100.
        - desired_iterations (int): This is used to calculate the radius of the next step. Default value is 10.
        - max_radius_factor (float): maximum radius factor of the arc. Default value is 1.0.
        - min_radius_factor (float): minimum radius factor of the arc. Default value is 0.1.

    """
    max_iterations: int = 15
    min_iterations: int = 6
    number_cycles: int = 100
    desired_iterations: int = 10
    max_radius_factor: float = 1.0
    min_radius_factor: float = 0.1

    @property
    def strategy_type(self):
        """
        Returns the strategy type name of the arc length strategy

        Returns:
            - str: strategy type name

        """
        return "arc_length"


@dataclass
class LinearSolverSettingsABC(ABC):
    """
    Class containing information about the linear solver settings
    """

    @property
    @abc.abstractmethod
    def solver_type(self):
        """
        Abstract property for returning the solver type

        Raises:
            - Exception: abstract class of linear solver settings is called

        """
        raise Exception("abstract class of linear solver settings is called")


@dataclass
class Amgcl(LinearSolverSettingsABC):
    """
    Class containing information about the amgcl linear solver settings

    Inheritance:
        - :class:`LinearSolverSettingsABC`

    Attributes:
        - scaling (bool): if true, the system matrix will be scaled before solving the linear system of equations.\
            Default value is False.
        - tolerance (float): tolerance for the linear solver convergence criteria. Default value is 1e-6.
        - max_iteration (int): maximum number of iterations for the linear solver. Default value is 1000.

    """
    scaling: bool = False
    tolerance: float = 1e-6
    max_iteration: int = 1000

    @property
    def solver_type(self):
        """
        Property for returns the solver type name of the amgcl linear solver settings

        Returns:
            - str: solver type name

        """
        return "amgcl"


@dataclass
class TimeIntegration:
    """
    Class containing information about the time integration

    Attributes:
        - start_time (float): start time of the analysis
        - end_time (float): end time of the analysis
        - delta_time (float): initial time step
        - reduction_factor (float): factor used to reduce the time step when the solution diverges
        - increase_factor (float): factor used to increase the time step when the solution converges within the minimum\
            number of iterations
        - max_delta_time_factor (float): maximum time step factor, used to limit the time step increase. Default value\
            is 1000.

    """

    start_time: float
    end_time: float
    delta_time: float

    reduction_factor: float
    increase_factor: float
    max_delta_time_factor: float = 1000


class AnalysisType(Enum):
    """
    Enum class containing the analysis type

    Attributes:
        - MECHANICAL_GROUNDWATER_FLOW (int): coupled mechanical and groundwater flow analysis
        - MECHANICAL (int): mechanical analysis
        - GROUNDWATER_FLOW (int): groundwater flow analysis
    """
    MECHANICAL_GROUNDWATER_FLOW = 1
    MECHANICAL = 2
    GROUNDWATER_FLOW = 3


@dataclass
class SolverSettings:
    """
    Class containing information about the time integration, builder, strategy, scheme and linear solver.

    Attributes:
        - solution_type (:class:`SolutionType`): solution type, QUASI_STATIC or DYNAMIC
        - stress_initialisation_type (:class:`StressInitialisationType`): stress initialisation type, \
            NONE, GRAVITY_LOADING OR K0_PROCEDURE
        - time_integration (:class:`TimeIntegration`): time integration settings
        - is_stiffness_matrix_constant (bool): if true, the lhs matrix is only built once, else, the lhs matrix is \
            rebuilt at each non-linear iteration
        - are_mass_and_damping_constant (bool): if true, the mass and damping matrices are prebuilt and directly used \
            to calculate the rhs. If false, the mass and damping matrices are built at each non-linear iteration \
            for calculating the rhs and possibly the lhs
        - convergence_criteria (:class:`ConvergenceCriteriaABC`): convergence criteria, \
            :class:`DisplacementConvergenceCriteria`, :class:`ResidualConvergenceCriteria`, \
            :class:`WaterPressureConvergenceCriteria` or :class:`DisplacementAndWaterPressureConvergenceCriteria`
        - reset_displacements (bool): if true, the displacements are reset at the beginning of the phase
        - calculate_stresses_on_nodes (bool): if true, the stresses are also calculated on the nodes and not only on \
            the gauss points. Default value is True.
        - strategy_type (:class:`StrategyTypeABC`): strategy type, :class:`NewtonRaphsonStrategy`, \
            :class:`LineSearchStrategy` or :class:`ArcLengthStrategy`. Default value is :class:`NewtonRaphsonStrategy`.
        - scheme (:class:`SchemeABC`): scheme, :class:`NewmarkSceme` or :class:`BackwardEulerScheme`. Default value \
            is :class:`NewmarkSceme`.
        - linear_solver_settings (:class:`LinearSolverSettingsABC`): linear solver settings, currently only \
            :class:`Amgcl` is supported
        - rayleigh_m (Optional[float]): mass proportional damping parameter
        - rayleigh_k (Optional[float]): stiffness proportional damping parameter
        - echo_level (int): echo level. Default value is 1. If 0, only time information is printed. If 1, time \
            information and convergence information are printed. If 2, time information, convergence information, \
            intermediate rhs results and linear solver settings are printed.
    """
    analysis_type: AnalysisType
    solution_type: SolutionType
    stress_initialisation_type: StressInitialisationType
    time_integration: TimeIntegration

    is_stiffness_matrix_constant: bool
    are_mass_and_damping_constant: bool

    convergence_criteria: ConvergenceCriteriaABC
    reset_displacements: bool = False
    calculate_stresses_on_nodes: bool = True
    strategy_type: StrategyTypeABC = field(default_factory=NewtonRaphsonStrategy)
    scheme: SchemeABC = field(default_factory=NewmarkScheme)
    linear_solver_settings: LinearSolverSettingsABC = field(default_factory=Amgcl)
    rayleigh_m: Optional[float] = None
    rayleigh_k: Optional[float] = None
    echo_level: int = 1

    def __post_init__(self):
        """
        Post initialization method

        Raises:
            - ValueError: if the Rayleigh damping parameters are not provided for dynamic analysis
            - ValueError: if the K0-procedure is selected for dynamic analysis
        """
        if self.solution_type == SolutionType.DYNAMIC:
            if self.rayleigh_m is None or self.rayleigh_k is None:
                raise ValueError("Rayleigh damping parameters must be provided for dynamic analysis")

            if self.stress_initialisation_type == StressInitialisationType.K0_PROCEDURE:
                raise ValueError("Kratos Multiphysics does not support the K0-procedure for dynamic analysis")


@dataclass
class Problem:
    """
    Class containing information about the problem settings and the solver settings

    Attributes:
        - problem_name (str): name of the problem
        - number_of_threads (int): number of threads used for the analysis
        - settings (:class:`SolverSettings`): dictionary containing the solver settings

    """

    problem_name: str
    number_of_threads: int
    settings: SolverSettings

