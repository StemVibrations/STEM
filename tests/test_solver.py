import pytest

from stem.solver import *


class TestSolverSettings:

    @pytest.mark.parametrize(
        "strategy_type", [NewtonRaphsonStrategy, LinearNewtonRaphsonStrategy, LineSearchStrategy, ArcLengthStrategy])
    def test_strategy(self, strategy_type):
        """
        Test strategies for solving the problem.
        """

        analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
        solution_type = SolutionType.DYNAMIC
        time_integration = TimeIntegration(start_time=0.0,
                                           end_time=0.15,
                                           delta_time=0.0025,
                                           reduction_factor=1.0,
                                           increase_factor=1.0,
                                           max_delta_time_factor=1000)
        convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-12,
                                                                displacement_absolute_tolerance=1.0E-6)
        stress_initialisation_type = StressInitialisationType.NONE

        scheme = NewmarkScheme()
        strategy = strategy_type()

        solver_settings = SolverSettings(analysis_type=analysis_type,
                                         solution_type=solution_type,
                                         stress_initialisation_type=stress_initialisation_type,
                                         time_integration=time_integration,
                                         is_stiffness_matrix_constant=True,
                                         are_mass_and_damping_constant=True,
                                         strategy_type=strategy,
                                         scheme=scheme,
                                         convergence_criteria=convergence_criterion,
                                         rayleigh_k=6e-6,
                                         rayleigh_m=0.02)

        # check if default strategy is LinearNewtonRaphson
        assert isinstance(solver_settings.strategy_type, strategy_type)

    def test_initialise_instance_and_validate(self):
        """
        Test that the solver settings are initialised correctly and validated correctly.

        """
        analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
        solution_type = SolutionType.DYNAMIC
        time_integration = TimeIntegration(start_time=0.0,
                                           end_time=0.15,
                                           delta_time=0.0025,
                                           reduction_factor=1.0,
                                           increase_factor=1.0,
                                           max_delta_time_factor=1000)
        convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-12,
                                                                displacement_absolute_tolerance=1.0E-6)
        stress_initialisation_type = StressInitialisationType.NONE

        strategy = NewtonRaphsonStrategy()
        scheme = NewmarkScheme()

        solver_settings = SolverSettings(analysis_type=analysis_type,
                                         solution_type=solution_type,
                                         stress_initialisation_type=stress_initialisation_type,
                                         time_integration=time_integration,
                                         is_stiffness_matrix_constant=True,
                                         are_mass_and_damping_constant=True,
                                         strategy_type=strategy,
                                         scheme=scheme,
                                         convergence_criteria=convergence_criterion,
                                         rayleigh_k=6e-6,
                                         rayleigh_m=0.02)

        # check if validated settings have not changed compared to the input
        assert isinstance(solver_settings.scheme, NewmarkScheme)
        assert solver_settings.are_mass_and_damping_constant

        # check if strategy is NewtonRaphson
        assert isinstance(solver_settings.strategy_type, NewtonRaphsonStrategy)

        # set solution type to quasi-static
        solver_settings.solution_type = SolutionType.QUASI_STATIC

        # validate settings
        solver_settings.validate_settings()

        # check if settings are updated
        assert isinstance(solver_settings.scheme, BackwardEulerScheme)
        assert not solver_settings.are_mass_and_damping_constant

        # set solution type to dynamic and remove Rayleigh damping parameter
        solver_settings.solution_type = SolutionType.DYNAMIC
        solver_settings.rayleigh_m = None

        # check if error is raised correctly
        with pytest.raises(ValueError, match="Rayleigh damping parameters must be provided for dynamic analysis"):
            solver_settings.validate_settings()

        # set stress initialisation type to K0-procedure
        solver_settings.rayleigh_m = 0.02
        solver_settings.stress_initialisation_type = StressInitialisationType.K0_PROCEDURE

        # check if error is raised correctly
        with pytest.raises(ValueError,
                           match="Kratos Multiphysics does not support the K0-procedure for dynamic analysis"):
            solver_settings.validate_settings()

    def test_solver_type(self):
        """
        Test that the solver type name is returned correctly.

        """

        amgcl_linear_solver = Amgcl()
        assert amgcl_linear_solver.solver_type == "amgcl"

        cg_linear_solver = Cg()
        assert cg_linear_solver.solver_type == "cg"

        lu_linear_solver = Lu()
        assert lu_linear_solver.solver_type == "LinearSolversApplication.sparse_lu"
