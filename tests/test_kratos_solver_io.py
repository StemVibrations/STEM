import json

from stem.IO.kratos_solver_io import KratosSolverIO
from stem.solver import *
from stem.model_part import ModelPart, BodyModelPart

from tests.utils import TestUtils


class TestKratosSolverIO:

    def test_create_settings_dictionary(self):
        """
        Test the creation of the problem data and solver settings dictionary. This test compares a created dictionary
        with a reference dictionary.

        """

        # set up solver settings
        analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW

        solution_type = SolutionType.DYNAMIC

        time_integration = TimeIntegration(start_time=0.0, end_time=1.0, delta_time=0.1, reduction_factor=0.5,
                                           increase_factor=2.0, max_delta_time_factor=500)

        convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1e-5,
                                                                displacement_absolute_tolerance=1e-7)

        strategy_type = NewtonRaphsonStrategy(min_iterations=5, max_iterations=30, number_cycles=50)

        scheme_type = NewmarkScheme(newmark_beta=0.35, newmark_gamma=0.4, newmark_theta=0.6)

        linear_solver_settings = Amgcl(tolerance=1e-8, max_iteration=500, scaling=True)

        stress_initialisation_type = StressInitialisationType.NONE

        solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                         stress_initialisation_type=stress_initialisation_type,
                                         time_integration=time_integration,
                                         is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                         convergence_criteria=convergence_criterion,
                                         strategy_type=strategy_type, scheme=scheme_type,
                                         linear_solver_settings=linear_solver_settings, rayleigh_k=0.001,
                                         rayleigh_m=0.001)

        # set up problem data
        problem_data = Problem(problem_name="test", number_of_threads=2, settings=solver_settings)

        # create model parts
        model_part1 = ModelPart("ModelPart1")

        body_model_part1 = BodyModelPart("BodyModelPart1")

        model_parts = [model_part1, body_model_part1]

        # create solver IO
        solver_io = KratosSolverIO(3, "testDomain")

        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name", "material_test_name.json",
                                                         model_parts)

        # open expected settings dictionary
        with open("tests/test_data/expected_solver_settings.json") as f:
            expected_solver_settings = json.load(f)

        # assert that the settings dictionary is as expected
        TestUtils.assert_dictionary_almost_equal(expected_solver_settings, test_dict)






