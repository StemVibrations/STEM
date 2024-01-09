import json

import pytest

from stem.IO.kratos_solver_io import KratosSolverIO
from stem.model_part import ModelPart, BodyModelPart
from stem.solver import *
from stem.load import UvecLoad
from tests.utils import TestUtils


class TestKratosSolverIO:

    @pytest.fixture()
    def set_solver_settings(self) -> SolverSettings:
        """
        Set up solver settings for testing.

        Returns:
            - SolverSettings: solver settings for testing
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

        return solver_settings

    def test_create_settings_dictionary(self, set_solver_settings: SolverSettings):
        """
        Test the creation of the problem data and solver settings dictionary. This test compares a created dictionary
        with a reference dictionary.

        Args:
            - set_solver_settings (SolverSettings): solver settings for testing

        """

        solver_settings = set_solver_settings

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

        # check variants of the settings dictionary
        # 1. analysis type = MECHANICAL
        problem_data.settings.analysis_type = AnalysisType.MECHANICAL
        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name", "material_test_name.json",
                                                         model_parts)

        assert test_dict["solver_settings"]["solver_type"] == "U_Pw"

        # 2. analysis type = GROUNDWATER_FLOW
        problem_data.settings.analysis_type = AnalysisType.GROUNDWATER_FLOW
        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name", "material_test_name.json",
                                                         model_parts)

        assert test_dict["solver_settings"]["solver_type"] == "Pw"

        # 3. solution type = STATIC
        problem_data.settings.solution_type = SolutionType.QUASI_STATIC

        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name", "material_test_name.json",
                                                         model_parts)

        assert test_dict["solver_settings"]["solution_type"] == "quasi_static"

    def test_create_settings_dictionary_with_uvec(self, set_solver_settings: SolverSettings):
        """
        Test the creation of the problem data and solver settings dictionary including uvec data.
        This test compares a created dictionary with a reference dictionary.

        Args:
            - set_solver_settings (SolverSettings): solver settings for testing

        """

        solver_settings = set_solver_settings

        # set up uvec model part
        uvec_model_part = ModelPart("UvecModelPart")

        # set up uvec load
        uvec_parameters = {"load_wheel_1": -10.0, "load_wheel_2": -20.0}
        uvec_state_variables = {"state_1": [0.0, 1.0], "state_2": [9, 8]}
        uvec_load = UvecLoad(direction=[1, 1, 0], velocity=5, origin=[0.0, 1.0, 0.0], wheel_configuration=[0.0, 2.0],
                             uvec_file=r"sample_uvec.py", uvec_function_name="uvec_test",
                             uvec_parameters=uvec_parameters, uvec_state_variables=uvec_state_variables)

        uvec_model_part.parameters = uvec_load

        model_parts = [uvec_model_part]

        # set up problem data
        problem_data = Problem(problem_name="test", number_of_threads=2, settings=solver_settings)

        # create solver IO
        solver_io = KratosSolverIO(3, "testDomain")

        # create settings dictionary
        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name",
                                                         "material_test_name.json", model_parts)

        # open expected settings dictionary
        with open("tests/test_data/expected_solver_settings_with_uvec.json") as f:
            expected_solver_settings = json.load(f)

        # assert that the settings dictionary is as expected
        TestUtils.assert_dictionary_almost_equal(expected_solver_settings, test_dict)

    def test_number_of_cycles(self, set_solver_settings: SolverSettings):
        """
        Test if the number of cycles is set correctly in the settings dictionary. Firsty, the number of cycles is set
        to 50. Secondly, the reduction factor is set to 1.0 which should result in a single cycle.

        Args:
            - set_solver_settings (SolverSettings): solver settings for testing

        """

        solver_settings = set_solver_settings

        # set up problem data
        problem_data = Problem(problem_name="test", number_of_threads=2, settings=solver_settings)

        # create model parts
        model_part1 = ModelPart("ModelPart1")

        body_model_part1 = BodyModelPart("BodyModelPart1")

        model_parts = [model_part1, body_model_part1]

        # create solver IO
        solver_io = KratosSolverIO(3, "testDomain")

        # create settings dictionary
        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name", "material_test_name.json",
                                                         model_parts)

        # assert that the number of cycles is as expected
        assert test_dict["solver_settings"]["number_cycles"] == 50

        # set reduction factor to 1.0
        solver_settings.time_integration.reduction_factor = 1.0

        # create settings dictionary
        test_dict = solver_io.create_settings_dictionary(problem_data, "mesh_test_name", "material_test_name.json",
                                                         model_parts)

        # assert that the number of cycles is as expected
        assert test_dict["solver_settings"]["number_cycles"] == 1
