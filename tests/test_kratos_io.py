import json
from typing import List

import numpy.testing as npt
from stem.load import LineLoad
from stem.model import Model
from stem.model_part import *
from stem.output import NodalOutput, GaussPointOutput, GiDOutputParameters, Output
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from gmsh_utils import gmsh_IO

import pytest

from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from tests.utils import TestUtils
from stem.IO.kratos_io import KratosIO


class TestKratosModelIO:

    @pytest.fixture(autouse=True)
    def close_gmsh(self):
        """
        Initializer to close gmsh if it was not closed before. In case a test fails, the destroyer method is not called
        on the Model object and gmsh keeps on running. Therefore, nodes, lines, surfaces and volumes ids are not
        reset to one. This causes also the next test after the failed one to fail as well, which has nothing to do
        the test itself.

        Returns:
            - None

        """
        gmsh_IO.GmshIO().finalize_gmsh()

    @pytest.fixture
    def create_default_2d_model_and_mesh(self):
        """
        Sets expected geometry data for a 3D geometry group. The group is a geometry of a cube.

        Returns:
            - :class:`stem.model.Model`: the default 2D model of a square soil layer and a line load.
        """
        ndim=2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        load_coordinates = [layer_coordinates[2], layer_coordinates[3]]
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        # define load properties
        line_load = LineLoad(active=[False, True, False], value=[0, -20, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")
        model.add_load_by_coordinates(load_coordinates, line_load, "load1")
        model.synchronise_geometry()

        model.set_mesh_size(1)
        model.generate_mesh()

        return model

    @pytest.fixture
    def create_default_outputs(self):
        """
        Sets default output parameters.

        Returns:
            - List[:class:`stem.output.Output`]: list of default output processes.
        """
        # Nodal results
        nodal_results = [NodalOutput.DISPLACEMENT]
        # Gauss point results
        gauss_point_results = [
            GaussPointOutput.GREEN_LAGRANGE_STRAIN_TENSOR,
            GaussPointOutput.CAUCHY_STRESS_TENSOR
        ]
        # define output process

        gid_output_process = Output(
            part_name="soil1",
            output_name="gid_output_soil1",
            output_dir="dir_test",
            output_parameters=GiDOutputParameters(
                file_format="binary",
                output_interval=100,
                nodal_results=nodal_results,
                gauss_point_results=gauss_point_results,
            )
        )

        return [gid_output_process]

    @pytest.fixture
    def create_default_solver_settings(self):
        """
        Sets default output parameters.

        Returns:
            - :class:`stem.solver.Problem`: the Problem object containing the solver settings.
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
        linear_solver_settings = Amgcl(tolerance=1e-8, max_iterations=500, scaling=True)
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
        return Problem(problem_name="test", number_of_threads=2, settings=solver_settings)

    def test_write_project_parameters_json(
        self,
        create_default_2d_model_and_mesh:Model,
        create_default_outputs:List[Output],
        create_default_solver_settings:Problem
    ):
        """
        Test correct writing of the project parameters for the default output, model and settings.

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_and_mesh
        kratos_io = KratosIO(ndim=model.ndim)
        model.project_parameters = create_default_solver_settings

        actual_dict = kratos_io.write_project_parameters_json(
            model=model,
            outputs=create_default_outputs,
            mesh_file_name="test_mdpa_file.mdpa",
            materials_file_name="MaterialParameters.json",
            output_folder="dir_test"
        )
        expected_dict = json.load(open("tests/test_data/expected_ProjectParameters.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_material_parameters_json(
        self,
        create_default_2d_model_and_mesh:Model
    ):
        """
        Test correct writing of the material parameters for the default model.

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
        """
        model = create_default_2d_model_and_mesh
        kratos_io = KratosIO(ndim=model.ndim)

        actual_dict = kratos_io.write_material_parameters_json(model=model, output_folder="dir_test")
        expected_dict = json.load(open("tests/test_data/expected_MaterialParameters.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_mdpa_file(
        self,
        create_default_2d_model_and_mesh:Model,
        create_default_solver_settings:Problem
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings.

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_and_mesh
        kratos_io = KratosIO(ndim=model.ndim)
        model.project_parameters = create_default_solver_settings

        actual_text = kratos_io.write_mesh_to_mdpa(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa",
            output_folder="dir_test"
        )
        with open('tests/test_data/expected_mdpa_file.mdpa', 'r') as openfile:
            expected_text = openfile.readlines()

        npt.assert_equal(actual=actual_text, desired=expected_text)