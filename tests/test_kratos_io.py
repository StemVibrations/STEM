import sys
import os

import json
from typing import List
import re
from shutil import rmtree

import numpy as np
import numpy.testing as npt
import pytest
from gmsh_utils import gmsh_IO

from stem.IO.io_utils import IOUtils
from stem.IO.kratos_io import KratosIO
from stem.additional_processes import ParameterFieldParameters
from stem.boundary import DisplacementConstraint
from stem.field_generator import RandomFieldGenerator
from stem.load import LineLoad, SurfaceLoad, MovingLoad
from stem.model import Model
from stem.model_part import *
from stem.output import NodalOutput, GaussPointOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated, EulerBeam
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.table import Table
from tests.utils import TestUtils

IS_LINUX = sys.platform == "linux"


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
    def create_default_2d_model(self) -> Model:
        """
        Sets expected geometry data for a 2D geometry group. And it sets a time dependent line load at the top and
        bottom and another line load at the sides. The group is a geometry of a square.

        Returns:
            - :class:`stem.model.Model`: the default 2D model of a square soil layer and line loads.
        """
        ndim = 2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        load_coordinates_top = [(1, 1, 0), (0, 1, 0)]  # top
        load_coordinates_bottom = [(0, 0, 0), (1, 0, 0)]  # bottom
        load_coordinates_left = [(0, 1, 0), (0, 0, 0)]  # left
        load_coordinates_right = [(1, 0, 0), (1, 1, 0)]  # right
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        # define tables
        time = np.arange(6)*0.5
        values = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(times=time, values=values)
        table2 = Table(times=time, values=-values)

        # define load properties
        line_load1 = LineLoad(active=[False, True, False], value=[0, table1, 0])
        line_load2 = LineLoad(active=[True, False, False], value=[table2, 0, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        model.add_load_by_coordinates(load_coordinates_top, line_load1, "load_top")
        model.add_load_by_coordinates(load_coordinates_bottom, line_load1, "load_bottom")
        model.add_load_by_coordinates(load_coordinates_left, line_load2, "load_left")
        model.add_load_by_coordinates(load_coordinates_right, line_load2, "load_right")

        # add pin parameters
        no_displacement_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                            value=[0, 0, 0])

        # add boundary conditions in 0d, 1d and 2d
        model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "no_displacement")

        model.synchronise_geometry()

        return model

    @pytest.fixture
    def create_default_2d_model_and_output_by_coordinates(self) -> Model:
        """
        Sets expected geometry data for a 2D geometry group. And it sets a time dependent line load at the top and
        outputs by coordinate so to include the center of the square. The group is a geometry of a square.
        Output is provided along a line.

        Returns:
            - :class:`stem.model.Model`: the default 2D model of a square soil layer, line loads and output on a line.
        """
        ndim = 2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        output_coordinates = [(0.5, 0, 0), (0.5, 0.5, 0), (0.5, 1, 0)]

        load_coordinates_top = [(1, 1, 0), (0, 1, 0)]  # top
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        # define tables
        _time = np.arange(6)*0.5
        _value1 = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(times=_time, values=_value1)

        # define load properties
        line_load1 = LineLoad(active=[False, True, False], value=[0, table1, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        model.add_load_by_coordinates(load_coordinates_top, line_load1, "load_top")

        # add pin parameters
        no_displacement_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                            value=[0, 0, 0])

        # add boundary conditions in 0d, 1d and 2d
        model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "no_displacement")

        # add output
        # - Nodal results
        nodal_results = [NodalOutput.ACCELERATION, NodalOutput.VELOCITY, NodalOutput.DISPLACEMENT]
        # - define output process
        model.add_output_settings_by_coordinates(
            coordinates=output_coordinates,
            part_name="line_output_soil1",
            output_name="json_line_output_soil1",
            output_dir="dir_test",
            output_parameters=JsonOutputParameters(
                output_interval=100,
                nodal_results=nodal_results
            )
        )

        model.synchronise_geometry()

        return model

    @pytest.fixture
    def create_default_2d_model_randomfield(self) -> Model:
        """
        Sets expected geometry data for a 2D geometry group. And it sets a time dependent line load at the top and
        bottom and another line load at the sides. The group is a geometry of a square.

        Returns:
            - :class:`stem.model.Model`: the default 2D model of a square soil layer and line loads.
        """
        ndim = 2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        load_coordinates_top = [(1, 1, 0), (0, 1, 0)]  # top
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        # define load properties
        line_load1 = LineLoad(active=[False, True, False], value=[0, -500, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        model.add_load_by_coordinates(load_coordinates_top, line_load1, "load_top")

        # add pin parameters
        no_displacement_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                            value=[0, 0, 0])

        # add boundary conditions in 0d, 1d and 2d
        model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "no_displacement")

        # Define the field generator
        random_field_generator = RandomFieldGenerator(
            n_dim=3, cov=0.1, model_name="Gaussian",
            v_scale_fluctuation=1, anisotropy=[0.5, 0.5], angle=[0, 0]
        )

        field_parameters_json = ParameterFieldParameters(
            property_name="YOUNG_MODULUS",
            function_type="json_file",
            field_generator=random_field_generator
        )

        model.add_field(part_name="soil1", field_parameters=field_parameters_json)

        model.synchronise_geometry()

        return model

    @pytest.fixture
    def create_default_3d_model_and_mesh(self):
        """
        Sets expected geometry data for a 3D geometry group. It sets a surface load on the bottom and another load
        at the top. The group is a geometry of a cube.

        Returns:
            - :class:`stem.model.Model`: the default 3D model of a cube soil and a surface load.
        """
        ndim = 3
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        load_coordinates_bottom = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        load_coordinates_top = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        # define tables
        _time = np.arange(6)*0.5
        _value1 = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(times=_time, values=-_value1)
        table2 = Table(times=_time, values=_value1)

        # define load properties

        # define load properties
        surface_load_top = SurfaceLoad(active=[False, True, False], value=[0, table1, 0])
        surface_load_bottom = SurfaceLoad(active=[False, True, False], value=[0, table2, 0])

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")
        model.add_load_by_coordinates(load_coordinates_top, surface_load_top, "load_top")
        model.add_load_by_coordinates(load_coordinates_bottom, surface_load_bottom, "load_bottom")

        # add pin parameters
        no_displacement_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                            value=[0, table2, 0])

        # add boundary conditions in 0d, 1d and 2d
        model.add_boundary_condition_by_geometry_ids(2, [6], no_displacement_parameters, "no_displacement")

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

        vtk_output_process = Output(
            part_name="soil1",
            output_name="vtk_output_soil1",
            output_dir="dir_test",
            output_parameters=VtkOutputParameters(
                file_format="ascii",
                output_interval=100,
                nodal_results=nodal_results,
                gauss_point_results=gauss_point_results,
            )
        )

        return vtk_output_process

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
        linear_solver_settings = Amgcl(tolerance=1e-8, max_iteration=500, scaling=False)
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
        create_default_2d_model: Model,
        create_default_outputs: List[Output],
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the project parameters for the default output, model and settings.

        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model

        model.project_parameters = create_default_solver_settings
        model.add_output_settings(**create_default_outputs.__dict__)

        model.post_setup()

        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        actual_dict = kratos_io._KratosIO__write_project_parameters_json(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa",
            materials_file_name="MaterialParameters.json",
        )
        expected_dict = json.load(open("tests/test_data/expected_ProjectParameters.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_project_parameters_json_for_line_output(
        self,
        create_default_2d_model_and_output_by_coordinates: Model,
        create_default_outputs: List[Output],
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the project parameters for the default output, model and settings.

        Args:
            - create_default_2d_model_and_output_by_coordinates (:class:`stem.model.Model`): the default 2D model of \
                a square soil layer, a line load and a line output.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_and_output_by_coordinates

        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings

        actual_dict = kratos_io._KratosIO__write_project_parameters_json(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa",
            materials_file_name="MaterialParameters.json",
        )
        expected_dict = json.load(open("tests/test_data/expected_ProjectParameters_line_output.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_random_field_json_file(
        self,
        create_default_2d_model_randomfield: Model,
        create_default_outputs: List[Output],
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the random field parameters.

        Args:
            - create_default_2d_model_randomfield (:class:`stem.model.Model`): the default 2D model of a \
                square soil layer, a line load and a random field for the Young's modulus.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_randomfield

        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)

        model.project_parameters = create_default_solver_settings
        model.add_output_settings(**create_default_outputs.__dict__)

        kratos_io._KratosIO__write_project_parameters_json(
            model=model, mesh_file_name="test_mdpa_file.mdpa", materials_file_name="MaterialParameters.json"
        )

        expected_random_field_values = json.load(open("tests/test_data/expected_json_soil1_young_modulus_field.json",
                                                      'r'))

        actual_random_field_values = json.load(open("soil1_young_modulus_field.json", 'r'))

        # check number of values is equal to the number of elements
        assert len(actual_random_field_values["values"]) == len(model.body_model_parts[0].mesh.elements)

        TestUtils.assert_dictionary_almost_equal(expected_random_field_values, actual_random_field_values)

    def test_validation_of_random_field_parameters(
        self,
        create_default_2d_model_randomfield: Model,
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the random field parameters.

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load and a random field for the Young's modulus.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_randomfield
        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)

        model.project_parameters = create_default_solver_settings

        kratos_io._KratosIO__write_project_parameters_json(
            model=model, mesh_file_name="test_mdpa_file.mdpa", materials_file_name="MaterialParameters.json"
        )

        expected_random_field_values = json.load(open("tests/test_data/expected_json_soil1_young_modulus_field.json",
                                                      'r'))

        actual_random_field_values = json.load(open("soil1_young_modulus_field.json", 'r'))

        # check number of values is equal to the number of elements
        assert len(actual_random_field_values["values"]) == len(model.body_model_parts[0].mesh.elements)

        TestUtils.assert_dictionary_almost_equal(expected_random_field_values, actual_random_field_values)

    def test_write_project_parameters_json_for_randomfield(
        self,
        create_default_2d_model_randomfield: Model,
        create_default_outputs: List[Output],
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the project parameters for the default output, model and settings.

        Args:
            - create_default_2d_model_randomfield (:class:`stem.model.Model`): the default 2D model of \
                a square soil layer, a line load and a random field for the Young's modulus.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_randomfield
        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings

        actual_dict = kratos_io._KratosIO__write_project_parameters_json(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa",
            materials_file_name="MaterialParameters.json",
        )
        expected_dict = json.load(open("tests/test_data/expected_ProjectParameters_random_field_2d.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_material_parameters_json(
        self,
        create_default_2d_model: Model
    ):
        """
        Test correct writing of the material parameters for the default model.
        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
        """
        model = create_default_2d_model

        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        actual_dict = kratos_io._KratosIO__write_material_parameters_json(model=model)
        expected_dict = json.load(open("tests/test_data/expected_MaterialParameters.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_mdpa_file_2d(
        self,
        create_default_2d_model: Model,
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings in 2D.

        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model
        model.project_parameters = create_default_solver_settings

        model.post_setup()

        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings

        actual_text = kratos_io._KratosIO__write_mesh_to_mdpa(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa"
        )
        with open('tests/test_data/expected_mdpa_file.mdpa', 'r') as openfile:
            expected_text = openfile.readlines()

        npt.assert_equal(actual=actual_text, desired=expected_text)

    def test_write_mdpa_file_with_line_output(
        self,
        create_default_2d_model_and_output_by_coordinates: Model,
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings in 2D.

        Args:
            - create_default_2d_model_and_output_by_coordinates (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_and_output_by_coordinates
        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings

        actual_text = kratos_io._KratosIO__write_mesh_to_mdpa(
            model=model,
            mesh_file_name="mdpa_file_with_line_output_2d.mdpa"
        )
        with open('tests/test_data/expected_mdpa_file_with_line_output_2d.mdpa', 'r') as openfile:
            expected_text = openfile.readlines()

        npt.assert_equal(actual=actual_text, desired=expected_text)

    def test_write_mdpa_file_with_random_field(
        self,
        create_default_2d_model_randomfield: Model,
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings in 2D.

        Args:
            - create_default_2d_model_randomfield (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer, a line load and a random field for the Young's modulus.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model_randomfield
        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings

        actual_text = kratos_io._KratosIO__write_mesh_to_mdpa(
            model=model,
            mesh_file_name="mdpa_file_with_random_field_2d.mdpa"
        )
        with open('tests/test_data/expected_mdpa_file_with_random_field_2d.mdpa', 'r') as openfile:
            expected_text = openfile.readlines()

        npt.assert_equal(actual=actual_text, desired=expected_text)

    def test_write_mdpa_file_3d(
        self,
        create_default_3d_model_and_mesh: Model,
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings in 3D.

        Args:
            - create_default_3d_model_and_mesh (:class:`stem.model.Model`): the default 3D model of a cube \
                soil and a surface load on top.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_3d_model_and_mesh
        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings

        actual_text = kratos_io._KratosIO__write_mesh_to_mdpa(
            model=model,
            mesh_file_name="test_mdpa_file_3d.mdpa"
        )

        if IS_LINUX:
            with open('tests/test_data/expected_mdpa_file_3d_linux.mdpa', 'r') as openfile:
                expected_text = openfile.readlines()
        else:
            with open('tests/test_data/expected_mdpa_file_3d.mdpa', 'r') as openfile:
                expected_text = openfile.readlines()

        npt.assert_equal(actual=actual_text, desired=expected_text)

    def test_table_validation(self, create_default_solver_settings: Problem):
        """
        Test that the initialisation of the table ids raises the correct errors.

        Args:
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.

        """
        ndim = 2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        load_coordinates_top = [(1, 1, 0), (0, 1, 0)]  # top
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        # define tables
        time = np.arange(6)*0.5
        values = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(times=time, values=values)

        # define load properties
        line_load1 = LineLoad(active=[False, True, False], value=["hello world", table1, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")
        model.add_load_by_coordinates(load_coordinates_top, line_load1, "load_top")

        kratos_io = KratosIO(ndim=model.ndim)
        kratos_io.project_folder = "dir_test"

        model.project_parameters = create_default_solver_settings
        model.synchronise_geometry()

        model.set_mesh_size(1)
        model.generate_mesh()

        # string is not an accepted input for the load values
        msg = ("'value' attribute in LineLoad in model part `load_top`. The value (hello world) is a `str` object but "
               "only a Table, float or integer are valid inputs.")
        with pytest.raises(ValueError, match=re.escape(msg)):

            IOUtils.create_value_and_table("load_top", line_load1)

        # create value and table from moving load raises an error
        moving_load_parameters = MovingLoad(
            origin=[0, 1, 0.5],
            load=[0.0, -10.0, 0.0],
            velocity=5.0,
            offset=3.0,
            direction=[1, 1, 1]
        )
        msg = "Attribute `value` does not exist in class: MovingLoad."
        with pytest.raises(ValueError, match=msg):

            IOUtils.create_value_and_table("moving_load", moving_load_parameters)

        # adjust table to the right parameter, but not initialised table. Raises ValueError.
        model.process_model_parts[0].parameters.value[0] = 0
        msg = "Table id is not initialised for values in LineLoad in model part: load_top."
        with pytest.raises(ValueError, match=msg):

            IOUtils.create_value_and_table(model.process_model_parts[0].name, model.process_model_parts[0].parameters)

    def test_write_input_files_for_kratos(
        self,
        create_default_2d_model: Model,
        create_default_solver_settings: Problem,
        create_default_outputs: List[Output]
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings.

        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer line loads.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
        """
        model = create_default_2d_model

        # apply default solver settings and output settings
        model.project_parameters = create_default_solver_settings
        model.add_output_settings(**create_default_outputs.__dict__)

        # perform post setup
        model.post_setup()

        # set mesh size and generate mesh
        model.set_mesh_size(1)
        model.generate_mesh()

        kratos_io = KratosIO(ndim=model.ndim)

        kratos_io.write_input_files_for_kratos(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa"
        )

        # test mdpa
        with open('./test_mdpa_file.mdpa', 'r') as openfile:
            actual_text = openfile.readlines()
        with open('tests/test_data/expected_mdpa_file.mdpa', 'r') as openfile:
            expected_text = openfile.readlines()
        npt.assert_equal(actual=actual_text, desired=expected_text)

        # test json material
        actual_dict = json.load(open('./MaterialParameters.json', 'r'))
        expected_dict = json.load(open('tests/test_data/expected_MaterialParameters.json', 'r'))
        TestUtils.assert_dictionary_almost_equal(expected=expected_dict, actual=actual_dict)

        # test json project
        actual_dict = json.load(open('./ProjectParameters.json', 'r'))
        expected_dict = json.load(open('tests/test_data/expected_ProjectParameters.json', 'r'))
        TestUtils.assert_dictionary_almost_equal(expected=expected_dict, actual=actual_dict)

    def test_create_submodelpart_text(self, create_default_2d_model:Model):
        """
        Test the creation of the mdpa text of a model part

        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default model to use in testing
        """
        # load the default 2D model
        model = create_default_2d_model

        model.set_mesh_size(1)
        model.generate_mesh()

        body_model_part_to_write = model.body_model_parts[0]
        process_model_part_to_write = model.process_model_parts[0]

        # IO object
        kratos_io = KratosIO(ndim=model.ndim)

        # initialise ids for table, materials and processes
        kratos_io.initialise_model_ids(model)

        # generate text block body model part: soil1
        actual_text_body = kratos_io.write_submodelpart_body_model_part(
            body_model_part=body_model_part_to_write
        )
        # define expected block text
        expected_text_body = [
            "",
            "Begin SubModelPart soil1",
            "  Begin SubModelPartTables",
            "  End SubModelPartTables",
            "  Begin SubModelPartNodes",
            "  1",
            "  2",
            "  3",
            "  4",
            "  5",
            "  End SubModelPartNodes",
            "  Begin SubModelPartElements",
            "  5",
            "  6",
            "  7",
            "  8",
            "  End SubModelPartElements",
            "End SubModelPart",
            ""
        ]
        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_body, desired=expected_text_body)

        # generate text block body model part: soil1
        actual_text_load = kratos_io.write_submodelpart_process_model_part(
            process_model_part=process_model_part_to_write
        )
        # define expected block text
        expected_text_load = [
            "",
            "Begin SubModelPart load_top",
            "  Begin SubModelPartTables",
            "  1",
            "  End SubModelPartTables",
            "  Begin SubModelPartNodes",
            "  3",
            "  4",
            "  End SubModelPartNodes",
            "  Begin SubModelPartConditions",
            "  1",
            "  End SubModelPartConditions",
            "End SubModelPart",
            ""
        ]

        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_load, desired=expected_text_load)

    def test_write_mdpa_text(self, create_default_2d_model: Model):
        """
        Test the creation of the mdpa text of the whole model

        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default model to use in testing

        """
        # load the default 2D model
        model = create_default_2d_model

        model.project_parameters = TestUtils.create_default_solver_settings()

        model.post_setup()

        model.set_mesh_size(1)
        model.generate_mesh()

        # IO object
        kratos_io = KratosIO(ndim=model.ndim)

        # initialise ids for table, materials and processes
        kratos_io.initialise_model_ids(model)

        # generate text block body model part: soil1
        actual_text_body = kratos_io._KratosIO__write_mdpa_text(model=model)

        # define expected block text
        with open('tests/test_data/expected_mdpa_file.mdpa', 'r') as openfile:
            expected_text_body = openfile.readlines()

        expected_text_body = [line.rstrip() for line in expected_text_body]
        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_body, desired=expected_text_body)

    def test_write_mdpa_two_conditions_same_position(self):
        """
        Test the creation of the mdpa text of the whole model with two conditions at the same position. The two
        conditions should be written with unique condition element ids.

        """

        ndim = 2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        load_coordinates_top = [(1, 1, 0), (0, 1, 0)]  # top

        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        # define load properties
        line_load1 = LineLoad(active=[False, True, False], value=[0, 10, 0])
        line_load2 = LineLoad(active=[True, False, False], value=[10, 0, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        model.add_load_by_coordinates(load_coordinates_top, line_load1, "line_load_1")
        model.add_load_by_coordinates(load_coordinates_top, line_load2, "line_load_2")

        model.synchronise_geometry()

        # set mesh size and generate mesh
        model.set_mesh_size(1)
        model.generate_mesh()
        model.project_parameters = TestUtils.create_default_solver_settings()

        # write mdpa text
        kratos_io = KratosIO(ndim=model.ndim)
        actual_mdpa_text = kratos_io._KratosIO__write_mdpa_text(model=model)

        # get expected mdpa text
        with open('tests/test_data/expected_mdpa_file_two_conds_same_position.mdpa', 'r') as f:
            expected_mdpa_text = f.readlines()

        expected_mdpa_text = [line.rstrip() for line in expected_mdpa_text]

        # check if mdpa data is as expected
        npt.assert_equal(actual=actual_mdpa_text, desired=expected_mdpa_text)

    def test_write_mdpa_with_spring_damper_element(
            self, create_default_outputs: List[Output], create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the mdpa file for the default model with 4 spring dampers of which two are in series.

        Args:
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.

        """
        model = Model(ndim=2)
        kratos_io = KratosIO(ndim=model.ndim)
        model.project_parameters = create_default_solver_settings
        model.output_settings = create_default_outputs

        # add elastic spring damper element
        spring_damper = ElasticSpringDamper(
            NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1],
            NODAL_ROTATIONAL_STIFFNESS=[1, 1, 2],
            NODAL_DAMPING_COEFFICIENT=[1, 1, 3],
            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[1, 1, 4])

        # create model part
        # 3 lines, one broken with a mid-point, which should result in 4 springs
        # the lines are in different size so all the line are broken in smaller lines except the last.

        top_coordinates = [(0, 1, 0), (0, 2, 0), (1, 1, 0), (2, 0.3, 0)]
        bottom_coordinates = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 0, 0)]

        gmsh_input_top = {"top_coordinates": {"coordinates": top_coordinates, "ndim": 0}}
        gmsh_input_bottom = {"bottom_coordinates": {"coordinates": bottom_coordinates, "ndim": 0}}

        model.gmsh_io.generate_geometry(gmsh_input_top, "")
        model.gmsh_io.generate_geometry(gmsh_input_bottom, "")

        # create rail pad geometries
        top_point_ids = model.gmsh_io.make_points(top_coordinates)
        bot_point_ids = model.gmsh_io.make_points(bottom_coordinates)

        spring_line_ids = [model.gmsh_io.create_line([top_point_id, bot_point_id])
                           for top_point_id, bot_point_id in zip(top_point_ids, bot_point_ids)]

        model.gmsh_io.add_physical_group("spring_damper", 1, spring_line_ids)
        # assign spring damper to geometry
        spring_damper_model_part = BodyModelPart("spring_damper")
        spring_damper_model_part.material = StructuralMaterial("spring_damper", spring_damper)
        spring_damper_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "spring_damper")

        # add concentrated masses on same and different points
        # add nodal concentrated element
        nodal_concentrated = NodalConcentrated(
            NODAL_MASS=1,
            NODAL_DAMPING_COEFFICIENT=[1, 1, 1],
            NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1]
        )

        # create model part
        nodal_concentrated_model_part = BodyModelPart("nodal_concentrated")
        nodal_concentrated_model_part.material = StructuralMaterial("nodal_concentrated", nodal_concentrated)

        # assign nodal concentrated to geometry
        model.gmsh_io.add_physical_group("nodal_concentrated", 0, [3])
        nodal_concentrated_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "nodal_concentrated")

        # add model parts to model
        model.body_model_parts.append(spring_damper_model_part)
        model.body_model_parts.append(nodal_concentrated_model_part)

        model.synchronise_geometry()
        model.set_mesh_size(0.4)

        model.generate_mesh()

        # write mdpa text
        actual_mdpa_text = kratos_io._KratosIO__write_mdpa_text(model=model)

        # get expected mdpa text
        with open('tests/test_data/expected_mdpa_spring_dampers.mdpa', 'r') as f:
            expected_mdpa_text = f.readlines()

        expected_mdpa_text = [line.rstrip() for line in expected_mdpa_text]

        # check if mdpa data is as expected
        npt.assert_equal(actual=actual_mdpa_text, desired=expected_mdpa_text)

    def test_write_project_parameters_with_spring_damper_and_mass_element(self,
                                                                          create_default_2d_model: Model,
                                                                          create_default_outputs: List[Output],
                                                                          create_default_solver_settings: Problem):
        """
        Test correct writing of the project parameters for the default output, model and settings with a spring damper
        and a nodal concentrated element.

        Args:
            - create_default_2d_model (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer and a line load.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
        """
        model = create_default_2d_model
        kratos_io = KratosIO(ndim=model.ndim)
        model.project_parameters = create_default_solver_settings
        model.add_output_settings(**create_default_outputs.__dict__)
        kratos_io.project_folder = "dummy"

        # add elastic spring damper element
        spring_damper = ElasticSpringDamper(
            NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1],
            NODAL_ROTATIONAL_STIFFNESS=[1, 1, 2],
            NODAL_DAMPING_COEFFICIENT=[1, 1, 3],
            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[1, 1, 4])

        # create model part
        spring_damper_model_part = BodyModelPart("spring_damper")
        spring_damper_model_part.material = StructuralMaterial("spring_damper", spring_damper)

        # assign spring damper to geometry
        model.gmsh_io.add_physical_group("spring_damper", 1, [1])
        spring_damper_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "spring_damper")

        # add nodal concentrated element
        nodal_concentrated = NodalConcentrated(
            NODAL_MASS=1,
            NODAL_DAMPING_COEFFICIENT=[1, 1, 1],
            NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1]
        )

        # create model part
        nodal_concentrated_model_part = BodyModelPart("nodal_concentrated")
        nodal_concentrated_model_part.material = StructuralMaterial("nodal_concentrated", nodal_concentrated)

        # assign nodal concentrated to geometry
        model.gmsh_io.add_physical_group("nodal_concentrated", 0, [2])
        nodal_concentrated_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "nodal_concentrated")

        # add model parts to model
        model.body_model_parts.append(spring_damper_model_part)
        model.body_model_parts.append(nodal_concentrated_model_part)

        # write project parameters
        actual_dict = kratos_io._KratosIO__write_project_parameters_json(model=model, mesh_file_name="dummy.mdpa",
                                                                         materials_file_name="dummy.json")

        # load expected project parameters
        expected_dict = json.load(open("tests/test_data/expected_ProjectParameters_with_nodal_parameters.json", 'r'))

        # assert the dictionaries to be equal
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_create_auxiliary_process_list_dictionary_expected_raises(self):
        """
        Test the creation of the auxiliary process list dictionary with expected errors.

        """
        # create model
        model = Model(ndim=2)

        # create IO object
        kratos_io = KratosIO(ndim=model.ndim)

        empty_body_model_part = BodyModelPart("empty_body_model_part")
        model.body_model_parts = [empty_body_model_part]

        # create auxiliary process list dictionary
        with pytest.raises(ValueError, match=f"Body model part empty_body_model_part has no material assigned."):
            kratos_io._KratosIO__create_auxiliary_process_list_dictionary(model=model)

        nodal_concentrated = NodalConcentrated(NODAL_MASS=1, NODAL_DAMPING_COEFFICIENT=[1, 1, 1],
                                               NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1])
        empty_body_model_part.material = StructuralMaterial("empty_body_model_part", nodal_concentrated)

        # create auxiliary process list dictionary
        with pytest.raises(ValueError, match=f"Body model part empty_body_model_part has no id initialised."):
            kratos_io._KratosIO__create_auxiliary_process_list_dictionary(model=model)

    def test_create_folder_for_json_output(self):
        """
        Test the creation of the folder for the json output. Since the folder is not created by kratos.

        """

        # check relative directory creation
        kratos_io = KratosIO(ndim=2)
        kratos_io.project_folder = "json_test_project_folder"
        output_settings = [Output(output_parameters=JsonOutputParameters(
            output_interval=1),output_dir="json_test_output"
        )]

        kratos_io._KratosIO__create_folder_for_json_output(output_settings)

        expected_folder = os.path.join("json_test_project_folder", "json_test_output")

        assert os.path.exists(expected_folder)
        rmtree(expected_folder)

        # check absolute path directory creation
        absolute_path_json_output = os.path.join(os.getcwd(), "json_test_output")
        output_settings[0].output_dir = absolute_path_json_output

        kratos_io._KratosIO__create_folder_for_json_output(output_settings)

        assert os.path.exists(absolute_path_json_output)
        rmtree(absolute_path_json_output)
