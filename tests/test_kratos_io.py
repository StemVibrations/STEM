import json
import sys
from typing import List

import numpy as np
import numpy.testing as npt
import pytest
from gmsh_utils import gmsh_IO

from stem.IO.kratos_io import KratosIO
from stem.boundary import DisplacementConstraint
from stem.load import LineLoad, SurfaceLoad
from stem.model import Model
from stem.model_part import *
from stem.output import NodalOutput, GaussPointOutput, GiDOutputParameters, Output
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
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
    def create_default_2d_model_and_mesh(self) -> Model:
        """
        Sets expected geometry data for a 2D geometry group. And it sets a time dependent line load at the top and
        bottom and another line load at the sides. The group is a geometry of a square.

        Returns:
            - :class:`stem.model.Model`: the default 2D model of a square soil layer and line loads.
        """
        ndim = 2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        output_coordinates = [(0.5, 0, 0), (0.5, 0.5, 0), (0.5, 1, 0)]

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
        _time = np.arange(6)*0.5
        _value1 = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(times=_time, values=_value1)
        table2 = Table(times=_time, values=-_value1)

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

        # add output
        # - Nodal results
        nodal_results = [NodalOutput.ACCELERATION, NodalOutput.VELOCITY, NodalOutput.DISPLACEMENT]
        # - define output process
        model.add_output_part_by_coordinates(
            coordinates=output_coordinates,
            part_name="test_output_part",
            output_name="gid_output_soil1",
            output_dir="dir_test",
            output_parameters=GiDOutputParameters(
                file_format="binary",
                output_interval=100,
                nodal_results=nodal_results
            )
        )
        model.synchronise_geometry()

        model.set_mesh_size(0.2)
        model.generate_mesh()

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
        model.extrusion_length = [0, 0, 1]

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

        return gid_output_process

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
        create_default_2d_model_and_mesh: Model,
        create_default_outputs: List[Output],
        create_default_solver_settings: Problem
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
        model.add_model_part_output(**create_default_outputs.__dict__)

        actual_dict = kratos_io.write_project_parameters_json(
            model=model,
            mesh_file_name="test_mdpa_file.mdpa",
            materials_file_name="MaterialParameters.json",
            output_folder="dir_test"
        )
        expected_dict = json.load(open("tests/test_data/expected_ProjectParameters.json", 'r'))
        TestUtils.assert_dictionary_almost_equal(expected_dict, actual_dict)

    def test_write_material_parameters_json(
        self,
        create_default_2d_model_and_mesh: Model
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

    def test_write_mdpa_file_2d(
        self,
        create_default_2d_model_and_mesh: Model,
        create_default_solver_settings: Problem
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings in 2D.

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
        model.project_parameters = create_default_solver_settings

        actual_text = kratos_io.write_mesh_to_mdpa(
            model=model,
            mesh_file_name="test_mdpa_file_3d.mdpa",
            output_folder="dir_test"
        )

        if IS_LINUX:
            with open('tests/test_data/expected_mdpa_file_3d_linux.mdpa', 'r') as openfile:
                expected_text = openfile.readlines()
        else:
            with open('tests/test_data/expected_mdpa_file_3d.mdpa', 'r') as openfile:
                expected_text = openfile.readlines()

        npt.assert_equal(actual=actual_text, desired=expected_text)

    def test_write_input_files_for_kratos(
        self,
        create_default_2d_model_and_mesh: Model,
        create_default_solver_settings: Problem,
        create_default_outputs: List[Output]
    ):
        """
        Test correct writing of the mdpa file (mesh) for the default model and solver settings.

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default 2D model of a square \
                soil layer line loads.
            - create_default_solver_settings (:class:`stem.solver.Problem`): the Problem object containing the \
                solver settings.
            - create_default_outputs (List[:class:`stem.output.Output`]): list of default output processes.
        """
        model = create_default_2d_model_and_mesh
        kratos_io = KratosIO(ndim=model.ndim)
        model.project_parameters = create_default_solver_settings
        model.add_model_part_output(**create_default_outputs.__dict__)

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

    def test_create_submodelpart_text(self, create_default_2d_model_and_mesh:Model):
        """
        Test the creation of the mdpa text of a model part

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default model to use in testing
        """
        # load the default 2D model
        model = create_default_2d_model_and_mesh
        # reduce the size of the model
        model.set_mesh_size(0.5)
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
            "  6",
            "  7",
            "  8",
            "  9",
            "  10",
            "  11",
            "  12",
            "  13",
            "  End SubModelPartNodes",
            "  Begin SubModelPartElements",
            "  11",
            "  12",
            "  13",
            "  14",
            "  15",
            "  16",
            "  17",
            "  18",
            "  19",
            "  20",
            "  21",
            "  22",
            "  23",
            "  24",
            "  25",
            "  26",
            "  End SubModelPartElements",
            "End SubModelPart",
            "",
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
            "  7",
            "  End SubModelPartNodes",
            "  Begin SubModelPartConditions",
            "  7",
            "  9",
            "  End SubModelPartConditions",
            "End SubModelPart",
            "",
        ]

        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_load, desired=expected_text_load)

    def test_write_mdpa_text(self, create_default_2d_model_and_mesh: Model):
        """
        Test the creation of the mdpa text of the whole model

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default model to use in testing

        """
        # load the default 2D model
        model = create_default_2d_model_and_mesh
        model.project_parameters = TestUtils.create_default_solver_settings()

        # IO object
        kratos_io = KratosIO(ndim=model.ndim)

        # initialise ids for table, materials and processes
        kratos_io.initialise_model_ids(model)

        # generate text block body model part: soil1
        actual_text_body = kratos_io.write_mdpa_text(model=model)

        # define expected block text
        with open('tests/test_data/expected_mdpa_file.mdpa', 'r') as openfile:
            expected_text_body = openfile.readlines()

        expected_text_body = [line.rstrip() for line in expected_text_body]
        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_body, desired=expected_text_body)