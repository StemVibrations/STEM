import os
from pathlib import Path
from unittest.mock import MagicMock
from copy import deepcopy
import threading

import KratosMultiphysics
from gmsh_utils import gmsh_IO

import pytest

from stem.stem import Stem
from stem.model import Model
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, VtkOutputParameters, GiDOutputParameters, JsonOutputParameters
from stem.IO.kratos_io import KratosIO

from tests.utils import TestUtils


@pytest.mark.serial
class TestStem:

    # Lock to prevent multiple threads from running the same fixture
    lock = threading.Lock()

    @pytest.fixture(scope="function")
    def create_default_model(self):

        # Create a model with a soil column and a line load
        with self.lock:
            ndim = 2
            model = Model(ndim)

            DENSITY_SOLID = 2700
            POROSITY = 0.3
            YOUNG_MODULUS = 50e6
            POISSON_RATIO = 0.3
            soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
            constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
            retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
            material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

            # Specify the coordinates for the column: x:1m x y:10m
            layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 10, 0), (0, 10, 0)]

            # Create the soil layer
            model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_column")

            # Boundary conditions and Loads
            load_coordinates = [(0.0, 10.0, 0), (1.0, 10.0, 0)]

            # Add line load
            line_load = LineLoad(active=[False, True, False], value=[0, -1000, 0])
            model.add_load_by_coordinates(load_coordinates, line_load, "load")

            # Define boundary conditions
            no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                                is_fixed=[True, True, True],
                                                                value=[0, 0, 0])

            # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
            model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")

            # Set mesh size
            # --------------------------------
            model.set_mesh_size(element_size=1)

            # Define project parameters
            # --------------------------------

            # Set up solver settings
            analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
            solution_type = SolutionType.DYNAMIC
            # Set up start and end time of calculation, time step and etc
            time_integration = TimeIntegration(start_time=0.0,
                                               end_time=0.15,
                                               delta_time=0.0025,
                                               reduction_factor=1.0,
                                               increase_factor=1.0,
                                               max_delta_time_factor=1000)
            convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-12,
                                                                    displacement_absolute_tolerance=1.0E-6)
            stress_initialisation_type = StressInitialisationType.NONE
            solver_settings = SolverSettings(analysis_type=analysis_type,
                                             solution_type=solution_type,
                                             stress_initialisation_type=stress_initialisation_type,
                                             time_integration=time_integration,
                                             is_stiffness_matrix_constant=True,
                                             are_mass_and_damping_constant=True,
                                             convergence_criteria=convergence_criterion,
                                             rayleigh_k=6e-6,
                                             rayleigh_m=0.02)

            # Set up problem data
            problem = Problem(problem_name="test_1d_wave_prop_drained_soil",
                              number_of_threads=2,
                              settings=solver_settings)
            model.project_parameters = problem

            # Define the results to be written to the output file
            # Nodal results
            nodal_results = [NodalOutput.DISPLACEMENT]

            # Define the output process
            model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                            output_interval=10,
                                                                            nodal_results=nodal_results,
                                                                            gauss_point_results=[],
                                                                            output_control_type="step"),
                                      output_dir="output",
                                      output_name="vtk_output")

            # return the model
            yield model

            # make sure gmsh is finalized after each test
            gmsh_IO.GmshIO().finalize_gmsh()

    def test_create_new_stage_with_valid_parameters(self, create_default_model):

        input_folder = "tests/test_data/generated_input/inputs_kratos"
        stem = Stem(initial_stage=create_default_model, input_files_dir=input_folder)
        new_stage = stem.create_new_stage(delta_time=0.1, stage_duration=1.0)
        assert isinstance(new_stage, Model)
        assert new_stage.project_parameters.settings.time_integration.delta_time == 0.1
        assert new_stage.project_parameters.settings.time_integration.end_time == 1.15

        assert new_stage.output_settings[0].output_dir == Path("output/output_vtk_full_model_stage_2")

    def test_create_new_stage_with_no_project_parameters(self, create_default_model):

        # create the stem object
        input_folder = "tests/test_data/generated_input/inputs_kratos"
        stem = Stem(initial_stage=create_default_model, input_files_dir=input_folder)

        # Set the project parameters of the last stage to None
        stem._Stem__stages[0].project_parameters = None

        # Check if ValueError is raised
        with pytest.raises(Exception, match="Project parameters of the last stage are not set"):
            stem.create_new_stage(delta_time=0.1, stage_duration=1.0)

    def test_stem_init(self):
        """
        Test the initialization of the Stem class with valid parameters it checks if the post_setup and generate_mesh
        methods of the initial stage are called

        """

        # initialize the initial stage with mock objects
        initial_stage = Model(2)
        initial_stage.post_setup = MagicMock()
        initial_stage.generate_mesh = MagicMock()
        stem = Stem(initial_stage, "input_files")

        assert isinstance(stem.kratos_io, KratosIO)
        assert isinstance(stem.kratos_model, KratosMultiphysics.Model)
        assert stem._Stem__stages == [initial_stage]

        # Check if the post_setup and generate_mesh methods of the initial stage are called
        initial_stage.post_setup.assert_called_once()
        initial_stage.generate_mesh.assert_called_once()

    def test_add_calculation_stage_with_valid_model(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")

        stage2 = deepcopy(create_default_model)

        # Mock the methods of the stage object
        stage2.gmsh_io.generate_geo_from_geo_data = MagicMock()
        stage2.post_setup = MagicMock()
        stage2.generate_mesh = MagicMock()

        # Add the stage to the stem object
        stem.add_calculation_stage(stage2)

        # check if the stage is added to the stem object
        assert len(stem.stages) == 2

        # Check if the methods of the stage object are called
        stage2.gmsh_io.generate_geo_from_geo_data.assert_called_once()
        stage2.post_setup.assert_called_once()
        stage2.generate_mesh.assert_called_once()

        # create an invalid stage with extra body model parts
        stage3 = deepcopy(create_default_model)

        # change the coordinates of the body model part such that a different mesh is generated
        stage3.gmsh_io.geo_data["points"][2] = [10, 10, 0]
        # Check if ValueError is raised
        with pytest.raises(Exception,
                           match="Meshes between stages in body model part: "
                           "soil_column are not the same between stages"):
            stem.add_calculation_stage(stage3)

    def test_validate_latest_stage(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        stage2 = deepcopy(create_default_model)

        # Mock the method of the stem object
        stem._Stem__check_if_mesh_between_stages_is_the_same = MagicMock()

        # add new stage to the stem object
        stem._Stem__stages.append(stage2)
        assert len(stem.stages) == 2

        # check if no exception is raised
        stem.validate_latest_stage()
        stem._Stem__check_if_mesh_between_stages_is_the_same.assert_called_once()

        stage3 = deepcopy(create_default_model)
        # add new stage to the stem object
        stem._Stem__stages.append(stage3)
        assert len(stem.stages) == 3

        # add a new body model part to the new stage
        stage3.body_model_parts.append("new_part")

        # check if ValueError is raised
        with pytest.raises(Exception, match="Number of body model parts are not the same between stages"):
            stem.validate_latest_stage()

        stage3 = deepcopy(create_default_model)
        # add new stage to the stem object
        stem._Stem__stages[2] = stage3
        assert len(stem.stages) == 3

        # add a new process model part to the new stage
        stage3.process_model_parts.append("new_part")

        # check if ValueError is raised
        with pytest.raises(Exception, match="Number of process model parts are not the same between stages"):
            stem.validate_latest_stage()

    def test_write_all_input_files(self, create_default_model):
        input_folder = "tests/test_data/generated_input/test_generate_write_all_input_files"
        stem = Stem(initial_stage=create_default_model, input_files_dir=input_folder)
        stage2 = deepcopy(create_default_model)
        stem.add_calculation_stage(stage2)

        # mock validate settings
        create_default_model.project_parameters.settings.validate_settings = MagicMock()
        stage2.project_parameters.settings.validate_settings = MagicMock()

        stem.write_all_input_files()

        # check if settings are validated
        create_default_model.project_parameters.settings.validate_settings.assert_called_once()
        stage2.project_parameters.settings.validate_settings.assert_called_once()

        # check if all input files are written
        assert Path(input_folder).is_dir()
        assert Path(input_folder).joinpath("ProjectParameters_stage_1.json").is_file()
        assert Path(input_folder).joinpath("ProjectParameters_stage_2.json").is_file()
        assert Path(input_folder).joinpath("MaterialParameters_stage_1.json").is_file()
        assert Path(input_folder).joinpath("MaterialParameters_stage_2.json").is_file()
        assert Path(input_folder).joinpath("test_1d_wave_prop_drained_soil_stage_1.mdpa").is_file()
        assert Path(input_folder).joinpath("test_1d_wave_prop_drained_soil_stage_2.mdpa").is_file()

        # check if filenames are correctly stored
        assert stem._Stem__stage_settings_file_names == {
            1: "ProjectParameters_stage_1.json",
            2: "ProjectParameters_stage_2.json"
        }

        # Cleanup
        TestUtils.clean_test_directory(Path(input_folder))

    def test_run_stage(self, create_default_model):
        """
        Test the run_stage method of the Stem class

        Args:
            create_default_model:

        Returns:

        """

        input_folder = "tests/test_data/generated_input/test_run_stage"
        stem = Stem(initial_stage=create_default_model, input_files_dir=input_folder)
        stem.write_all_input_files()

        stem.run_stage(1)

        # check if output is generated
        assert Path(input_folder).joinpath("output/output_vtk_full_model").is_dir()

        # Cleanup
        TestUtils.clean_test_directory(Path(input_folder))

    def test_run_stage_out_of_order(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        stage2 = deepcopy(create_default_model)
        stem.add_calculation_stage(stage2)
        with pytest.raises(Exception, match="Stages should be run in order"):
            stem.run_stage(2)

    def test_finalise_one_stage(self, create_default_model):

        # Test if the transfer vtk files method is not called when there is only one stage
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")

        # mock the method
        stem._Stem__transfer_vtk_files_to_main_output_directories = MagicMock()

        # run the finalise method
        stem.finalise()

        # check if the method is not called
        stem._Stem__transfer_vtk_files_to_main_output_directories.assert_not_called()

    def test_finalise_multiple_stages(self, create_default_model):

        # Test if the transfer vtk files method is called when there are multiple stages
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")

        # create and add second stage
        stage2 = deepcopy(create_default_model)
        stem.add_calculation_stage(stage2)

        # mock the method
        stem._Stem__transfer_vtk_files_to_main_output_directories = MagicMock()

        # run the finalise method
        stem.finalise()

        # check if the method is called
        stem._Stem__transfer_vtk_files_to_main_output_directories.assert_called_once()

    def test_run_calculation_with_valid_stages(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        stage2 = deepcopy(create_default_model)
        stem.add_calculation_stage(stage2)

        # mock the methods
        stem.run_stage = MagicMock()
        stem.finalise = MagicMock()

        # run the calculation
        stem.run_calculation()

        # check if the run_stage method is called twice
        assert stem.run_stage.call_count == 2

        # check if correct arguments are passed to the run_stage method
        assert stem.run_stage.call_args_list[0][0][0] == 1
        assert stem.run_stage.call_args_list[1][0][0] == 2

        # check if finalise is called
        stem.finalise.assert_called_once()

    def test_check_mesh_between_stages_same(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        stage2 = deepcopy(create_default_model)
        stem.add_calculation_stage(stage2)

        # No exception should be raised as the body model parts are the same
        stem._Stem__check_if_mesh_between_stages_is_the_same(stem.stages[0], stem.stages[1])

        stage2 = deepcopy(create_default_model)
        stage2.body_model_parts.append("new_part")
        stem.stages[1] = stage2

        with pytest.raises(Exception, match="Number of body model parts are not the same between stages"):
            stem._Stem__check_if_mesh_between_stages_is_the_same(stem.stages[0], stem.stages[1])

        stage2 = deepcopy(create_default_model)
        stage2.body_model_parts[0]._ModelPart__name = "new_name"
        stem.stages[1] = stage2

        with pytest.raises(Exception, match="Body model part names are not the same between stages"):
            stem._Stem__check_if_mesh_between_stages_is_the_same(stem.stages[0], stem.stages[1])

        # add a new stage with a different mesh
        stage2 = deepcopy(create_default_model)
        stage2.body_model_parts[0].mesh = "new_mesh"
        stem.stages[1] = stage2

        # check if exception is raised correctly
        with pytest.raises(Exception,
                           match="Meshes between stages in body model part: "
                           "soil_column are not the same between stages"):
            stem._Stem__check_if_mesh_between_stages_is_the_same(stem.stages[0], stem.stages[1])

    def test_transfer_vtk_files_to_main_output_directories_single_stage(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        stem._Stem__transfer_vtk_files_to_main_output_directories()
        # No exception should be raised as there is only one stage

    def test_transfer_vtk_files_to_main_output_directories_multiple_stages(self, create_default_model):

        input_dir = "tests/test_data/generated_input/test_transfer_vtk_files"
        stem = Stem(initial_stage=create_default_model, input_files_dir=input_dir)
        stage2 = deepcopy(create_default_model)

        # create stage_2 output directory
        Path(input_dir).joinpath("output/output_vtk_full_model_stage_2").mkdir(parents=True, exist_ok=True)

        stage2.output_settings[0].output_dir = "output/output_vtk_full_model_stage_2"
        stage2.output_settings[0].part_name = None

        # add the stage to the stem object
        stem.add_calculation_stage(stage2)

        # Create a dummy vtk file in the second stage output directory
        dummy_vtk_file = input_dir / Path("output/output_vtk_full_model_stage_2") / "dummy.vtk"
        dummy_vtk_file.touch()

        # generate main output directory
        main_output_dir = Path(input_dir).joinpath("output/output_vtk_full_model")
        main_output_dir.mkdir(parents=True, exist_ok=True)

        # Run the method
        stem._Stem__transfer_vtk_files_to_main_output_directories()

        # Check if the vtk file of stage 2 is moved to the main output directory
        assert (main_output_dir / "dummy.vtk").is_file()

        # Check if the second stage output directory is removed
        assert not (input_dir / Path("output/output_vtk_full_model_stage_2")).is_dir()

        # Cleanup
        TestUtils.clean_test_directory(Path(input_dir))

    def test_set_output_name_new_stage_with_vtk_output(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        new_stage = deepcopy(create_default_model)

        stem.stages.append(new_stage)
        stem._Stem__set_output_name_new_stage(new_stage, 5)
        assert new_stage.output_settings[0].output_dir == Path("output/output_vtk_full_model_stage_5")

    def test_set_output_name_new_stage_with_gid_output(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        new_stage = stem.create_new_stage(delta_time=0.1, stage_duration=1.0)
        new_stage.output_settings[0].output_parameters = MagicMock(spec=GiDOutputParameters)
        new_stage.output_settings[0].output_name = "gid_output"
        stem._Stem__set_output_name_new_stage(new_stage, 4)
        assert new_stage.output_settings[0].output_name == "gid_output_stage_4"

    def set_output_name_new_stage_with_json_output(self, create_default_model):
        stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
        new_stage = stem.create_new_stage(delta_time=0.1, stage_duration=1.0)
        new_stage.output_settings[0].output_parameters = MagicMock(spec=JsonOutputParameters)
        new_stage.output_settings[0].output_name = "json_output"
        stem._Stem__set_output_name_new_stage(new_stage, 3)
        assert new_stage.output_settings[0].output_name == "json_output_stage_3"
