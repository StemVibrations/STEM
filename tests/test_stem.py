import os
from pathlib import Path
from unittest.mock import MagicMock
from copy import deepcopy

import KratosMultiphysics

import pytest

from stem.stem import Stem
from stem.model import Model
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, VtkOutputParameters
from stem.IO.kratos_io import KratosIO

class TestStem:


    @pytest.fixture()
    def create_default_model(self):
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
        problem = Problem(problem_name="test_1d_wave_prop_drained_soil", number_of_threads=2, settings=solver_settings)
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

        return model

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
        with pytest.raises(Exception, match="Meshes between stages in body model part: "
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
        assert stem._Stem__stage_settings_file_names == {1: "ProjectParameters_stage_1.json",
                                                         2: "ProjectParameters_stage_2.json"}

        # Cleanup
        for file in Path(input_folder).iterdir():
            file.unlink()
        Path(input_folder).rmdir()

    # def test_run_stage_in_order(self, create_default_model):
    #
    #     stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
    #     stage2 = deepcopy(create_default_model)
    #     stem.add_calculation_stage(stage2)
    #
    #     stem.run_stage(1)
    #     stem.run_stage(2)
    #
    # def test_run_stage_out_of_order(self,create_default_model):
    #     stem = Stem(initial_stage=create_default_model, input_files_dir="input_files")
    #     stage2 = deepcopy(create_default_model)
    #     stem.add_calculation_stage(stage2)
    #     with pytest.raises(Exception, match="Stages should be run in order"):
    #         stem.run_stage(2)





