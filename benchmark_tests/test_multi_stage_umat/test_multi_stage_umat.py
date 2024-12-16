import json
import os
import sys
from shutil import rmtree, copyfile

import numpy as np

from stem.boundary import DisplacementConstraint
from stem.load import LineLoad
from stem.model import Model
from stem.output import JsonOutputParameters, NodalOutput, VtkOutputParameters
from stem.soil_material import LinearElasticSoil, OnePhaseSoil, SaturatedBelowPhreaticLevelLaw, SmallStrainUmatLaw, SoilMaterial
from stem.solver import AnalysisType, DisplacementConvergenceCriteria, Problem, SolutionType, SolverSettings, StressInitialisationType, TimeIntegration, NewtonRaphsonStrategy, LinearNewtonRaphsonStrategy
from stem.stem import Stem
from stem.table import Table
from stem.utils import Utils

from tests.utils import TestUtils

SHOW_RESULTS = False


def test_stem():
    """
    Test STEM: 2D block with distributed cyclic loading with multistage for the umat using umat and changing the
    stiffness of the material in the second stage (halved).

    Note that currently, 2D linear elastic plane strain law  is written with the incremental formulation, and the
    linear elastic 3D with the full formulation. Different results in multistage analyses are expected
    """

    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model_stage_1 = Model(ndim)

    # Specify material model
    # Linear elastic drained soil with a Density of 2700, a Young's modulus of 50e6,
    # a Poisson ratio of 0.3 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2700
    POROSITY = 0.3
    YOUNG_MODULUS = 50e6
    POISSON_RATIO = 0.3
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:1m x y:0.5m
    layer1_coordinates = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.0, 0.5, 0.0)]
    output_coordinates = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.0)]
    # Create the soil layer
    model_stage_1.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_block")

    # Boundary conditions and Loads
    # a sinus load at 20Hz (load period T=0.05s)
    load_frequency = 20  # Hz
    delta_time = 0.005  # s = 10 points per cycle
    total_simulation_time = 0.5  # s =  10 cycles
    load_pulse = load_frequency * (2 * np.pi)  # rad/s

    t = np.arange(0, total_simulation_time + delta_time, delta_time)  # s
    values = -1000 * np.sin(load_pulse * t)  # N

    LOAD_Y = Table(times=t, values=values)
    # Add line load
    load_coordinates = [(0.0, 0.5, 0.0), (1.0, 0.5, 0.0)]
    load = LineLoad(value=[0, LOAD_Y, 0], active=[False, True, False])
    model_stage_1.add_load_by_coordinates(load_coordinates, load, "point_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model_stage_1.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model_stage_1.add_boundary_condition_by_geometry_ids(1, [2, 4], sym_parameters, "side_rollers")

    # Synchronize geometry
    model_stage_1.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model_stage_1.set_mesh_size(element_size=0.5)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC

    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=total_simulation_time / 2,
                                       delta_time=delta_time,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-6,
                                                            displacement_absolute_tolerance=1.0E-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     rayleigh_k=1e-03,
                                     rayleigh_m=0.02)

    # Set up problem data
    problem = Problem(problem_name="test_multi_stage_umat", number_of_threads=2, settings=solver_settings)
    model_stage_1.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

    # Uncomment this bock if you need to check the results in PARAVIEW

    # model_stage_1.add_output_settings(
    #     output_parameters=VtkOutputParameters(
    #         file_format="ascii",
    #         output_interval=1,
    #         nodal_results=nodal_results,
    #         gauss_point_results=[],
    #     ),
    #     output_dir="output",
    #     output_name="vtk_output",
    # )

    model_stage_1.add_output_settings_by_coordinates(coordinates=output_coordinates,
                                                     part_name="midline_output",
                                                     output_parameters=JsonOutputParameters(output_interval=delta_time,
                                                                                            nodal_results=nodal_results,
                                                                                            gauss_point_results=[]),
                                                     output_dir="output",
                                                     output_name="json_output")

    # define the STEM instance
    input_folder = "benchmark_tests/test_multi_stage_umat/inputs_kratos"
    stem = Stem(model_stage_1, input_folder)

    # create new stage:
    # the new material parameters have a Young's modulus half of the stage 1 material
    model_stage_2 = stem.create_new_stage(delta_time, total_simulation_time / 2)

    YOUNG_MODULUS_2 = YOUNG_MODULUS / 2
    SHEAR_MODULUS = YOUNG_MODULUS_2 / (2 * (1 + POISSON_RATIO))

    # copy the linear elastic umat to the input folder
    if sys.platform == "linux":
        extension = "so"
    elif sys.platform == "win32":
        extension = "dll"
    else:
        raise Exception("Unknown platform")

    soil_formulation_stage_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law_stage_2 = SmallStrainUmatLaw(UMAT_NAME=f"../linear_elastic.{extension}",
                                                  IS_FORTRAN_UMAT=True,
                                                  UMAT_PARAMETERS=[SHEAR_MODULUS, POISSON_RATIO],
                                                  STATE_VARIABLES=[0.0])

    retention_parameters_stage_2 = SaturatedBelowPhreaticLevelLaw()
    material_stage_2 = SoilMaterial("soil2", soil_formulation_stage_2, constitutive_law_stage_2,
                                    retention_parameters_stage_2)

    model_stage_2.body_model_parts[0].material = material_stage_2

    # add the new stage to the calculation
    stem.add_calculation_stage(model_stage_2)

    # write the kratos input files
    stem.write_all_input_files()

    # copy the linear elastic dll to the input folder
    copyfile(src=rf"benchmark_tests/user_defined_models/linear_elastic.{extension}",
             dst=rf"benchmark_tests/test_multi_stage_umat/linear_elastic.{extension}")

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    # Assert results stage 1
    with open(r"benchmark_tests/test_multi_stage_umat/output_/json_output.json") as f:
        expected_data_stage1 = json.load(f)

    with open(os.path.join(input_folder, "output/json_output.json")) as f:
        calculated_data_stage1 = json.load(f)

    # Check if the expected displacements and velocities in stage 1 are equal to the calculated ones
    TestUtils.assert_dictionary_almost_equal(expected=expected_data_stage1, actual=calculated_data_stage1)

    # Assert results stage 2
    with open(r"benchmark_tests/test_multi_stage_umat/output_/json_output_stage_2.json") as f:
        expected_data_stage2 = json.load(f)

    with open(os.path.join(input_folder, "output/json_output_stage_2.json")) as f:
        calculated_data_stage2 = json.load(f)

    # Check if the expected displacements and velocities in stage 2 are equal to the calculated ones
    TestUtils.assert_dictionary_almost_equal(expected=expected_data_stage2, actual=calculated_data_stage2)

    merged_expected_data = Utils.merge(expected_data_stage1, expected_data_stage2)
    merged_calculated_data = Utils.merge(calculated_data_stage1, calculated_data_stage2)

    # Only calculate analytical solution and show results if SHOW_RESULTS is True
    if SHOW_RESULTS:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex="all")
        ax[0].set_title("Displacements X")
        ax[0].set_ylabel("displacement_x [m]")
        ax[0].plot(merged_expected_data["TIME"], merged_expected_data["NODE_5"]["DISPLACEMENT_X"], label="Expected")
        ax[0].plot(merged_calculated_data["TIME"],
                   merged_calculated_data["NODE_5"]["DISPLACEMENT_X"],
                   label="Calculated")

        ax[1].set_title("Displacements Y")
        ax[0].set_ylabel("displacement_y [m]")
        ax[1].set_xlabel("time [s]")
        ax[1].plot(merged_expected_data["TIME"], merged_expected_data["NODE_5"]["DISPLACEMENT_Y"], label="Expected")
        ax[1].plot(merged_calculated_data["TIME"],
                   merged_calculated_data["NODE_5"]["DISPLACEMENT_Y"],
                   label="Calculated")
        ax[0].legend()
        ax[1].legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    rmtree(input_folder)
