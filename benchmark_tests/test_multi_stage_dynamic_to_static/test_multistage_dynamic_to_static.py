import json
import os
from shutil import rmtree

from stem.boundary import DisplacementConstraint
from stem.load import LineLoad
from stem.model import Model
from stem.output import JsonOutputParameters, NodalOutput
from stem.soil_material import LinearElasticSoil, OnePhaseSoil, SaturatedBelowPhreaticLevelLaw, SoilMaterial
from stem.solver import AnalysisType, DisplacementConvergenceCriteria, Problem, SolutionType, SolverSettings, \
    StressInitialisationType, TimeIntegration, NewtonRaphsonStrategy
from stem.stem import Stem
from stem.table import Table
from stem.utils import Utils
from tests.utils import TestUtils

SHOW_RESULTS = False


def test_stem():
    """
    Test STEM: 2D block with distributed loading with multistage and change from dynamic -> quasi-static analysis type.
    A heaviside load is applied to the soil block. After one oscillation, the solution type is switched to QUASI_STATIC.
    The quasi-static displacement should match with the dynamic one if the oscillations are damped out.
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

    # Specify the coordinates for the block: x:1m x y:1m
    layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    output_coordinates = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.0)]
    # Create the soil layer
    model_stage_1.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_block")

    # Boundary conditions and Loads
    # A heaviside load is applied to the soil block
    load_coordinates = [(0.0, 1.0, 0), (1.0, 1.0, 0)]
    t = (0.0, 0.0075, 1)
    values = (0.0, -1000.0, -1000.0)
    LINE_LOAD_Y = Table(times=t, values=values)
    # Add line load
    line_load = LineLoad(active=[False, True, False], value=[0, LINE_LOAD_Y, 0])
    model_stage_1.add_load_by_coordinates(load_coordinates, line_load, "load")

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
    model_stage_1.set_mesh_size(element_size=0.3)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC

    delta_time = 0.0005
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(
        start_time=0.0,
        end_time=0.03,
        delta_time=delta_time,
        reduction_factor=1.0,
        increase_factor=1.0,
        max_delta_time_factor=1000,
    )
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-6,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    strategy = NewtonRaphsonStrategy()
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy,
                                     rayleigh_k=1.5e-3,
                                     rayleigh_m=0.02)

    # Set up problem data
    problem = Problem(
        problem_name="test_multi_stage_dynamic_to_static",
        number_of_threads=2,
        settings=solver_settings,
    )
    model_stage_1.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

    # Uncomment this block if you want to see the outputs in PARAVIEW
    #
    # model_stage_1.add_output_settings(
    #     output_parameters=VtkOutputParameters(
    #         file_format="ascii",
    #         output_interval=1,
    #         nodal_results=nodal_results,
    #         gauss_point_results=[],
    #         output_control_type="step",
    #     ),
    #     output_dir="output",
    #     output_name="vtk_output",
    # )

    model_stage_1.add_output_settings_by_coordinates(
        coordinates=output_coordinates,
        part_name="midline_output",
        output_parameters=JsonOutputParameters(
            output_interval=delta_time,
            nodal_results=nodal_results,
            gauss_point_results=[],
        ),
        output_dir="output",
        output_name="json_output",
    )

    # define the STEM instance
    input_folder = "benchmark_tests/test_multi_stage_dynamic_to_static/inputs_kratos"
    stem = Stem(model_stage_1, input_folder)

    # create new stage
    delta_time_stage_2 = 0.01
    model_stage_2 = stem.create_new_stage(delta_time_stage_2, 0.03)

    # Set up solver settings for the new stage
    model_stage_2.project_parameters.settings.solution_type = SolutionType.QUASI_STATIC
    model_stage_2.project_parameters.settings.rayleigh_k = 0.0
    model_stage_2.project_parameters.settings.rayleigh_m = 0.0

    model_stage_2.output_settings[-1].output_parameters.output_interval = (delta_time_stage_2 - 1e-08)

    # add the new stage to the calculation
    stem.add_calculation_stage(model_stage_2)

    # write the kratos input files
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    # Assert results stage 1
    with open(r"benchmark_tests/test_multi_stage_dynamic_to_static/output_/json_output.json") as f:
        expected_data_stage1 = json.load(f)

    with open(os.path.join(input_folder, "output/json_output.json")) as f:
        calculated_data_stage1 = json.load(f)

    # Check if the expected displacements and velocities in stage 1 are equal to the calculated ones
    TestUtils.assert_dictionary_almost_equal(expected=expected_data_stage1, actual=calculated_data_stage1)

    # Assert results stage 2
    with open(r"benchmark_tests/test_multi_stage_dynamic_to_static/output_/json_output_stage_2.json") as f:
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
        ax[0].set_xlabel("time [s]")
        ax[0].plot(
            merged_expected_data["TIME"],
            merged_expected_data["NODE_5"]["DISPLACEMENT_X"],
            label="Expected",
        )
        ax[0].plot(
            merged_calculated_data["TIME"],
            merged_calculated_data["NODE_5"]["DISPLACEMENT_X"],
            label="Calculated",
        )
        ax[0].legend()

        ax[1].set_title("Displacements Y")
        ax[1].set_ylabel("displacement_y [m]")
        ax[1].set_xlabel("time [s]")
        ax[1].plot(
            merged_expected_data["TIME"],
            merged_expected_data["NODE_5"]["DISPLACEMENT_Y"],
            label="Expected",
        )
        ax[1].plot(
            merged_calculated_data["TIME"],
            merged_calculated_data["NODE_5"]["DISPLACEMENT_Y"],
            label="Calculated",
        )
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    rmtree(input_folder)
