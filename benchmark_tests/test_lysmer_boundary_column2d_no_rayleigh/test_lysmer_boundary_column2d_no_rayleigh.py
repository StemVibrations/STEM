import os
import json
from shutil import rmtree

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.boundary import AbsorbingBoundary
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy)

from stem.output import NodalOutput, JsonOutputParameters
from stem.stem import Stem

from tests.utils import TestUtils

SHOW_RESULTS = False


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    # Specify material model
    # Linear elastic drained soil with a Density of 2650, a Young's modulus of 10.0e7,
    # a Poisson ratio of 0.2 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2650
    POROSITY = 0.3
    YOUNG_MODULUS = 10.0e7
    POISSON_RATIO = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:1m x y:10m
    layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 10, 0), (0, 10, 0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil")

    # Boundary conditions and Loads
    load_coordinates = [(0.0, 10.0, 0), (1.0, 10.0, 0)]

    # Add line load
    line_load = LineLoad(active=[False, True, False], value=[0, -10, 0])
    model.add_load_by_coordinates(load_coordinates, line_load, "load")

    # Define absorbing boundary condition
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=1000.0)

    # Define displacement conditions
    displacement_parameters = DisplacementConstraint(is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, [1], absorbing_boundaries_parameters, "abs")
    model.add_boundary_condition_by_geometry_ids(1, [2, 4], displacement_parameters, "sides")

    # Synchronize geometry
    model.synchronise_geometry()

    # Show geometry and geometry ids
    # model.show_geometry(show_line_ids=True)

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=2)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.005
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.2,
                                       delta_time=delta_time,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-6,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     rayleigh_k=0,
                                     rayleigh_m=0)

    # Set up problem data
    problem = Problem(problem_name="test_lysmer_boundary_column2d_no_rayleigh",
                      number_of_threads=1,
                      settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]
    # Gauss point results
    gauss_point_results = []

    model.add_output_settings_by_coordinates([(0, 5, 0)],
                                             JsonOutputParameters(output_interval=delta_time,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=gauss_point_results),
                                             "calculated_json_output")

    # Define the kratos input folder
    input_folder = f"benchmark_tests/test_lysmer_boundary_column2d_no_rayleigh/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    model.mesh_settings.element_order = 2
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    with open(os.path.join(input_folder, "calculated_json_output.json")) as f:
        calculated_output = json.load(f)

    with open(r"benchmark_tests/test_lysmer_boundary_column2d_no_rayleigh/_output/expected_json_output.json") as f:
        expected_output = json.load(f)

    if SHOW_RESULTS:
        import matplotlib.pyplot as plt
        time = calculated_output["TIME"]
        calculated_displacement = calculated_output["NODE_5"]["DISPLACEMENT_Y"]
        expected_displacement = expected_output["NODE_5"]["DISPLACEMENT_Y"]

        calculated_velocity = calculated_output["NODE_5"]["VELOCITY_Y"]
        expected_velocity = expected_output["NODE_5"]["VELOCITY_Y"]

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time, calculated_displacement, label='Calculated Displacement', marker='o')
        plt.plot(time, expected_displacement, label='Expected Displacement', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(time, calculated_velocity, label='Calculated Velocity', marker='o')
        plt.plot(time, expected_velocity, label='Expected Velocity', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    TestUtils.assert_dictionary_almost_equal(calculated_output, expected_output)

    rmtree(input_folder)
