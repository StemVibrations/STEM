from pathlib import Path
from shutil import rmtree
import json

import numpy as np
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import PointLoad, SurfaceLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters, GaussPointOutput
from stem.stem import Stem

from benchmark_tests.analytical_solutions.boussinesq import Boussinesq

SHOW_RESULTS = False


def discretise_quarter_circle(radius: float, n_points: int) -> np.ndarray:
    """
    Discretised quarter circle of radius r in the xy-plane.

    Args:
        - radius (float): Radius of the quarter circle
        - n_points (int): Number of discretisation points

    Returns:
        - np.ndarray: Array of shape (n_points, 3) containing the coordinates of the quarter circle
    """
    theta = np.linspace(0, np.pi / 2, n_points)
    x = radius * np.cos(theta)
    y = np.zeros_like(theta)
    z = radius * np.sin(theta)

    return np.column_stack((x, y, z))


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)

    # Specify material model
    DENSITY_SOLID = 2000
    POROSITY = 0
    YOUNG_MODULUS = 20e6
    POISSON_RATIO = 0.3
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the 3D block: x, y, z
    x_max = 10
    y_max = 30
    z_max = 10
    force = -10e3
    layer1_coordinates = [(0.0, 0.0, 0.0), (x_max, 0.0, 0.0), (x_max, y_max, 0.0), (0.0, y_max, 0.0)]
    model.extrusion_length = z_max

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil")

    # calculate geometry coordinates of a quarter circle with radius 1

    load_radius = 0.1
    surface_load_coords = discretise_quarter_circle(load_radius, 100)
    surface_load_coords = np.vstack((surface_load_coords, [0, 0, 0]))
    surface_load_coords[:, 1] = y_max  # move to y=y_max plane

    # Define load
    load = SurfaceLoad(active=[True, True, True], value=[0, force, 0])
    model.add_load_by_coordinates(surface_load_coords, load, "circular_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement_parameters_x = DisplacementConstraint(is_fixed=[True, False, False], value=[0, 0, 0])
    roller_displacement_parameters_z = DisplacementConstraint(is_fixed=[False, False, True], value=[0, 0, 0])
    abs_boundary_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=10)

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, 0, z_max)], no_displacement_parameters,
                                          "base_fixed")
    model.add_boundary_condition_on_plane([(0, 0, 0), (0, y_max, 0), (0, y_max, z_max)],
                                          roller_displacement_parameters_x, "sides_roler_x=0")
    model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, y_max, 0)],
                                          roller_displacement_parameters_z, "sides_roler_z=0")
    model.add_boundary_condition_on_plane([(x_max, 0, 0), (x_max, y_max, 0), (x_max, y_max, z_max)],
                                          roller_displacement_parameters_x, "abs_x=x_max")
    model.add_boundary_condition_on_plane([(0, 0, z_max), (x_max, 0, z_max), (x_max, y_max, z_max)],
                                          roller_displacement_parameters_z, "abs_z=z_max")

    model.set_element_size_of_group(0.015, "circular_load")
    model.set_mesh_size(element_size=1)
    model.mesh_settings.element_order = 2

    # Synchronize geometry
    model.synchronise_geometry()

    # Define project parameters
    # --------------------------------
    # Set up solver settings
    time_step = 0.1
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.15,
                                       delta_time=time_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)

    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.QUASI_STATIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     linear_solver_settings=Cg())

    # Set up problem data
    problem = Problem(problem_name="Boussinesq", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    # Result output settings
    json_output_parameters_surface = JsonOutputParameters(time_step, [NodalOutput.DISPLACEMENT])

    json_output_parameters_depth = JsonOutputParameters(time_step, [NodalOutput.CAUCHY_STRESS_VECTOR])

    # displacements at surface load

    output_x_coordinates_surface = np.linspace(0, load_radius * 3, 20)
    output_coordinates_surface = [(x, y_max, 0) for x in output_x_coordinates_surface]

    model.add_output_settings_by_coordinates(output_coordinates_surface, json_output_parameters_surface,
                                             "json_output_surface")

    output_y_coordinates = np.linspace(y_max - 1e-3, y_max - 2, 100)
    output_coordinates_below_load = [(0, y, 0) for y in output_y_coordinates]

    model.add_output_settings_by_coordinates(output_coordinates_below_load, json_output_parameters_depth,
                                             "json_output_depth")

    # uncomment to output at all nodes
    model.add_output_settings(output_parameters=VtkOutputParameters(
        file_format="ascii",
        output_interval=1,
        nodal_results=[NodalOutput.DISPLACEMENT],
        gauss_point_results=[GaussPointOutput.CAUCHY_STRESS_VECTOR],
        output_control_type="step"),
                              part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output")

    # Write KRATOS input files
    # --------------------------------
    input_folder = "benchmark_tests/test_boussinesq_3d/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    output_vert_displacements_surface_file_name = Path(input_folder) / "json_output_surface.json"
    output_vert_stresses_depth_file_name = Path(input_folder) / "json_output_depth.json"

    with open(output_vert_displacements_surface_file_name, 'r') as f:
        output_vert_displacements_surface = json.load(f)

    with open(output_vert_stresses_depth_file_name, 'r') as f:
        output_vert_stresses_depth = json.load(f)

    analytical_sol = Boussinesq(YOUNG_MODULUS, POISSON_RATIO, load_radius, force)
    calculated_x_coordinates_surface = []
    calculated_disp_results = []
    analytical_disp_results = []
    for key, value in output_vert_displacements_surface.items():
        if key != "TIME":
            calculated_x_coordinates_surface.append(value["COORDINATES"][0])
            calculated_disp_results.append(value["DISPLACEMENT_Y"][0])
            analytical_disp_results.append(
                analytical_sol.calculate_vertical_displacement_on_surface(value["COORDINATES"][0]))

    sort_idx = np.argsort(calculated_x_coordinates_surface)
    calculated_x_coordinates_surface_sorted = np.array(calculated_x_coordinates_surface)[sort_idx]
    calculated_disp_results_sorted = np.array(calculated_disp_results)[sort_idx]
    analytical_disp_results_sorted = np.array(analytical_disp_results)[sort_idx]

    calculated_y_coordinates = []
    calculated_stress_results = []
    analytical_stress_results = []
    for key, value in output_vert_stresses_depth.items():
        if key != "TIME":
            calculated_y_coordinates.append(value["COORDINATES"][1])
            calculated_stress = value["CAUCHY_STRESS_VECTOR"][0][1]
            calculated_stress_results.append(calculated_stress)
            depth = y_max - value["COORDINATES"][1]
            analytical_stress_results.append(analytical_sol.calculate_vertical_stress_below_load_centre(depth))

    sort_idx = np.argsort(calculated_y_coordinates)
    calculated_y_coordinates_sorted = np.array(calculated_y_coordinates)[sort_idx]
    calculated_stress_results_sorted = np.array(calculated_stress_results)[sort_idx]
    analytical_stress_results_sorted = np.array(analytical_stress_results)[sort_idx]

    # Compare displacements results within 10% relative tolerance
    assert np.allclose(calculated_disp_results_sorted, analytical_disp_results_sorted, rtol=1e-1)

    # Compare stress results within 2% relative tolerance
    assert np.allclose(calculated_stress_results_sorted, analytical_stress_results_sorted, rtol=2e-2)

    if SHOW_RESULTS:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        # Plot displacement results
        ax1.plot(calculated_x_coordinates_surface_sorted,
                 calculated_disp_results_sorted,
                 'o-',
                 label='Calculated Displacement Y')
        ax1.plot(calculated_x_coordinates_surface_sorted,
                 analytical_disp_results_sorted,
                 'x--',
                 label='Analytical Displacement Y')
        ax1.set_xlabel('X-Coordinate [m]')
        ax1.set_ylabel('Displacement Y [m]')
        ax1.set_title('Vertical Displacement at Surface')
        ax1.grid()
        ax1.legend()

        ax2.plot(calculated_y_coordinates_sorted, calculated_stress_results_sorted, 'o-', label='Calculated stress yy')
        ax2.plot(calculated_y_coordinates_sorted, analytical_stress_results_sorted, 'x--', label='Analytical stress yy')

        ax2.set_xlabel('Y-Coordinate [m]')
        ax2.set_ylabel('Stress yy [Pa]')
        ax2.set_title('Vertical Stress Below Load')
        ax2.grid()
        ax2.legend()

        fig.tight_layout()
        plt.show()

    rmtree(input_folder)
