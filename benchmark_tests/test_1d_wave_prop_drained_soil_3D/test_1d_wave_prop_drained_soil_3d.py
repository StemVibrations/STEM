import os
import sys
import json
from shutil import rmtree

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import SurfaceLoad
from stem.table import Table
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Amgcl)
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal

SHOW_RESULTS = False


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 1

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

    # Specify the coordinates for the column: x:1m x y:10m
    layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 10, 0), (0, 10, 0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_column")

    # Boundary conditions and Loads
    load_coordinates = [(0.0, 10.0, 0), (1.0, 10.0, 0), (1.0, 10.0, 1), (0.0, 10.0, 1)]
    # Add table for the load in the mdpa file
    t = (0.0, 0.0025, 1)
    values = (0.0, -1000.0, -1000.0)
    LOAD_Y = Table(times=t, values=values)
    # Add line load
    surface_load = SurfaceLoad(active=[False, True, False], value=[0, LOAD_Y, 0])
    model.add_load_by_coordinates(load_coordinates, surface_load, "load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, True], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [2], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [1, 3, 6, 5], sym_parameters, "side_rollers")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.15)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.0025
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.4,
                                       delta_time=delta_time,
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
                                     linear_solver_settings=Amgcl(tolerance=1e-6),
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     rayleigh_k=6e-6,
                                     rayleigh_m=0.02)

    # Set up problem data
    problem = Problem(problem_name="test_1d_wave_prop_drained_soil", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.VELOCITY, NodalOutput.DISPLACEMENT]

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=10,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=[],
                                                                    output_control_type="step"),
                              output_dir="output",
                              output_name="vtk_output")

    model.add_output_settings_by_coordinates([[0, 5, 0], [1, 5, 0]],
                                             JsonOutputParameters(output_interval=delta_time,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=[]),
                                             "calculated_output",
                                             output_dir="output")

    # Define the kratos input folder
    input_folder = "benchmark_tests/test_1d_wave_prop_drained_soil_3D/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    if SHOW_RESULTS:
        with open(os.path.join(input_folder, "output/calculated_output.json")) as f:
            calculated_data_stage1 = json.load(f)

        import matplotlib.pyplot as plt
        from benchmark_tests.analytical_solutions.analytical_wave_prop import OneDimWavePropagation
        p = OneDimWavePropagation(nb_terms=50, nb_cycles=10)

        DENSITY_SOLID = 2700
        POROSITY = 0.3
        YOUNG_MODULUS = 50e6
        POISSON_RATIO = 0.3

        density = (1 - POROSITY) * DENSITY_SOLID  # Density of the soil
        M_modulus = (YOUNG_MODULUS * (1 - POISSON_RATIO)) / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
        p.properties(density, M_modulus, -1000, 10, int(100))
        p.solution()
        p.write_results()

        _, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(calculated_data_stage1["TIME"],
                   calculated_data_stage1["NODE_9"]["DISPLACEMENT_Y"],
                   label="numerical")
        ax[0].plot(calculated_data_stage1["TIME"],
                   calculated_data_stage1["NODE_10"]["DISPLACEMENT_Y"],
                   label="numerical")
        ax[0].plot(p.time, p.u[50, :], marker='o', markersize=2, label="analytical")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Displacement (m)")
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(calculated_data_stage1["TIME"], calculated_data_stage1["NODE_9"]["VELOCITY_Y"], label="numerical")
        ax[1].plot(calculated_data_stage1["TIME"], calculated_data_stage1["NODE_10"]["VELOCITY_Y"], label="numerical")
        ax[1].plot(p.time, p.v[50, :], marker='o', markersize=2, label="analytical")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Velocity (m/s)")
        ax[1].grid()
        ax[1].legend()
        plt.show()

    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_1d_wave_prop_drained_soil_3D/output_windows/output_vtk_full_model"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_1d_wave_prop_drained_soil_3D/output_linux/output_vtk_full_model"
    else:
        raise Exception("Unknown platform")

    result = assert_files_equal(expected_output_dir, os.path.join(input_folder, "output/output_vtk_full_model"))

    assert result is True
    rmtree(input_folder)
