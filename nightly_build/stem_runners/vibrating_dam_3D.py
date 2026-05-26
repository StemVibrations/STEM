import os
import sys
from pathlib import Path
from shutil import rmtree
import json

import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import PointLoad, LineLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg, Lu
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters, GaussPointOutput
from stem.stem import Stem


def run_vibrating_dam_3d(input_folder):
    """
    Kramer, S. L., Geotechnical earthquake engineering chapter 7.3.4, vibrating dam, shear beam approach
    """

    # Define geometry, conditions and material parameters

    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)

    feet_to_m = 0.3048

    # Specify material model

    DENSITY_SOLID = 1800
    POROSITY = 0

    shear_wave_velocity = 1200 * feet_to_m
    shear_modulus = DENSITY_SOLID * shear_wave_velocity**2

    # prevent volumetric deformation
    POISSON_RATIO = 0.499
    YOUNG_MODULUS = 2 * shear_modulus * (1 + POISSON_RATIO)
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the 3D block: x, y, z
    y_max = 150 * feet_to_m
    x_max = y_max * 3.5

    z_max = 4
    force = 1e6

    top_coordinate = (y_max * 2, y_max, 0.0)
    layer1_coordinates = [(0.0, 0.0, 0.0), top_coordinate, (x_max, 0.0, 0.0)]
    model.extrusion_length = z_max

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil")

    # calculate geometry coordinates of a quarter circle with radius 1

    # Define load
    load = LineLoad(active=[True, True, True], value=[force, 0, 0])
    load_coordinates = [top_coordinate, (top_coordinate[0], top_coordinate[1], z_max)]
    model.add_load_by_coordinates(load_coordinates, load, "line_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    no_vertical_displacement = DisplacementConstraint(is_fixed=[False, True, True], value=[0, 0, 0])

    # model.show_geometry(show_surface_ids=True, show_line_ids=True)
    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [4], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(3, [1], no_vertical_displacement, "no_vert_displacement")

    # model.set_element_size_of_group(0.015, "circular_load")
    model.set_mesh_size(element_size=2)
    model.mesh_settings.element_order = 2

    # Synchronize geometry
    model.synchronise_geometry()

    # Define project parameters
    # --------------------------------
    # Set up solver settings
    time_step = 0.001
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=2,
                                       delta_time=time_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)

    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.DYNAMIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     linear_solver_settings=Lu(),
                                     rayleigh_k=0,
                                     rayleigh_m=0)

    # Set up problem data
    problem = Problem(problem_name="vibrating_dam", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    # Result output settings
    json_output_parameters = JsonOutputParameters(time_step, [NodalOutput.DISPLACEMENT_X])

    model.add_output_settings_by_coordinates([top_coordinate], json_output_parameters, "json_output_top")

    # # uncomment to output at all nodes
    # model.add_output_settings(output_parameters=VtkOutputParameters(
    #     file_format="ascii",
    #     output_interval=100,
    #     nodal_results=[NodalOutput.DISPLACEMENT],
    #     gauss_point_results=[GaussPointOutput.CAUCHY_STRESS_VECTOR],
    #     output_control_type="step"),
    #                           part_name="porous_computational_model_part",
    #                           output_dir="output",
    #                           output_name="vtk_output")

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()


if __name__ == "__main__":
    input_folder = "benchmark_tests/test_vibrating_dam_3d/inputs_kratos"
    run_vibrating_dam_3d(input_folder)
