import os
from shutil import rmtree

import numpy as np

from stem.model import Model
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated, StructuralMaterial
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, Output, VtkOutputParameters
from stem.stem import Stem
from benchmark_tests.utils import assert_floats_in_directories_almost_equal


def test_moving_load_on_track_on_soil():
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 10

    # add the track
    rail_parameters = EulerBeam(ndim=ndim,
                                YOUNG_MODULUS=30e9,
                                POISSON_RATIO=0.2,
                                DENSITY=7200,
                                CROSS_AREA=0.01,
                                I33=1e-4,
                                I22=1e-4,
                                TORSIONAL_INERTIA=2e-4)
    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 750e6, 1],
                                              NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                              NODAL_DAMPING_COEFFICIENT=[1, 750e3, 1],
                                              NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    origin_point = np.array([1.0, 3.0, 0.0])
    direction_vector = np.array([0, 0, 1])
    rail_pad_thickness = 0.025

    # create a straight track with rails, sleepers and rail pads
    model.generate_straight_track(0.5, 21, rail_parameters, sleeper_parameters, rail_pad_parameters, rail_pad_thickness,
                                  origin_point, direction_vector, "rail_track_1")

    # add hanging sleepers
    damaged_rail_pads = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e3, 0],
                                            NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                            NODAL_DAMPING_COEFFICIENT=[0, 750e1, 0],
                                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    # define which geometry ids are to be redefined
    damaged_material = StructuralMaterial(name="damaged_rail_pads", material_parameters=damaged_rail_pads)
    model.split_model_part("rail_pads_rail_track_1", "rail_pads_rail_track_1_hanging", [50, 51], damaged_material)

    moving_load = MovingLoad(load=[0.0, -10000.0, 0.0],
                             direction=[1, 1, 1],
                             velocity=10,
                             origin=[1.0, 3 + rail_pad_thickness, 0.0],
                             offset=0.0)

    model.add_load_on_line_model_part("rail_track_1", moving_load, "moving_load")

    # model.show_geometry(show_surface_ids=True, show_point_ids=True)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    sleeper_point_ids = list(model.body_model_parts[1].geometry.points.keys())
    model.add_boundary_condition_by_geometry_ids(0, sleeper_point_ids, no_displacement_parameters, "base_fixed")

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=1.0,
                                       delta_time=0.01,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.01,
                                     rayleigh_m=0.0001)

    # Set up problem data
    problem = Problem(problem_name="test_moving_load_on_track_hanging_sleepers",
                      number_of_threads=4,
                      settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.TOTAL_DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    vtk_output_process = Output(part_name="porous_computational_model_part",
                                output_name="vtk_output",
                                output_dir="output",
                                output_parameters=VtkOutputParameters(file_format="ascii",
                                                                      output_interval=10,
                                                                      nodal_results=nodal_results,
                                                                      gauss_point_results=gauss_point_results,
                                                                      output_control_type="step"))

    model.output_settings = [vtk_output_process]

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=1)

    input_folder = "benchmark_tests/test_moving_load_on_track_hanging_sleepers/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    expected_output_dir = "benchmark_tests/test_moving_load_on_track_hanging_sleepers/output_/output_vtk_porous_computational_model_part"

    assert_floats_in_directories_almost_equal(
        expected_output_dir, os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"), 4)

    rmtree(input_folder)
