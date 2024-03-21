import pytest
import numpy as np

from stem.model import Model
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, Output, VtkOutputParameters
from stem.stem import Stem


@pytest.mark.skip(reason="This test is not working yet")
def test_moving_load_on_track():
    model = Model(3)

    rail_parameters = EulerBeam(ndim=3,
                                YOUNG_MODULUS=30e6,
                                POISSON_RATIO=0.2,
                                DENSITY=7200,
                                CROSS_AREA=0.01,
                                I33=1e-4,
                                I22=1e-4,
                                TORSIONAL_INERTIA=1e-4)
    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1],
                                              NODAL_ROTATIONAL_STIFFNESS=[1, 1, 1],
                                              NODAL_DAMPING_COEFFICIENT=[1, 1, 1],
                                              NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[1, 1, 1])
    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=1,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    origin_point = np.array([0.0, 3.0, 1.0])
    direction_vector = np.array([1, 0, 0])

    # create a straight track with rails, sleepers and rail pads
    model.generate_straight_track(0.6, 20, rail_parameters, sleeper_parameters, rail_pad_parameters, origin_point,
                                  direction_vector, "rail_track_1")
    model.synchronise_geometry()

    no_displacement_boundary = DisplacementConstraint(active=[True, True, True],
                                                      is_fixed=[True, True, True],
                                                      value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(
        1, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], no_displacement_boundary,
        "base_fixed")

    model.synchronise_geometry()

    moving_load = MovingLoad(load=[0.0, 10.0, 0.0],
                             direction=[1, 1, 1],
                             velocity=5,
                             origin=[0.0, 3.001, 1.0],
                             offset=0.75)

    model.add_load_on_line_model_part("rail_track_1", moving_load, "moving_load")

    model.synchronise_geometry()

    # model.gmsh_io.generate_mesh(3, open_gmsh_gui=True)

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=1.0,
                                       delta_time=0.01,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)
    strategy_type = NewtonRaphsonStrategy(min_iterations=6, max_iterations=15, number_cycles=100)
    scheme_type = NewmarkScheme(newmark_beta=0.25, newmark_gamma=0.5, newmark_theta=0.5)
    linear_solver_settings = Amgcl(tolerance=1e-8, max_iteration=500, scaling=False)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=False,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy_type,
                                     scheme=scheme_type,
                                     linear_solver_settings=linear_solver_settings,
                                     rayleigh_k=0.0,
                                     rayleigh_m=0.0)

    # Set up problem data
    problem = Problem(problem_name="calculate_moving_load_on_embankment_3d2",
                      number_of_threads=1,
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

    stem = Stem(model, "benchmark_moving_load2")
    stem.write_all_input_files()
    stem.run_calculation()
