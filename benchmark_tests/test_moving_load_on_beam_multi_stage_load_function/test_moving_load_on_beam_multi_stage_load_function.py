import os
import json

from numpy.testing import assert_array_almost_equal

from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import *
from stem.load import MovingLoad
from stem.boundary import RotationConstraint
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy)
from stem.output import NodalOutput, GaussPointOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem

from shutil import rmtree

SHOW_RESULTS = False


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    # Specify beam material model
    YOUNG_MODULUS = 210000000000
    POISSON_RATIO = 0.30000
    DENSITY = 7850
    CROSS_AREA = 0.01
    I22 = 0.0001
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I22)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    beam_coordinates = [(0, 0, 0), (1, 0, 0)]
    # Create the beam
    gmsh_input = {name: {"coordinates": beam_coordinates, "ndim": 1}}
    # check if extrusion length is specified in 3D
    model.gmsh_io.generate_geometry(gmsh_input, "")
    #
    # create body model part
    body_model_part = BodyModelPart(name)
    body_model_part.material = structural_material

    # set the geometry of the body model part
    body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, name)
    model.body_model_parts.append(body_model_part)

    # Show geometry and geometry ids
    # model.show_geometry(show_point_ids=True, show_line_ids=True)

    # Define moving load
    load = ["0.0", "-10000*t", "0.0"]
    moving_load = MovingLoad(load=load, direction_signs=[1, 1, 1], velocity=1.0, origin=[0.0, 0.0, 0.0], offset=0.0)

    model.add_load_by_geometry_ids([1], moving_load, "moving_load")

    # Define rotation boundary condition
    rotation_boundaries_parameters = RotationConstraint(is_fixed=[True, True, True], value=[0, 0, 0])

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(0, [2], rotation_boundaries_parameters, "rotation")
    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()

    json_output = JsonOutputParameters(0.01, [NodalOutput.DISPLACEMENT_Y])
    model.add_output_settings_by_coordinates([[0.5, 0.0, 0.0]], json_output, "json_output")
    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.05)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.5,
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
                                     is_stiffness_matrix_constant=False,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     rayleigh_k=0.001,
                                     rayleigh_m=0.1)

    # Set up problem data
    problem = Problem(problem_name="calculate_moving_load_on_beam", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # uncomment the following lines to write VTK output files
    # # Nodal results
    # nodal_results = [NodalOutput.DISPLACEMENT]
    # # Gauss point results
    # gauss_point_results = [GaussPointOutput.FORCE]
    #
    # # Define the output process
    # model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
    #                                                                 output_interval=1,
    #                                                                 nodal_results=nodal_results,
    #                                                                 gauss_point_results=gauss_point_results,
    #                                                                 output_control_type="step"),
    #                           output_dir="output",
    #                           output_name="vtk_output")

    input_folder = "benchmark_tests/test_moving_load_on_beam_multi_stage_load_function/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stage2 = stem.create_new_stage(0.01, 0.5)
    stem.add_calculation_stage(stage2)

    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    with open(r"benchmark_tests/test_moving_load_on_beam_multi_stage_load_function/inputs_kratos/json_output.json",
              'r') as json_file:
        json_output1 = json.load(json_file)
        time = json_output1["TIME"]
        displacement_y = json_output1["NODE_3"]["DISPLACEMENT_Y"]

    with open(
            r"benchmark_tests/test_moving_load_on_beam_multi_stage_load_function/inputs_kratos/json_output_stage_2.json",
            'r') as json_file:
        json_output2 = json.load(json_file)

        time = time + json_output2["TIME"]
        displacement_y = displacement_y + json_output2["NODE_3"]["DISPLACEMENT_Y"]

    with open(r"benchmark_tests/test_moving_load_on_beam_multi_stage_load_function/output_/expected_output.json",
              'r') as json_file:
        expected_output = json.load(json_file)
        expected_time = expected_output["TIME"]
        expected_displacement_y = expected_output["NODE_3"]["DISPLACEMENT_Y"]

    assert_array_almost_equal(time, expected_time)
    assert_array_almost_equal(displacement_y, expected_displacement_y)

    if SHOW_RESULTS:

        import matplotlib.pyplot as plt
        plt.plot(time, displacement_y)
        plt.plot(expected_time, expected_displacement_y, "--")
        plt.xlabel("Time [s]")
        plt.ylabel("Displacement Y [m]")
        plt.legend(["Calculated", "Expected"])
        plt.show()

    rmtree(input_folder)
