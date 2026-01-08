import os
import json

import numpy as np
import pytest

from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import EulerBeam, StructuralMaterial
from stem.load import LineLoad
from stem.boundary import DisplacementConstraint, RotationConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy, Lu
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem
from shutil import rmtree

PLOT_RESULTS = False


@pytest.mark.parametrize(
    "ndim, axis_index",
    [
        (2, 0),
        (3, 0),
        (3, 2),
    ],
)
def test_stem(ndim: int, axis_index: int):
    """
    Test for a simply supported beam and unloading of a distributed load.

    Args:
        - ndim (int): Dimension of the problem (2 or 3).
        - axis_index (int): Index of the axis along which the UVEC moves (0 for x, 1 for y, 2 for z).
    """

    # Specify dimension and initiate the model
    model = Model(ndim)

    # Specify beam material model
    YOUNG_MODULUS = (2000 / np.pi)**2
    POISSON_RATIO = 0.3
    DENSITY = 1
    CROSS_AREA = 1
    I33 = 1
    total_length = 25
    q = 1  # uniform load in N/m

    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I33, I33, I33)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    middle_coordinate = [0, 0, 0]
    middle_coordinate[axis_index] = total_length / 2

    beam_coordinates = [[0, 0, 0], [0, 0, 0]]
    beam_coordinates[1][axis_index] = total_length
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

    # Define loads
    line_load = LineLoad(active=[False, True, False], value=[0, -q, 0])
    model.add_load_by_geometry_ids([1], line_load, "line_load")

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # no torsion
    is_active = [False, False, False]
    is_active[axis_index] = True
    rot_boundary_parameters = RotationConstraint(active=is_active, is_fixed=is_active, value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")
    model.add_boundary_condition_by_geometry_ids(0, [1], rot_boundary_parameters, "rotation")

    # Synchronize geometry
    model.synchronise_geometry()
    # model.show_geometry(show_line_ids=True, show_point_ids=True)

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=1.25 / 4)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.25
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.5,
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
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=NewtonRaphsonStrategy(),
                                     linear_solver_settings=Lu(),
                                     rayleigh_k=0,
                                     rayleigh_m=0)

    # Set up problem data
    problem = Problem(problem_name="uvec_sdof", number_of_threads=1, settings=solver_settings)
    model.project_parameters = problem

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # uncomment code below to output vtk files
    # # Define the output process
    # vtk_output_process = Output(output_name="vtk_output",
    #                             output_dir="output",
    #                             output_parameters=VtkOutputParameters(file_format="ascii",
    #                                                                   output_interval=1,
    #                                                                   nodal_results=nodal_results,
    #                                                                   gauss_point_results=gauss_point_results,
    #                                                                   output_control_type="step"))
    #
    # model.output_settings = [vtk_output_process]

    dynamic_delta_time = 0.00125
    model.add_output_settings_by_coordinates([middle_coordinate],
                                             JsonOutputParameters(output_interval=dynamic_delta_time,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=gauss_point_results),
                                             "json_output")

    input_folder = r"benchmark_tests/test_simply_supported_beam_dynamic/input_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)

    stage2 = stem.create_new_stage(dynamic_delta_time, 2)
    stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    stage2.get_model_part_by_name("line_load").parameters.value = [0, 0, 0]
    stem.stages.append(stage2)

    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    # expected frequency and max displacement
    expected_f = 1 / (2 * np.pi) * (np.pi / total_length)**2 * np.sqrt(YOUNG_MODULUS * I33 / (DENSITY * CROSS_AREA))
    expected_max_disp = 5 * q * total_length**4 / (384 * YOUNG_MODULUS * I33)

    period = 1 / expected_f

    # Compare results
    with open(os.path.join(input_folder, "json_output_stage_2.json")) as f:
        calculated_data = json.load(f)

    max_disp_time_index = int(period / dynamic_delta_time) + 1  # +1 to account for initial time step at t=0

    # check that the expected maximum displacement is reached at the expected time
    assert pytest.approx(calculated_data["NODE_3"]["DISPLACEMENT_Y"][int(max_disp_time_index / 2)],
                         1e-3) == expected_max_disp
    assert pytest.approx(calculated_data["NODE_3"]["DISPLACEMENT_Y"][max_disp_time_index], 1e-3) == -expected_max_disp

    if PLOT_RESULTS:
        import matplotlib.pyplot as plt

        with open(os.path.join(input_folder, "json_output_stage_2.json")) as f:
            calculated_data = json.load(f)

        time = calculated_data["TIME"]
        displacement = calculated_data["NODE_3"]["DISPLACEMENT_Y"]

        plt.plot(time, displacement)

        # set vertical line at 1/f
        plt.axvline(x=1 / expected_f + 0.5 + dynamic_delta_time, color='r', linestyle='--', label='1/f')
        plt.axhline(y=expected_max_disp, color='g', linestyle='--')
        plt.axhline(y=-expected_max_disp, color='g', linestyle='--')

        plt.xlabel("Time (s)")
        plt.ylabel("Displacement Y (m)")
        plt.title("Displacement at Mid-span of Simply Supported Beam with Moving Load")
        plt.legend()
        plt.grid()
        plt.show()

    rmtree(input_folder)
