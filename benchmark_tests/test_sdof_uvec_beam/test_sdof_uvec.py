import os
import numpy as np
import numpy.testing as npt
import pytest

from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import EulerBeam, StructuralMaterial
from stem.load import UvecLoad
from stem.boundary import DisplacementConstraint, RotationConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem
from shutil import rmtree, copytree

from benchmark_tests.analytical_solutions.moving_vehicle import TwoDofVehicle
from benchmark_tests.utils import assert_floats_in_files_almost_equal

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
    Test for a single degree of freedom UVEC moving on a beam structure.

    Args:
        - ndim (int): Dimension of the problem (2 or 3).
        - axis_index (int): Index of the axis along which the UVEC moves (0 for x, 1 for y, 2 for z).
    """
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    model = Model(ndim)

    velocity = 100 / 3.6

    # Specify beam material model
    YOUNG_MODULUS = 2.87e9
    POISSON_RATIO = 0.30000
    DENSITY = 2303
    CROSS_AREA = 0.1
    I33 = 0.29
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I33, I33, 2 * I33)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    total_length = 25
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

    # Define UVEC load
    uvec_parameters = {
        "n_carts": 1,
        "cart_inertia": 0,
        "cart_mass": 0,
        "cart_stiffness": 0,
        "cart_damping": 0,
        "bogie_distances": [0],
        "bogie_inertia": 0,
        "bogie_mass": 3000,
        "wheel_distances": [0],
        "wheel_mass": 5750,
        "wheel_stiffness": 1595e5,
        "wheel_damping": 1000,
        "contact_coefficient": 9.1e-8,
        "contact_power": 1,
        "gravity_axis": 1,
        "file_name": r"test.csv"
    }

    uvec_load = UvecLoad(direction_signs=[1, 1, 1],
                         velocity=velocity,
                         origin=[0.0, 0, 0],
                         wheel_configuration=[0.0],
                         uvec_file=r"uvec_ten_dof_vehicle_2D/uvec.py",
                         uvec_function_name="uvec",
                         uvec_parameters=uvec_parameters)
    model.add_load_by_geometry_ids([1], uvec_load, "uvec_load")

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()
    # model.show_geometry(show_line_ids=True, show_point_ids=True)

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=2.5)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.002
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.9,
                                       delta_time=delta_time,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-3,
                                                            displacement_absolute_tolerance=1.0e-8)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.001,
                                     rayleigh_m=0.01)

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

    model.add_output_settings_by_coordinates([middle_coordinate],
                                             JsonOutputParameters(output_interval=delta_time,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=gauss_point_results),
                                             "json_output")

    input_folder = r"benchmark_tests/test_sdof_uvec_beam/input_kratos"
    # copy uvec to input folder
    os.makedirs(input_folder, exist_ok=True)
    copytree(r"benchmark_tests/test_sdof_uvec_beam/uvec_ten_dof_vehicle_2D",
             os.path.join(input_folder, "uvec_ten_dof_vehicle_2D"),
             dirs_exist_ok=True)

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    # middle beams displacement
    with open(os.path.join(input_folder, "json_output.json"), "r") as f:
        import json
        data = json.load(f)
    time_kratos = data["TIME"]
    disp_kratos_middle_beam = data["NODE_3"]["DISPLACEMENT_Y"]

    if PLOT_RESULTS:
        import matplotlib.pyplot as plt

        # read test.csv file with the numerical solution
        with open(os.path.join(input_folder, "test.csv"), "r") as f:
            test_data = f.read().splitlines()
        test_data = np.array([list(map(float, t.split(";"))) for t in test_data])

        # Extract unique time values and corresponding displacements
        unique_time_values = np.unique(test_data[:, 0])

        displacement_top = []
        displacement_bottom = []
        time = []
        for time_value in unique_time_values:
            # Find the corresponding displacement for each unique time value
            index = np.where(test_data[:, 0] == time_value)[0][-1]
            displacement_top.append(np.array(test_data)[index, 2])
            displacement_bottom.append(np.array(test_data)[index, 3])
            time.append(np.array(test_data)[index, 0])

        # calculate analytical solution
        ss = TwoDofVehicle()
        ss.vehicle(uvec_parameters["bogie_mass"], uvec_parameters["wheel_mass"], velocity,
                   uvec_parameters["wheel_stiffness"], uvec_parameters["wheel_damping"])
        ss.beam(YOUNG_MODULUS, I33, DENSITY, CROSS_AREA, total_length)
        ss.compute()

        # plot numerical and analytical solution
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(time_kratos, disp_kratos_middle_beam, label="kraton beam", color='b')
        ax[1].plot(time, displacement_top, label="kraton body", color='b')
        ax[0].plot(ss.time, -ss.displacement[:, 0], color='r', linestyle="--", label="analytical")
        ax[1].plot(ss.time, -ss.displacement[:, 1], color='r', linestyle="--", label="analytical")
        ax[0].set_ylabel("Displacement beam [m]")
        ax[1].set_ylabel("Displacement bogie [m]")
        ax[0].legend()
        ax[1].legend()

        # plot wheel displacement, without analytical solution
        ax[2].plot(time, displacement_bottom, label="kraton wheel", color='b')
        ax[2].set_ylabel("Displacement wheel [m]")
        ax[2].set_xlabel("Time [s]")
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    # test output
    with open(f"benchmark_tests/test_sdof_uvec_beam/output_/expected_disp_middle_beam_{ndim}d.json", "r") as f:
        data = json.load(f)
    expected_disp_middle_beam = data["NODE_3"]["DISPLACEMENT_Y"]

    npt.assert_array_almost_equal(disp_kratos_middle_beam, expected_disp_middle_beam, decimal=6)

    assert_floats_in_files_almost_equal(
        f"benchmark_tests/test_sdof_uvec_beam/output_/expected_vehicle_output_{ndim}d.csv",
        os.path.join(input_folder, "test.csv"),
        decimal=4)

    rmtree(input_folder)
