import os
import numpy as np

from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import EulerBeam, StructuralMaterial
from stem.load import UvecLoad
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy,
                         NewtonRaphsonStrategy)
from stem.output import VtkOutputParameters, Output, NodalOutput
from stem.stem import Stem
from shutil import rmtree, copytree

from benchmark_tests.utils import assert_floats_in_files_almost_equal

PLOT_RESULTS = False


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    velocity = 100 / 3.6

    # Specify beam material model
    YOUNG_MODULUS = 2.87e9
    POISSON_RATIO = 0.30000
    DENSITY = 2303
    CROSS_AREA = 0.1
    I22 = 0.29
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I22)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    beam_coordinates = [(0, 0, 0), (25, 0, 0)]
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
        "contact_coefficient": 9.1e-7,
        "contact_power": 1,
        "gravity_axis": 1,
        "file_name": r"calculated_results.txt"
    }

    uvec_load = UvecLoad(direction=[1, 1, 0],
                         velocity=0,
                         origin=[12.5, 0, 0],
                         wheel_configuration=[0.0],
                         uvec_file=r"uvec_ten_dof_vehicle_2D/uvec.py",
                         uvec_function_name="uvec_static",
                         uvec_parameters=uvec_parameters)
    model.add_load_by_geometry_ids([1], uvec_load, "uvec_load")

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")

    # uncomment to enable output in VTK format
    # # Define the output process
    # vtk_output_process = Output(part_name="porous_computational_model_part",
    #                             output_name="vtk_output",
    #                             output_dir="output",
    #                             output_parameters=VtkOutputParameters(file_format="ascii",
    #                                                                   output_interval=10,
    #                                                                   nodal_results=[NodalOutput.DISPLACEMENT],
    #                                                                   gauss_point_results=[],
    #                                                                   output_control_type="step"))
    # model.output_settings = [vtk_output_process]
    # Synchronize geometry
    model.synchronise_geometry()
    # model.show_geometry(show_line_ids=True, show_point_ids=True)

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=0.1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=-0.15,
                                       end_time=0.45,
                                       delta_time=0.15,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-5,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=False,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=NewtonRaphsonStrategy(),
                                     rayleigh_k=0.000,
                                     rayleigh_m=0.00)

    # Set up problem data
    problem = Problem(problem_name="uvec_sdof", number_of_threads=1, settings=solver_settings)
    model.project_parameters = problem

    input_folder = r"benchmark_tests/test_sdof_uvec_beam_multistage/input_kratos"
    # copy uvec to input folder
    os.makedirs(input_folder, exist_ok=True)
    copytree(r"benchmark_tests/test_sdof_uvec_beam_multistage/uvec_ten_dof_vehicle_2D",
             os.path.join(input_folder, "uvec_ten_dof_vehicle_2D"),
             dirs_exist_ok=True)

    # make sure the uvec output is cleared as results are appended to the file at each iteration
    if os.path.exists(os.path.join(input_folder, uvec_parameters["file_name"])):
        os.remove(os.path.join(input_folder, uvec_parameters["file_name"]))

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    model_stage_2 = stem.create_new_stage(0.0005, 0.45)
    model_stage_2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    model_stage_2.project_parameters.settings.rayleigh_k = 1e-6
    model_stage_2.project_parameters.settings.rayleigh_m = 0.02

    model_stage_2.process_model_parts[0].parameters.uvec_function_name = "uvec"
    model_stage_2.process_model_parts[0].parameters.velocity = velocity

    stem.add_calculation_stage(model_stage_2)

    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    if PLOT_RESULTS:
        import matplotlib.pyplot as plt

        # read results file
        with open(os.path.join(input_folder, "calculated_results.txt"), "r") as f:
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

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(time, displacement_top, label="kraton body", color='b')
        ax[1].plot(time, displacement_bottom, label="kraton wheel", color='r')
        ax[0].set_ylabel("Displacement beam [m]")
        ax[1].set_ylabel("Displacement bogie [m]")
        ax[1].set_xlabel("Time [s]")
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    # test output
    assert_floats_in_files_almost_equal("benchmark_tests/test_sdof_uvec_beam_multistage/output_/expected_results.txt",
                                        os.path.join(input_folder, uvec_parameters["file_name"]),
                                        decimal=3)

    rmtree(input_folder)
