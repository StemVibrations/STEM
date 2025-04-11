import os
import sys
from shutil import rmtree, copytree

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedLaw
from stem.load import UvecLoad
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, Amgcl)
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.stem import Stem
import UVEC.uvec_ten_dof_vehicle_2D as uvec

from benchmark_tests.utils import assert_files_equal


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)

    # Specify material model
    solid_density = 2650
    porosity = 0.3
    young_modulus = 30e6
    poisson_ratio = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density, POROSITY=porosity)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus, POISSON_RATIO=poisson_ratio)
    retention_parameters1 = SaturatedLaw()
    material_soil1 = SoilMaterial("soil1", soil_formulation1, constitutive_law1, retention_parameters1)
    material_soil2 = SoilMaterial("soil2", soil_formulation1, constitutive_law1, retention_parameters1)
    material_embankment = SoilMaterial("embankment", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:2m x y:2m x z:10m
    soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
    model.extrusion_length = 50

    # Create the soil layer
    model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil1, "soil1")
    model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil2, "soil2")
    model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment")

    # Define UVEC load
    load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]

    uvec_parameters = {
        "n_carts": 1,
        "cart_inertia": (1128.8e3) / 2,
        "cart_mass": (50e3) / 2,
        "cart_stiffness": 2708e3,
        "cart_damping": 64e3,
        "bogie_distances": [-9.95, 9.95],
        "bogie_inertia": (0.31e3) / 2,
        "bogie_mass": (6e3) / 2,
        "wheel_distances": [-1.25, 1.25],
        "wheel_mass": 1.5e3,
        "wheel_stiffness": 4800e3,
        "wheel_damping": 0.25e3,
        "gravity_axis": 1,
        "contact_coefficient": 9.1e-5,
        "contact_power": 1.5,
        "static_initialisation": False,
    }

    uvec_load = UvecLoad(
        direction=[1, 1, 1],
        velocity=1000,
        origin=[0.75, 3, 5],
        wheel_configuration=[0.0, 2.5, 19.9, 22.4],
        uvec_parameters=uvec_parameters,
        uvec_model=uvec,
    )

    model.add_load_by_coordinates(load_coordinates, uvec_load, "train_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                                 roller_displacement_parameters, "sides_roller")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.00999,
                                       delta_time=0.00005,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-3,
                                                            displacement_absolute_tolerance=1.0e-9)
    stress_initialisation_type = StressInitialisationType.NONE

    linear_solver = Amgcl(krylov_type="gmres", tolerance=1e-6)
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     linear_solver_settings=linear_solver,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.001,
                                     rayleigh_m=0.01)

    # Set up problem data
    problem = Problem(problem_name="uvec_3d", number_of_threads=8, settings=solver_settings)
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
                                                                      output_interval=50,
                                                                      nodal_results=nodal_results,
                                                                      gauss_point_results=gauss_point_results,
                                                                      output_control_type="step"))

    model.output_settings = [vtk_output_process]

    input_folder = r"benchmark_tests/test_train_uvec_3d/input_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_train_uvec_3d/output_windows/output_vtk_porous_computational_model_part"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_train_uvec_3d/output_linux/output_vtk_porous_computational_model_part"
    else:
        raise Exception("Unknown platform")

    # test output
    assert assert_files_equal(expected_output_dir,
                              os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    rmtree(input_folder)
