import os
from shutil import rmtree, copyfile

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import UvecLoad
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, Amgcl, NewtonRaphsonStrategy)
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.stem import Stem

from benchmark_tests.utils import assert_files_equal


def test_stem():
    """
    This test tests a UVEC with two wheels with different wheel loads on a 2d soil model.

    """

    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    # Specify material model
    # Linear elastic drained soil with a Density of 2650, a Young's modulus of 30e6,
    # a Poisson ratio of 0.2 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2650
    POROSITY = 0.3
    YOUNG_MODULUS = 30e6
    POISSON_RATIO = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:5m x y:1m
    layer1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_layer")

    # Define UVEC load
    load_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0)]

    uvec_parameters = {"load_wheel_1": -30.0, "load_wheel_2": -10.0}
    uvec_load = UvecLoad(direction=[1, 1, 0],
                         velocity=5,
                         origin=[0.0, 1.0, 0.0],
                         wheel_configuration=[0.0, 2.0],
                         uvec_file=r"sample_uvec.py",
                         uvec_function_name="uvec_test",
                         uvec_parameters=uvec_parameters)

    model.add_load_by_coordinates(load_coordinates, uvec_load, "uvec_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, False],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [2, 4], roller_displacement_parameters, "roller_fixed")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=0.1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=1.0,
                                       delta_time=0.01,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
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
                                     strategy_type=NewtonRaphsonStrategy(),
                                     rayleigh_k=0.0001,
                                     rayleigh_m=0.01)

    # Set up problem data
    problem = Problem(problem_name="calculate_uvec_on_soil_2d", number_of_threads=2, settings=solver_settings)
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

    input_folder = "benchmark_tests/test_uvec_on_soil_2d/inputs_kratos"

    # copy uvec to input folder
    os.makedirs(input_folder, exist_ok=True)
    copyfile("benchmark_tests/test_uvec_on_soil_2d/sample_uvec.py", os.path.join(input_folder, "sample_uvec.py"))

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    assert assert_files_equal("benchmark_tests/test_uvec_on_soil_2d/output_/output_vtk_porous_computational_model_part",
                              os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    rmtree(input_folder)
