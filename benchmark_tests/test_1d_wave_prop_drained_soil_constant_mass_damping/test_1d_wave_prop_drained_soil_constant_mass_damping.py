import os
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.table import Table
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

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
    load_coordinates = [(0.0, 10.0, 0), (1.0, 10.0, 0)]
    # Add table for the load in the mdpa file
    t = (0.0, 0.0025, 1)
    values = (0.0, -1000.0, -1000.0)
    LINE_LOAD_Y = Table(times=t, values=values)
    # Add line load
    line_load = LineLoad(active=[False, True, False], value=[0, LINE_LOAD_Y, 0])
    model.add_load_by_coordinates(load_coordinates, line_load, "load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [2, 4], sym_parameters, "side_rollers")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.45)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.15,
                                       delta_time=0.0025,
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
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=6e-6,
                                     rayleigh_m=0.02)

    # Set up problem data
    problem = Problem(problem_name="test_1d_wave_prop_drained_soil_constant_mass_damping",
                      number_of_threads=4,
                      settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.VELOCITY]
    # Gauss point results
    gauss_point_results = []

    # Define the output process and add to the model
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=10,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=gauss_point_results,
                                                                    output_control_type="step"),
                              part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output")

    # Define the kratos input folder
    input_folder = "benchmark_tests/test_1d_wave_prop_drained_soil_constant_mass_damping/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    result = assert_files_equal(
        "benchmark_tests/test_1d_wave_prop_drained_soil_constant_mass_damping/output_/output_vtk_porous_computational_model_part",
        os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    assert result is True
    rmtree(input_folder)
