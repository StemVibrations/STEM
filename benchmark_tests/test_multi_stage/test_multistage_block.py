import os
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.table import Table
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy)
from stem.output import NodalOutput, VtkOutputParameters
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def test_stem():
    """
    Test STEM: 2D block with distributed loading with multistage
    """

    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model_stage_1 = Model(ndim)

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
    layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

    # Create the soil layer
    model_stage_1.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_block")

    # Boundary conditions and Loads
    load_coordinates = [(0.0, 1.0, 0), (1.0, 1.0, 0)]
    t = (0.0, 0.0075, 1)
    values = (0.0, -1000.0, -1000.0)
    LINE_LOAD_Y = Table(times=t, values=values)
    # Add line load
    line_load = LineLoad(active=[False, True, False], value=[0, LINE_LOAD_Y, 0])
    # Add line load
    # line_load = LineLoad(active=[False, True, False], value=[0, -10000, 0])
    model_stage_1.add_load_by_coordinates(load_coordinates, line_load, "load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model_stage_1.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model_stage_1.add_boundary_condition_by_geometry_ids(1, [2, 4], sym_parameters, "side_rollers")

    # Synchronize geometry
    model_stage_1.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model_stage_1.set_mesh_size(element_size=0.1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.QUASI_STATIC

    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.15,
                                       delta_time=0.05,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-12,
                                                            displacement_absolute_tolerance=1.0E-6)
    stress_initialisation_type = StressInitialisationType.NONE
    strategy = NewtonRaphsonStrategy()
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=False,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy,
                                     rayleigh_k=0.0,
                                     rayleigh_m=0.0)

    # Set up problem data
    problem = Problem(problem_name="test_multi_stage_block", number_of_threads=2, settings=solver_settings)
    model_stage_1.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

    # Define the output process
    model_stage_1.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                            output_interval=1,
                                                                            nodal_results=nodal_results,
                                                                            gauss_point_results=[],
                                                                            output_control_type="step"),
                                      output_dir="output",
                                      output_name="vtk_output")

    # define the STEM instance
    input_folder = "benchmark_tests/test_multi_stage/inputs_kratos"
    stem = Stem(model_stage_1, input_folder)

    # create new stage
    model_stage_2 = stem.create_new_stage(0.0025, 0.05)

    # Set up solver settings for the new stage
    model_stage_2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    model_stage_2.project_parameters.settings.rayleigh_k = 1e-6
    model_stage_2.project_parameters.settings.rayleigh_m = 0.02

    model_stage_2.process_model_parts[0].parameters.value = [0, -1050, 0]
    model_stage_2.output_settings[0].output_parameters.nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

    # add the new stage to the calculation
    stem.add_calculation_stage(model_stage_2)

    # write the kratos input files
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    result = assert_files_equal("benchmark_tests/test_multi_stage/output_/output_vtk_full_model",
                                os.path.join(input_folder, "output/output_vtk_full_model"))

    assert result is True
    rmtree(input_folder)
