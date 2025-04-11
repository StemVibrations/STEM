import os

from benchmark_tests.test_mass_on_spring_damper.test_mass_on_spring_damper import SHOW_RESULTS
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import GravityLoad
from stem.table import Table
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy, Amgcl, LinearNewtonRaphsonStrategy
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem
from stem.utils import Utils
from tests.utils import TestUtils

from shutil import rmtree

SHOW_RESULTS = False


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

    # create block which is loaded with gravity loading
    DENSITY_SOLID = 100
    POROSITY = 0.0
    YOUNG_MODULUS = 50e6
    POISSON_RATIO = 0.3
    soil_formulation2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law2 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters2 = SaturatedBelowPhreaticLevelLaw()
    gravity_block = SoilMaterial("block", soil_formulation2, constitutive_law2, retention_parameters2)

    # Specify the coordinates for the block: x:1m x y:10m
    block_coordinates = [(0, 10, 0), (1, 10, 0), (1, 11, 0), (0, 11, 0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(block_coordinates, gravity_block, "gravity_block")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [2, 4, 5, 7], sym_parameters, "side_rollers")

    # Synchronize geometry
    model.synchronise_geometry()

    # add gravity loading modelpart manually, since gravity should only be there in the block modelpart in this test
    # set gravity load at vertical axis
    gravity_load = GravityLoad(value=[0, -10, 0], active=[True, True, True])

    gravity_block_geometry_ids = model.gmsh_io.geo_data["physical_groups"]["gravity_block"]["geometry_ids"]

    model._Model__add_gravity_model_part(gravity_load, 2, gravity_block_geometry_ids)

    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.15)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.0015
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.15,
                                       delta_time=delta_time,
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
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     linear_solver_settings=Amgcl(tolerance=1e-6),
                                     rayleigh_k=6e-6,
                                     rayleigh_m=0.02)

    # Set up problem data
    problem = Problem(problem_name="test_1d_wave_prop_drained_soil_gravity",
                      number_of_threads=2,
                      settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT_Y, NodalOutput.VELOCITY_Y]

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=1,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=[],
                                                                    output_control_type="step"),
                              output_dir="output",
                              output_name="vtk_output")

    model.add_output_settings_by_coordinates([[0, 5, 0], [1, 5, 0]],
                                             JsonOutputParameters(output_interval=delta_time * 0.99,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=[]),
                                             "calculated_output",
                                             output_dir="output")

    # Define the kratos input folder
    input_folder = "benchmark_tests/test_1d_wave_prop_drained_soil_gravity/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    model_stage_2 = stem.create_new_stage(delta_time, 0.15)
    stem.add_calculation_stage(model_stage_2)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    import json
    with open(os.path.join(input_folder, "output/calculated_output.json")) as f:
        calculated_data_stage1 = json.load(f)

    with open(os.path.join(input_folder, "output/calculated_output_stage_2.json")) as f:
        calculated_data_stage2 = json.load(f)

    if SHOW_RESULTS:
        import matplotlib.pyplot as plt

        plt.plot(calculated_data_stage1["TIME"], calculated_data_stage1["NODE_7"]["VELOCITY_Y"])
        plt.plot(calculated_data_stage2["TIME"], calculated_data_stage2["NODE_7"]["VELOCITY_Y"])
        plt.show()

    calculated_results = Utils.merge(calculated_data_stage1, calculated_data_stage2)

    # open expected results
    with open("benchmark_tests/test_1d_wave_prop_drained_soil_gravity/output_/expected_output.json") as f:
        expected_results = json.load(f)

    # Assert dictionaries
    TestUtils.assert_dictionary_almost_equal(expected_results, calculated_results)

    rmtree(input_folder)
