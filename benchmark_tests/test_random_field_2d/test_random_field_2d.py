import os
from shutil import rmtree

from stem.additional_processes import ParameterFieldParameters
from stem.field_generator import RandomFieldGenerator
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy)
from stem.output import VtkOutputParameters, GaussPointOutput
from stem.stem import Stem

from benchmark_tests.utils import assert_files_equal


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=10, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="soil",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    width = 25
    height = 12
    # add soil layers
    model.add_soil_layer_by_coordinates([(0, 0, 0), (width, 0, 0), (width, height, 0), (0, height, 0)], soil_material,
                                        "layer1")

    # Define the field generator
    random_field_generator = RandomFieldGenerator(cov=0.1,
                                                  v_scale_fluctuation=1,
                                                  anisotropy=[10.0],
                                                  angle=[60],
                                                  model_name="Gaussian",
                                                  seed=14)

    field_parameters_json = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                     function_type="json_file",
                                                     field_generator=random_field_generator)

    model.add_field(part_name="layer1", field_parameters=field_parameters_json)

    model.synchronise_geometry()

    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [2, 4], roller_displacement_parameters, "sides_roller")

    # set mesh size
    model.set_mesh_size(element_size=1)

    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=1.0,
                                       delta_time=1,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)
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
    problem = Problem(problem_name="create_random_field_2d", number_of_threads=1, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Gauss point results
    gauss_point_results = [GaussPointOutput.YOUNG_MODULUS]

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=1,
                                                                    nodal_results=[],
                                                                    gauss_point_results=gauss_point_results,
                                                                    output_control_type="step"),
                              part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output")

    # Write KRATOS input files
    # --------------------------------

    input_folder = "benchmark_tests/test_random_field_2d/inputs_kratos"

    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    result = assert_files_equal(
        "benchmark_tests/test_random_field_2d/output_/output_vtk_porous_computational_model_part",
        os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    assert result is True
    rmtree(input_folder)
