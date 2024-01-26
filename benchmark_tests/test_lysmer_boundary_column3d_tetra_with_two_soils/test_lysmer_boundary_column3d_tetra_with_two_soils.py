import os
import sys

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import SurfaceLoad
from stem.boundary import AbsorbingBoundary
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
    NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, GaussPointOutput, VtkOutputParameters, Output
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)

    # Specify material model
    # Linear elastic drained soil with a Density of 2650, a Young's modulus of 10.0e7,
    # a Poisson ratio of 0.2 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2650
    POISSON_RATIO = 0.3
    YOUNG_MODULUS = 10.0e7
    POROSITY = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil1", soil_formulation1, constitutive_law1, retention_parameters1)
    material2 = SoilMaterial("soil2", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:2m x y:2m x z:10m
    layer1_coordinates = [(-1, -1, 0), (1, -1, 0), (1, 0, 0), (-1, 0, 0)]
    layer2_coordinates = [(-1, 0, 0), (1, 0, 0), (1, 1, 0), (-1, 1, 0)]
    model.extrusion_length = 10

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "layer1")
    model.add_soil_layer_by_coordinates(layer2_coordinates, material2, "layer2")

    # Boundary conditions and Loads
    load_coordinates = [(-1, -1, 10), (1, -1, 10), (1, 1, 10), (-1, 1, 10)]

    # Add line load
    surface_load = SurfaceLoad(active=[False, False, True], value=[0, 0, -10])
    model.add_load_by_coordinates(load_coordinates, surface_load, "load")

    # Define absorbing boundary condition
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

    # Define displacement conditions
    displacement_parameters = DisplacementConstraint(active=[True, True, False], is_fixed=[True, True, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [5,10], absorbing_boundaries_parameters, "abs")
    model.add_boundary_condition_by_geometry_ids(2, [1, 2, 4, 7, 8, 9], displacement_parameters, "sides")


    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.25)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0, end_time=0.3, delta_time=0.01, reduction_factor=1.0,
                                    increase_factor=1.0, max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                    stress_initialisation_type=stress_initialisation_type,
                                    time_integration=time_integration,
                                    is_stiffness_matrix_constant=True,
                                    are_mass_and_damping_constant=True,
                                    convergence_criteria=convergence_criterion,
                                    rayleigh_k=0.001,
                                    rayleigh_m=0.1)

    # Set up problem data
    problem = Problem(problem_name="test_lysmer_boundary_column3d_tetra_with_two_soils", number_of_threads=1, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT,
                    NodalOutput.VELOCITY]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(
        file_format="ascii",
        output_interval=10,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
    ), part_name="porous_computational_model_part", output_dir="output", output_name="vtk_output")

    input_folder = "benchmark_tests/test_lysmer_boundary_column3d_tetra_with_two_soils/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()


    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_lysmer_boundary_column3d_tetra_with_two_soils/output_windows/output_vtk_porous_computational_model_part"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_lysmer_boundary_column3d_tetra_with_two_soils/output_linux/output_vtk_porous_computational_model_part"
    else:
        raise Exception("Unknown platform")

    result = assert_files_equal(expected_output_dir, os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    assert result is True
    rmtree(input_folder)
