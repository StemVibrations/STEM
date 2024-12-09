import os
import sys
from shutil import rmtree

import pytest

from benchmark_tests.utils import assert_files_equal
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.model import Model
from stem.output import NodalOutput, VtkOutputParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy
from stem.stem import Stem


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)
    # add groups for extrusions
    model.add_group_for_extrusion("Group 1", reference_depth=0, extrusion_length=2)
    model.add_group_for_extrusion("Group 2", reference_depth=2, extrusion_length=1)
    model.add_group_for_extrusion("Group 3", reference_depth=3, extrusion_length=2)

    # Specify material model
    solid_density = 2650
    porosity = 0.3
    young_modulus = 30e6
    poisson_ratio = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density, POROSITY=porosity)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus, POISSON_RATIO=poisson_ratio)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material_soil = SoilMaterial("soil1", soil_formulation1, constitutive_law1, retention_parameters1)

    # material for the bridging section
    bridge_density = 7450
    bridge_porosity = 0.0
    bridge_young_modulus = 2.1e8
    bridge_poisson_ratio = 0.3
    bridge_soil_formulation = OnePhaseSoil(ndim,
                                           IS_DRAINED=True,
                                           DENSITY_SOLID=bridge_density,
                                           POROSITY=bridge_porosity)
    bridge_constitutive_law = LinearElasticSoil(YOUNG_MODULUS=bridge_young_modulus, POISSON_RATIO=bridge_poisson_ratio)
    bridge_retention_parameters = SaturatedBelowPhreaticLevelLaw()
    material_bridge = SoilMaterial("steel", bridge_soil_formulation, bridge_constitutive_law,
                                   bridge_retention_parameters)

    # Specify the coordinates for the shapes to extrude: x, y, z [m]

    embankment_coordinates_1 = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.5, 1.0, 0.0), (0, 1.0, 0.0)]
    bridge_coordinates = [(0.0, 0.8, 2.0), (1.0, 0.8, 2.0), (1.0, 1.0, 2.0), (0., 1.0, 2.0)]
    embankment_coordinates_2 = [(0.0, 0.0, 3.0), (3.0, 0.0, 3.0), (1.5, 1.0, 3.0), (0, 1.0, 3.0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(embankment_coordinates_1, material_soil, "embankment1", "Group 1")
    model.add_soil_layer_by_coordinates(bridge_coordinates, material_bridge, "bridge", "Group 2")
    model.add_soil_layer_by_coordinates(embankment_coordinates_2, material_soil, "embankment2", "Group 3")
    # model.show_geometry(show_surface_ids=True)

    # Define moving load
    load_coordinates = [(0.5, 1.0, 0.0), (0.5, 1.0, 5.0)]
    moving_load = MovingLoad(load=[0.0, -10.0, 0.0],
                             direction=[1, 1, 1],
                             velocity=2.5,
                             origin=[0.5, 1.0, 0.0],
                             offset=0.0)
    model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    model.synchronise_geometry()
    # model.show_geometry(show_surface_ids=True)

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [13, 19], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [11, 14, 16, 17, 22, 24], roller_displacement_parameters,
                                                 "sides_roller")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=2.0,
                                       delta_time=0.01,
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
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy,
                                     rayleigh_k=0.0,
                                     rayleigh_m=0.0)

    # Set up problem data
    problem = Problem(problem_name="calculate_moving_load_on_3_groups_3d",
                      number_of_threads=1,
                      settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.TOTAL_DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=10,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=gauss_point_results,
                                                                    output_control_type="step"),
                              part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output")

    input_folder = "benchmark_tests/test_moving_load_on_3_groups_3d/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_moving_load_on_3_groups_3d/output_windows/output_vtk_porous_computational_model_part"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_moving_load_on_3_groups_3d/output_linux/output_vtk_porous_computational_model_part"
    else:
        raise Exception("Unknown platform")

    result = assert_files_equal(expected_output_dir,
                                os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    assert result is True
    rmtree(input_folder)
