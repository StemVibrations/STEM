import os
import sys
from shutil import rmtree
import json

import numpy as np
import pytest

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal


def test_moving_load_on_extended_track():
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 10

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
    layer1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 3.0, 0.0), (0.0, 3.0, 0.0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_layer")

    # add the track
    rail_parameters = EulerBeam(ndim=ndim,
                                YOUNG_MODULUS=30e9,
                                POISSON_RATIO=0.2,
                                DENSITY=7200,
                                CROSS_AREA=0.01,
                                I33=1e-4,
                                I22=1e-4,
                                TORSIONAL_INERTIA=2e-4)
    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 750e6, 1],
                                              NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                              NODAL_DAMPING_COEFFICIENT=[1, 750e3, 1],
                                              NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
    soil_equivalent_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 3e6, 1],
                                                     NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                                     NODAL_DAMPING_COEFFICIENT=[1, 3e3, 1],
                                                     NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    origin_point = np.array([1.0, 3.0, -5.0])
    direction_vector = np.array([0, 0, 1])
    rail_pad_thickness = 0.025

    # create a straight track with rails, sleepers and rail pads
    model.generate_extended_straight_track(0.5, 40, rail_parameters, sleeper_parameters, rail_pad_parameters,
                                           rail_pad_thickness, origin_point, soil_equivalent_parameters, 5,
                                           direction_vector, "rail_track_1")
    origin = [float(origin_point[0]), float(origin_point[1] + rail_pad_thickness), float(origin_point[-1])]
    moving_load = MovingLoad(load=[0.0, -10000.0, 0.0], direction=[1, 1, 1], velocity=10, origin=origin, offset=0.0)

    model.add_load_on_line_model_part("rail_track_1", moving_load, "moving_load")

    # model.show_geometry(show_surface_ids=True, show_point_ids=True)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [2], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [1, 3, 5, 6], roller_displacement_parameters, "roller_fixed")

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=2.0,
                                       delta_time=0.01,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     rayleigh_k=0.01,
                                     rayleigh_m=0.0001)

    # Set up problem data
    problem = Problem(problem_name="test_extended_beam", number_of_threads=4, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [
        NodalOutput.DISPLACEMENT, NodalOutput.TOTAL_DISPLACEMENT, NodalOutput.DISPLACEMENT_X,
        NodalOutput.DISPLACEMENT_Y, NodalOutput.DISPLACEMENT_Z
    ]
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
    # define json output parameters so that we can test that the fixities work
    json_output_parameters = JsonOutputParameters(output_interval=0.5,
                                                  nodal_results=nodal_results,
                                                  gauss_point_results=gauss_point_results)
    model.add_output_settings(json_output_parameters, f"soil_equivalent_rail_track_1", "output")
    model.add_output_settings(json_output_parameters, f"constraint_soil_equivalent_rail_track_1", "output")

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=1)

    input_folder = "benchmark_tests/test_extended_beam/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    expected_output_dir_temp = "benchmark_tests/test_extended_beam/inputs_kratos/output"
    # check that the output is as expected (only the nodal displacements are checked)
    with open(os.path.join(expected_output_dir_temp, "constraint_soil_equivalent_rail_track_1.json"), "r") as f:
        constrain_output = json.load(f)
    nodes = [item for name, item in constrain_output.items() if 'TIME' not in name]
    assert np.all(np.array([node['DISPLACEMENT_X'] for node in nodes]) == 0)
    assert np.all(np.array([node['DISPLACEMENT_Y'] for node in nodes]) == 0)
    assert np.all(np.array([node['DISPLACEMENT_Z'] for node in nodes]) == 0)
    with open(os.path.join(expected_output_dir_temp, "soil_equivalent_rail_track_1.json"), "r") as f:
        soil_equivalent_output = json.load(f)
    nodes = [item for name, item in soil_equivalent_output.items() if 'TIME' not in name]
    assert np.all(np.array([node['DISPLACEMENT_X'] for node in nodes]) == 0)
    assert np.any(np.array([node['DISPLACEMENT_Y'] for node in nodes]) != 0)
    assert np.all(np.array([node['DISPLACEMENT_Z'] for node in nodes]) == 0)

    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_extended_beam/output_windows/output_vtk_porous_computational_model_part"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_extended_beam/output_linux/output_vtk_porous_computational_model_part"
    else:
        raise Exception("Unknown platform")

    result = assert_files_equal(expected_output_dir,
                                os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    assert result is True
    rmtree(input_folder)
