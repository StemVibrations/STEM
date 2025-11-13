import os
import sys
from shutil import rmtree
import numpy as np

from stem.model import Model
from stem.structural_material import EulerBeam, ElasticSpringDamper
from stem.boundary import DisplacementConstraint
from stem.load import PointLoad
from stem.solver import (
    AnalysisType,
    SolutionType,
    TimeIntegration,
    DisplacementConvergenceCriteria,
    StressInitialisationType,
    SolverSettings,
    Problem,
    NewtonRaphsonStrategy,
)
from stem.output import NodalOutput, Output, VtkOutputParameters
from stem.soil_material import (
    SoilMaterial,
    OnePhaseSoil,
    LinearElasticSoil,
    SaturatedBelowPhreaticLevelLaw,
)
from stem.stem import Stem

from benchmark_tests.utils import assert_floats_in_directories_almost_equal


def test_point_load_on_track_static():
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 20

    # Specify material model
    # Linear elastic drained soil with a Density of 2650, a Young's modulus of 30e6,
    # a Poisson ratio of 0.2 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2650
    POROSITY = 0.3
    YOUNG_MODULUS = 25e6
    POISSON_RATIO = 0.25
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil_1", soil_formulation1, constitutive_law1, retention_parameters1)
    # Specify the coordinates for the column: x:5m x y:1m
    layer1_coordinates = [
        (0.0, 0.0, 0.0),
        (3.47, 0.0, 0.0),
        (3.47, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_layer_1")
    DENSITY_SOLID = 2650
    POROSITY = 0.3
    YOUNG_MODULUS = 13e6
    POISSON_RATIO = 0.3
    soil_formulation2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law2 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters2 = SaturatedBelowPhreaticLevelLaw()
    material2 = SoilMaterial("soil_2", soil_formulation2, constitutive_law2, retention_parameters2)
    # Specify the coordinates for the column: x:5m x y:1m
    layer2_coordinates = [
        (0.0, 1.0, 0),
        (3.47, 1.0, 0),
        (3.47, 2.5, 0),
        (0.0, 2.5, 0),
    ]
    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer2_coordinates, material2, "soil_layer_2")

    # add the track
    rail_parameters = EulerBeam(
        ndim=ndim,
        YOUNG_MODULUS=30e9,
        POISSON_RATIO=0.2,
        DENSITY=7200,
        CROSS_AREA=0.01,
        I33=1e-4,
        I22=1e-4,
        TORSIONAL_INERTIA=2e-4,
    )
    rail_pad_parameters = ElasticSpringDamper(
        NODAL_DISPLACEMENT_STIFFNESS=[1, 750e6, 1],
        NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
        NODAL_DAMPING_COEFFICIENT=[1, 750e3, 1],
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0],
    )

    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2400, POROSITY=0.1)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=30e9, POISSON_RATIO=0.2)
    sleeper_parameters = SoilMaterial(
        name="concrete",
        soil_formulation=soil_formulation,
        constitutive_law=constitutive_law,
        retention_parameters=SaturatedBelowPhreaticLevelLaw(),
    )

    origin_point = np.array([0.75, 2.5, 0])
    direction_vector = np.array([0, 0, 1])
    # dimensions of the sleeper
    sleeper_height = 0.3
    rail_pad_thickness = 0.02
    sleeper_length = 2.5 / 2
    sleeper_width = 0.234
    sleeper_distance = 1.0
    sleeper_dimensions = [sleeper_width, sleeper_height, sleeper_length]
    distance_middle_sleeper_to_rail = origin_point[0]

    # create a straight track with rails, sleepers and rail pads
    model.generate_straight_track(sleeper_distance=sleeper_distance,
                                  n_sleepers=21,
                                  rail_parameters=rail_parameters,
                                  sleeper_parameters=sleeper_parameters,
                                  rail_pad_parameters=rail_pad_parameters,
                                  rail_pad_thickness=rail_pad_thickness,
                                  origin_point=origin_point,
                                  direction_vector=direction_vector,
                                  sleeper_dimensions=sleeper_dimensions,
                                  name="rail_track_1",
                                  distance_middle_sleeper_to_rail=distance_middle_sleeper_to_rail)

    load = PointLoad(active=[False, True, False], value=[0.0, -10000.0, 0.0])
    model.add_load_by_coordinates(
        coordinates=[[0.75, 2.5 + sleeper_height + rail_pad_thickness, 10.0]],
        load_parameters=load,
        name="point_load",
    )

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, False, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    # model.show_geometry(show_surface_ids=True)

    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")

    model.add_boundary_condition_on_plane([[0, 0, 0], [3.47, 0, 0], [0, 1, 0]], roller_displacement_parameters,
                                          "roller")
    model.add_boundary_condition_on_plane([[0, 0, 0], [0, 2.5, 0], [0, 0, model.extrusion_length]],
                                          roller_displacement_parameters, "roller")
    model.add_boundary_condition_on_plane([[3.47, 0, 0], [3.47, 2.5, 0], [3.47, 0, model.extrusion_length]],
                                          roller_displacement_parameters, "roller")
    model.add_boundary_condition_on_plane(
        [[0, 0, model.extrusion_length], [3.47, 0, model.extrusion_length], [0, 2.5, model.extrusion_length]],
        roller_displacement_parameters, "roller")

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(
        start_time=0.0,
        end_time=1.00,
        delta_time=1.00,
        reduction_factor=1.0,
        increase_factor=1.0,
        max_delta_time_factor=1000,
    )
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-9,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    strategy = NewtonRaphsonStrategy()
    solver_settings = SolverSettings(
        analysis_type=analysis_type,
        solution_type=solution_type,
        stress_initialisation_type=stress_initialisation_type,
        time_integration=time_integration,
        is_stiffness_matrix_constant=True,
        are_mass_and_damping_constant=True,
        convergence_criteria=convergence_criterion,
        strategy_type=strategy,
        rayleigh_k=0.0,
        rayleigh_m=0.0,
    )

    # Set up problem data
    problem = Problem(problem_name="test_extended_beam", number_of_threads=4, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [
        NodalOutput.DISPLACEMENT,
    ]
    # Gauss point results
    # gauss_point_results = [GaussPointOutput.CAUCHY_STRESS_VECTOR,GaussPointOutput.CAUCHY_STRESS_TENSOR ]
    gauss_point_results = []

    # Define the output process

    vtk_output_process = Output(
        part_name="porous_computational_model_part",
        output_name="vtk_output",
        output_dir="output",
        output_parameters=VtkOutputParameters(
            file_format="ascii",
            output_interval=1,
            output_control_type="step",
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
        ),
    )
    model.output_settings = [vtk_output_process]
    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=1)
    input_folder = "benchmark_tests/test_volume_sleepers/inputs_kratos_full_qs"
    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()
    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_volume_sleepers/output_windows/output_vtk_porous_computational_model_part"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_volume_sleepers/output_linux/output_vtk_porous_computational_model_part"
    else:
        raise Exception("Unknown platform")

    assert_floats_in_directories_almost_equal(
        expected_output_dir,
        os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"),
    )

    rmtree(input_folder)
