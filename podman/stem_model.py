import numpy as np

from stem.model import Model
from stem.default_materials import DefaultMaterial
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.load import MovingLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, NewtonRaphsonStrategy, Cg
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem


def run_analysis(speed, output_dir, vtk):

    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 36
    time_step = 0.005
    total_time = model.extrusion_length / (speed / 3.6)
    # round total_time to multiple of time_step
    total_time = 10 * time_step

    # Specify material model
    DENSITY_SOLID = 2050.164566112163
    POROSITY = 0.
    YOUNG_MODULUS = 437122125.0010577
    POISSON_RATIO = 0.304875
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil_1", soil_formulation1, constitutive_law1, retention_parameters1)

    DENSITY_SOLID = 1448.2094862311253
    POROSITY = 0.
    YOUNG_MODULUS = 43565548.270884626
    POISSON_RATIO = 0.4316
    soil_formulation2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law2 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters2 = SaturatedBelowPhreaticLevelLaw()
    material2 = SoilMaterial("soil_2", soil_formulation2, constitutive_law2, retention_parameters2)
    # max ele size = 0.6

    DENSITY_SOLID = 1086.7079215720141
    POROSITY = 0.
    YOUNG_MODULUS = 18276028.839450527
    POISSON_RATIO = 0.495
    soil_formulation3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law3 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters3 = SaturatedBelowPhreaticLevelLaw()
    material3 = SoilMaterial("soil_3", soil_formulation3, constitutive_law3, retention_parameters3)
    # max ele size = 1.5

    DENSITY_SOLID = 1819.432703775135
    POROSITY = 0.
    YOUNG_MODULUS = 148502682.37995887
    POISSON_RATIO = 0.495
    soil_formulation4 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law4 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters4 = SaturatedBelowPhreaticLevelLaw()
    material4 = SoilMaterial("soil_4", soil_formulation4, constitutive_law4, retention_parameters4)
    # max ele size = 3.5

    DENSITY_SOLID = 1850
    POROSITY = 0.
    YOUNG_MODULUS = 100e6
    POISSON_RATIO = 0.3
    soil_formulation5 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law5 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters5 = SaturatedBelowPhreaticLevelLaw()
    ballast = SoilMaterial("ballast", soil_formulation5, constitutive_law5, retention_parameters5)

    DENSITY_SOLID = 2000
    POROSITY = 0.
    YOUNG_MODULUS = 150e6
    POISSON_RATIO = 0.3
    soil_formulation6 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law6 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters6 = SaturatedBelowPhreaticLevelLaw()
    embankment = SoilMaterial("embankment", soil_formulation6, constitutive_law6, retention_parameters6)

    node_1 = (0.0, 5.0, 0.0)
    node_2 = (15.0, 5.0, 0.0)

    node_3 = (0.0, 10.0, 0.0)
    node_4 = (15.0, 10.0, 0.0)

    node_5 = (0.0, 10.5, 0.0)
    node_6 = (15.0, 10.5, 0.0)

    node_7 = (0.0, 11.4, 0.0)
    node_8 = (15.0, 11.4, 0.0)

    node_9 = (0.0, 11.8, 0.0)
    node_10 = (15.0, 11.8, 0.0)
    node_11 = (6.0, 11.8, 0.0)

    node_12 = (0.0, 14.4, 0.0)
    node_13 = (4.5, 14.4, 0.0)
    node_14 = (3.0, 14.4, 0.0)

    node_15 = (0, 14.8, 0.0)
    node_16 = (2.5, 14.8, 0.0)

    x_min = 0
    x_max = 15
    y_min = 5
    y_max = 14.8
    z_min = 0
    z_max = model.extrusion_length

    # Specify the coordinates for the column: x:5m x y:1m
    layer1_coordinates = [node_1, node_2, node_4, node_3]
    layer2_coordinates = [node_3, node_4, node_6, node_5]
    layer3_coordinates = [node_5, node_6, node_8, node_7]
    layer4_coordinates = [node_7, node_8, node_10, node_11, node_9]
    layer5_coordinates = [node_9, node_11, node_13, node_14, node_12]
    layer6_coordinates = [node_12, node_14, node_16, node_15]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material4, "soil_4")
    model.add_soil_layer_by_coordinates(layer2_coordinates, material3, "soil_3")
    model.add_soil_layer_by_coordinates(layer3_coordinates, material2, "soil_2")
    model.add_soil_layer_by_coordinates(layer4_coordinates, material1, "soil_1")
    model.add_soil_layer_by_coordinates(layer5_coordinates, embankment, "embankment")
    model.add_soil_layer_by_coordinates(layer6_coordinates, ballast, "ballast")

    # add the track
    rail_parameters = DefaultMaterial.Rail_60E1_3D.value.material_parameters
    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 6e8, 1],
                                              NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                              NODAL_DAMPING_COEFFICIENT=[1, 2.5e5, 1],
                                              NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=477 / 2,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    origin_point = np.array([0.75, 14.8, 0.0])
    direction_vector = np.array([0, 0, 1])
    rail_pad_thickness = 0.025

    # create a straight track with rails, sleepers and rail pads
    model.generate_straight_track(0.6, 61, rail_parameters, sleeper_parameters, rail_pad_parameters, rail_pad_thickness,
                                  origin_point, direction_vector, "rail_track_1")

    moving_load = MovingLoad(load=[0.0, -10000.0, 0.0],
                             direction=[1, 1, 1],
                             velocity=speed / 3.6,
                             origin=[0.75, 14.8 + rail_pad_thickness, 0.0],
                             offset=0.0)

    model.add_load_on_line_model_part("rail_track_1", moving_load, "moving_load")

    model.show_geometry()

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, False, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=0.1)

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_on_plane([(x_min, y_min, z_min), (x_max, y_min, z_min), (x_min, y_min, z_max)],
                                          no_displacement_parameters, "bottom")
    model.add_boundary_condition_on_plane([(x_min, y_min, z_min), (x_max, y_min, z_min), (x_min, y_max, z_min)],
                                          absorbing_boundaries_parameters, "abs")
    model.add_boundary_condition_on_plane([(x_min, y_min, z_max), (x_max, y_min, z_max), (x_min, y_max, z_max)],
                                          absorbing_boundaries_parameters, "abs")
    model.add_boundary_condition_on_plane([(x_min, y_min, z_min), (x_min, y_max, z_min), (x_min, y_min, z_max)],
                                          roller_displacement_parameters, "left")
    model.add_boundary_condition_on_plane([(x_max, y_min, z_min), (x_max, y_max, z_min), (x_max, y_min, z_max)],
                                          absorbing_boundaries_parameters, "abs")

    model.set_element_size_of_group(0.25, "embankment")
    model.set_element_size_of_group(0.25, "ballast")
    model.set_element_size_of_group(0.5, "soil_1")
    model.set_element_size_of_group(0.5, "soil_2")
    model.set_element_size_of_group(0.5, "soil_3")
    model.set_element_size_of_group(0.5, "soil_4")
    model.set_mesh_size(element_size=0.5)
    model.mesh_settings.element_order = 2

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=5 * time_step,
                                       delta_time=time_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)

    stress_initialisation_type = StressInitialisationType.NONE
    strategy = NewtonRaphsonStrategy()

    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     linear_solver_settings=Cg(scaling=False),
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy)

    # Set up problem data
    problem = Problem(problem_name="test_moving_load_on_track_on_soil", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process

    vtk_output_process = Output(part_name="porous_computational_model_part",
                                output_name="vtk_output",
                                output_dir="output",
                                output_parameters=VtkOutputParameters(file_format="binary",
                                                                      output_interval=1,
                                                                      nodal_results=nodal_results,
                                                                      gauss_point_results=gauss_point_results,
                                                                      output_control_type="step"))

    if vtk:
        model.output_settings = [vtk_output_process]
    else:
        model.output_settings = []

    desired_output_points = [
        (0.75, 14.8, 30),
        (2.5, 14.8, 30),
    ]

    model.add_output_settings_by_coordinates(part_name="subset_outputs",
                                             output_dir=output_dir,
                                             output_name="json_output",
                                             coordinates=desired_output_points,
                                             output_parameters=JsonOutputParameters(
                                                 output_interval=time_step,
                                                 nodal_results=nodal_results,
                                                 gauss_point_results=gauss_point_results))

    # create STEM object
    # --------------------------------
    stem = Stem(model, output_dir)

    # define stage 2
    # --------------------------------
    duration_stage_2 = total_time
    stage2 = stem.create_new_stage(time_step, duration_stage_2)
    stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    stage2.project_parameters.settings.linear_solver_settings = Cg(scaling=False)
    stage2.project_parameters.settings.is_stiffness_matrix_constant = True
    stage2.project_parameters.settings.are_mass_and_damping_constant = True
    stage2.project_parameters.settings.strategy_type = LinearNewtonRaphsonStrategy()

    # add rayleigh damping parameters
    stage2.project_parameters.settings.rayleigh_k = 7.86e-5
    stage2.project_parameters.settings.rayleigh_m = 0.248

    # increase the virtual thickness of the side absorbing boundary conditions for proper damping
    stage2.get_model_part_by_name("abs").parameters.virtual_thickness = 50

    # add the new stage to the stem calculation
    stem.add_calculation_stage(stage2)

    # Write KRATOS input files
    # --------------------------------
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()


if __name__ == "__main__":
    run_analysis(speed=140, output_dir=f"sbb/stem_calcs_2/results_{140}", vtk=True)
