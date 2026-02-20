import numpy as np
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import MovingLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, NewtonRaphsonStrategy, Cg
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem


def run_moving_load(input_folder):
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)
    x_max = 10
    y_max = 10
    z_max = 20

    t_step = 0.01

    # Specify material model
    # Linear elastic drained soil with a Density of 2650, a Young's modulus of 30e6,
    # a Poisson ratio of 0.2 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2000
    POROSITY = 0
    YOUNG_MODULUS = 30e6
    POISSON_RATIO = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:10m x y:10m
    layer1_coordinates = [(0.0, 0.0, 0.0), (x_max, 0.0, 0.0), (x_max, y_max, 0.0), (0.0, y_max, 0.0)]
    model.extrusion_length = z_max

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_layer")

    # Define moving load
    load_coordinates = [(0.0, y_max, 0.0), (0.0, y_max, z_max)]
    moving_load = MovingLoad(load=[0.0, -1e3, 0.0],
                             direction_signs=[1, 1, 1],
                             velocity=10,
                             origin=[0.0, y_max, 0.0],
                             offset=1.0)
    model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement_parameters_x = DisplacementConstraint(is_fixed=[True, False, False], value=[0, 0, 0])

    abs_boundary_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=10)

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, 0, z_max)], no_displacement_parameters,
                                          "base_fixed")
    model.add_boundary_condition_on_plane([(0, 0, 0), (0, y_max, 0), (0, y_max, z_max)],
                                          roller_displacement_parameters_x, "sides_roler_x=0")
    model.add_boundary_condition_on_plane([(x_max, 0, 0), (x_max, y_max, 0), (x_max, y_max, z_max)],
                                          abs_boundary_parameters, "sides_roler_x=x_max")

    model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, y_max, 0)], abs_boundary_parameters,
                                          "abs_z=0")
    model.add_boundary_condition_on_plane([(0, 0, z_max), (x_max, 0, z_max), (x_max, y_max, z_max)],
                                          abs_boundary_parameters, "abs_z=z_max")

    # Synchronize geometry
    model.synchronise_geometry()
    # Set mesh size
    model.set_mesh_size(element_size=0.5)
    model.mesh_settings.element_order = 2

    # Define project parameters
    # --------------------------------
    # Set up solver settings
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=5 * t_step,
                                       delta_time=t_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.QUASI_STATIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=NewtonRaphsonStrategy(),
                                     linear_solver_settings=Cg(),
                                     rayleigh_k=0,
                                     rayleigh_m=0)

    # Set up problem data
    problem = Problem(problem_name="calculate_moving_load_on_soil_3d", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    # Define the output process
    nodes = [[0, y_max - 1, n] for n in np.linspace(0, z_max, num=21)]
    model.add_output_settings_by_coordinates(nodes,
                                             JsonOutputParameters(output_interval=t_step,
                                                                  nodal_results=[NodalOutput.DISPLACEMENT],
                                                                  gauss_point_results=[]),
                                             "calculated_output",
                                             output_dir="output")

    model.add_output_settings(output_parameters=VtkOutputParameters(
        file_format="binary",
        output_interval=1,
        nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY],
        gauss_point_results=[],
        output_control_type="step"),
                              part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output")

    # stage 2
    # --------------------------------
    stem = Stem(model, input_folder)

    stage2 = stem.create_new_stage(t_step, 1.5)
    stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    stage2.project_parameters.settings.strategy_type = LinearNewtonRaphsonStrategy()
    stage2.project_parameters.settings.rayleigh_k = 3.92975e-5
    stage2.project_parameters.settings.rayleigh_m = 0.124
    stem.add_calculation_stage(stage2)
    stem.write_all_input_files()
    stem.run_calculation()
