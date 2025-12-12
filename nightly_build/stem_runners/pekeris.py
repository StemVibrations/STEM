from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import PointLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, NewtonRaphsonStrategy, Cg
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem


def run_pekeris(input_folder):
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)

    # Specify material model
    DENSITY_SOLID = 2000
    POROSITY = 0
    YOUNG_MODULUS = 30e6
    POISSON_RATIO = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the 3D block: x:10m x y:10m z:10m
    x_max = 5
    y_max = 5
    z_max = 5
    force = -1e6
    layer1_coordinates = [(0.0, 0.0, 0.0), (x_max, 0.0, 0.0), (x_max, y_max, 0.0), (0.0, y_max, 0.0)]
    model.extrusion_length = z_max

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil")

    # Define load
    node_coordinates = [(0.0, y_max, 0.0)]
    load = PointLoad(active=[True, True, True], value=[0, force, 0])
    model.add_load_by_coordinates(node_coordinates, load, "point_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters_x = DisplacementConstraint(active=[True, True, True],
                                                              is_fixed=[True, False, False],
                                                              value=[0, 0, 0])
    roller_displacement_parameters_z = DisplacementConstraint(active=[True, True, True],
                                                              is_fixed=[False, False, True],
                                                              value=[0, 0, 0])

    # abs_boundary_parameters = DisplacementConstraint(active=[True, True, True],
    #                                                     is_fixed=[True, True, True],
    #                                                     value=[0, 0, 0])
    abs_boundary_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=10)

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, 0, z_max)], no_displacement_parameters,
                                          "base_fixed")
    model.add_boundary_condition_on_plane([(0, 0, 0), (0, y_max, 0), (0, y_max, z_max)],
                                          roller_displacement_parameters_x, "sides_roler_x=0")
    model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, y_max, 0)],
                                          roller_displacement_parameters_z, "sides_roler_z=0")
    model.add_boundary_condition_on_plane([(x_max, 0, 0), (x_max, y_max, 0), (x_max, y_max, z_max)],
                                          abs_boundary_parameters, "abs_x=x_max")
    model.add_boundary_condition_on_plane([(0, 0, z_max), (x_max, 0, z_max), (x_max, y_max, z_max)],
                                          abs_boundary_parameters, "abs_z=z_max")

    model.set_mesh_size(element_size=0.2)
    model.mesh_settings.element_order = 2

    # Synchronize geometry
    model.synchronise_geometry()

    # Define project parameters
    # --------------------------------
    # Set up solver settings
    time_step = 0.0005
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.08,
                                       delta_time=time_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)

    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.DYNAMIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     linear_solver_settings=Cg(),
                                     rayleigh_k=1.9648758406406834e-05,
                                     rayleigh_m=0.062056151182020604)

    # Set up problem data
    problem = Problem(problem_name="Pekeris", number_of_threads=44, settings=solver_settings)
    model.project_parameters = problem

    # Result output settings
    json_output_parameters = JsonOutputParameters(time_step, [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY], [])
    model.add_output_settings_by_coordinates([
        (0, y_max, 0),
        (1, y_max, 0),
        (2, y_max, 0),
        (3, y_max, 0),
    ], json_output_parameters, "json_output")

    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=100,
                                                                    nodal_results=[NodalOutput.VELOCITY],
                                                                    gauss_point_results=[],
                                                                    output_control_type="step"),
                              part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output")

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()
