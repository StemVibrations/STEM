from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy, Lu
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters, GaussPointOutput
from stem.stem import Stem


def run_strip_2D(input_folder):

    # Specify dimension and initiate the model
    ndim = 2
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

    # Specify the coordinates for the 2D block: x:20m x y:10m
    layer1_coordinates = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 10.0, 0.0), (0.0, 10.0, 0.0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil")

    # Define load
    # Add line load
    line_load = LineLoad(active=[False, True, False], value=[0, -1e6, 0])
    model.add_load_by_coordinates([(0, 10.0, 0), (1, 10.0, 0)], line_load, "load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters_x = DisplacementConstraint(active=[True, True, True],
                                                              is_fixed=[True, False, False],
                                                              value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [4, 2], roller_displacement_parameters_x, "sides_roler")

    # model.show_geometry(show_line_ids=True)
    model.set_mesh_size(element_size=0.15)
    model.mesh_settings.element_order = 2

    # Synchronize geometry
    model.synchronise_geometry()

    # Define project parameters
    # --------------------------------
    # Set up solver settings
    time_step = 0.001
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.1,
                                       delta_time=time_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-6,
                                                            displacement_absolute_tolerance=1.0e-12)

    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.DYNAMIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=NewtonRaphsonStrategy(),
                                     linear_solver_settings=Lu(),
                                     rayleigh_k=7.86e-5,
                                     rayleigh_m=0.248)

    # Set up problem data
    problem = Problem(problem_name="strip", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    # Result output settings
    json_output_parameters = JsonOutputParameters(0.025, [NodalOutput.VELOCITY, NodalOutput.CAUCHY_STRESS_VECTOR], [])

    nodes = [(i * 0.3, 10, 0) for i in range(67)]
    model.add_output_settings_by_coordinates(nodes, json_output_parameters, "json_output")

    model.add_output_settings(output_parameters=VtkOutputParameters(
        file_format="ascii",
        output_interval=50,
        nodal_results=[NodalOutput.VELOCITY],
        gauss_point_results=[GaussPointOutput.CAUCHY_STRESS_VECTOR],
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
