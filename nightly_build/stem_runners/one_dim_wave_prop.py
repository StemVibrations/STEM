from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import SurfaceLoad, LineLoad
from stem.table import Table
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg)
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem


def run_column(input_folder, ndim):
    # dimension
    column_height = 10
    column_width = 0.25

    # Specify dimension and initiate the model
    model = Model(ndim)
    model.extrusion_length = column_width

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
    layer1_coordinates = [(0, 0, 0), (column_width, 0, 0), (column_width, column_height, 0), (0, column_height, 0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_column")

    # Add table for the load in the mdpa file
    t = (0.0, 0.0025, 1)
    values = (0.0, -1000.0, -1000.0)
    LOAD_Y = Table(times=t, values=values)

    # add load and boundary conditions based on dimension
    if ndim == 2:
        load_coordinates = [(0.0, column_height, 0), (column_width, column_height, 0)]

        # Add line load
        surface_load = LineLoad(active=[False, True, False], value=[0, LOAD_Y, 0])

        geometry_ids_base = [1]
        geometry_ids_sides = [2, 4]
    elif ndim == 3:
        load_coordinates = [(0.0, column_height, 0), (column_width, column_height, 0),
                            (column_width, column_height, column_width), (0.0, column_height, column_width)]
        # Add surface load
        surface_load = SurfaceLoad(active=[False, True, False], value=[0, LOAD_Y, 0])

        geometry_ids_base = [2]
        geometry_ids_sides = [1, 3, 6, 5]
    else:
        raise ValueError("ndim must be 2 or 3")

    model.add_load_by_coordinates(load_coordinates, surface_load, "load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, True], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(ndim - 1, geometry_ids_base, no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(ndim - 1, geometry_ids_sides, sym_parameters, "side_rollers")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.10)
    model.mesh_settings.element_order = 2

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.0025
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.5,
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
                                     linear_solver_settings=Cg(),
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     rayleigh_k=3.929751681281367e-05,
                                     rayleigh_m=0.12411230236404121)

    # Set up problem data
    problem = Problem(problem_name="test_1d_wave_prop_drained_soil", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.VELOCITY, NodalOutput.DISPLACEMENT]

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=1,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=[],
                                                                    output_control_type="step"),
                              output_dir="output",
                              output_name="vtk_output")

    model.add_output_settings_by_coordinates([[0, 2.5, 0], [0, 5, 0], [0, 7.5, 0]],
                                             JsonOutputParameters(output_interval=delta_time,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=[]),
                                             f"calculated_output_{ndim}",
                                             output_dir="output")

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()
