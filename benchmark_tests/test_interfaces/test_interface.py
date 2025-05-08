import os
from stem.model import Model
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.boundary import DisplacementConstraint
from stem.load import PointLoad, MovingLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    LinearNewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem, Cg
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw, Interface, OnePhaseSoilInterface
from stem.stem import Stem



def test_interface():

    ndim = 2
    model = Model(ndim)
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
    layer1_coordinates = [(1.0, 0.0, 0), (9.0, 0.0, 0), (9.0, 5.0, 0), (1.0, 5.0, 0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_block")

    # add another material on top of the first one
    layer2_coordinates = [(3.0, 5.0, 0), (7.0, 5.0, 0), (7.0, 7.0, 0), (3.0, 7.0, 0)]
    model.add_soil_layer_by_coordinates(layer2_coordinates, material1, "soil_block_2")

    constitutive_law2 = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=POISSON_RATIO)
    variables = OnePhaseSoilInterface(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY, MINIMUM_JOINT_WIDTH=0.001)
    interface_material = Interface(name="interface", constitutive_law=constitutive_law2, soil_formulation=variables,retention_parameters=retention_parameters1)

    model.set_interface_between_model_parts(["soil_block"], ["soil_block_2"], interface_material)


    # Boundary conditions and Loads
    load_parameters = PointLoad(value=[100, 0, 0], active=[True, True, True])
    model.add_load_by_coordinates([[3.0, 6.0, 0]], load_parameters, "load")

    # show the model
    #model.show_geometry(show_line_ids=True)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(1, [9], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [13, 10], sym_parameters, "side_rollers")

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.5,
                                       delta_time=0.01,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-6,
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
                                     linear_solver_settings=Cg(),
                                     rayleigh_k=0.01,
                                     rayleigh_m=0.0001)

    # Set up problem data
    problem = Problem(problem_name="test_interface", number_of_threads=4, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [
        NodalOutput.DISPLACEMENT,
        NodalOutput.VELOCITY_X,
        NodalOutput.VELOCITY_Y,
        NodalOutput.VELOCITY_Z,
    ]
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
    model.output_settings.append(vtk_output_process)
    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.1)
    input_folder = "benchmark_tests/test_interface/inputs_kratos"
    stem = Stem(model, input_folder)
    stem.write_all_input_files()
    stem.run_calculation()