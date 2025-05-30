import os
import json

from stem.model import Model
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.boundary import DisplacementConstraint
from stem.load import LineLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    LinearNewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem, Cg
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw, Interface, OnePhaseSoilInterface
from stem.stem import Stem


def test_interface():
    """
    This test creates a simple interface between two soil layers and applies a line load vertically on the top layer.
    There are 4 cases defined in this test:
    
    1. Interface with high stiffness and roller boundary conditions on the bottom soil side. Here the interface
    behaves like a rigid connection, so like it does not exist. Therefore, the load is transferred to the bottom layer
    where we expect displacements.
    
    2. Interface with low stiffness and roller boundary conditions on the bottom soil side. Here the interface behaves
    like a separation, so the load is not transferred to the bottom layer. Therefore, we expect no displacements
    in the bottom layer.
    
    3. Interface with high stiffness and no roller conditions on the sides. Here the interface behaves like a rigid
    connection, so like it does not exist. Therefore, the load is transferred to the bottom layer where we expect
    displacements. Here the displacements should be higher than in case 1.
    
    4. Interface with low stiffness and no roller conditions on the sides. Here the interface behaves like a separation,
    so the load is not transferred to the bottom layer. Therefore, we expect no displacements in the bottom layer. 

    """
    # Set up the settings for the tests
    names = ["interface_high_stiffness_roller", "interface_low_stiffness_roller",
             "interface_high_stiffness_no_roller", "interface_low_stiffness_no_roller"]
    output_dir = "benchmark_tests/test_interfaces/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    roller_settings = [True, True, False, False]
    stiffness_interface = [1e20, 1e3, 1e20, 1e1]
    results = {}
    for name, roller, stiffness in zip(names, roller_settings, stiffness_interface):
        ndim = 2
        model = Model(ndim)
        # Specify material model
        # Linear elastic drained soil with a Density of 2700, a Young's modulus of 50e6,
        # a Poisson ratio of 0.3 & a Porosity of 0.3 is specified.
        DENSITY_SOLID = 2700
        POROSITY = 0.0
        YOUNG_MODULUS = 1e6
        POISSON_RATIO = 0.0
        soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
        constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
        retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
        material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)
        # Specify the coordinates for the column: x:1m x y:10m
        layer1_coordinates = [(1.0, 0.0, 0), (2.0, 0.0, 0), (2.0, 1.0, 0), (1.0, 1.0, 0)]

        # Create the soil layer
        model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_block")

        # add another material on top of the first one
        layer2_coordinates = [(1.0, 0.0, 0), (2.0, 0.0, 0), (2.0, 2.0, 0), (1.0, 2.0, 0)]
        model.add_soil_layer_by_coordinates(layer2_coordinates, material1, "soil_block_2")

        constitutive_law2 = LinearElasticSoil(YOUNG_MODULUS=stiffness, POISSON_RATIO=POISSON_RATIO)
        variables = OnePhaseSoilInterface(ndim,
                                          IS_DRAINED=True,
                                          DENSITY_SOLID=DENSITY_SOLID,
                                          POROSITY=POROSITY,
                                          MINIMUM_JOINT_WIDTH=0.0001)
        interface_material = Interface(name="interface",
                                       constitutive_law=constitutive_law2,
                                       soil_formulation=variables,
                                       retention_parameters=retention_parameters1)

        model.set_interface_between_model_parts(["soil_block"], ["soil_block_2"], interface_material)
        #model.show_geometry(show_line_ids=True)
        # Boundary conditions and Loads
        load_coordinates = [(1.0, 1.0, 0), (1.0, 2.0, 0)]

        # Add line load
        line_load = LineLoad(active=[True, False, False], value=[100, 0, 0])
        model.add_load_by_coordinates(load_coordinates, line_load, "load")

        # show the model
        #model.show_geometry(show_line_ids=True)

        # Define boundary conditions
        no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, True, True],
                                                            value=[0, 0, 0])

        sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

        # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
        model.add_boundary_condition_by_geometry_ids(1, [5], no_displacement_parameters, "base_fixed")
        if roller:
            model.add_boundary_condition_by_geometry_ids(1, [7, 6], sym_parameters, "side_rollers")

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
        problem = Problem(problem_name=name, number_of_threads=4, settings=solver_settings)
        model.project_parameters = problem

        # Define the results to be written to the output file

        # Nodal results
        nodal_results = [
            NodalOutput.DISPLACEMENT,
        ]
        # Gauss point results
        gauss_point_results = []

        # Define the output process

        vtk_output_process = Output(part_name="porous_computational_model_part",
                                    output_name="vtk_output",
                                    output_dir=output_dir,
                                    output_parameters=VtkOutputParameters(file_format="binary",
                                                                          output_interval=1,
                                                                          nodal_results=nodal_results,
                                                                          gauss_point_results=gauss_point_results,
                                                                          output_control_type="step"))
        model.output_settings.append(vtk_output_process)

        model.add_output_settings_by_coordinates( 
            coordinates=[(1.5, 2.0, 0), (1.5, 0.75, 0)],
            output_parameters=JsonOutputParameters(output_interval=0.5,
                                                   nodal_results=nodal_results,
                                                   gauss_point_results=gauss_point_results),
            part_name="calculated_output",
            output_dir=output_dir),
        # Set mesh size
        # --------------------------------
        model.set_mesh_size(element_size=0.25)
        input_folder = "benchmark_tests/test_interfaces/" + name
        stem = Stem(model, input_folder)
        stem.write_all_input_files()
        stem.run_calculation()
        # read the json output file
        output_file = os.path.join(input_folder, output_dir, "calculated_output.json")
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                output_data = f.read()
            output_data = json.loads(output_data)
            results[name] = output_data
        else:
            results[name] = None
    # Check the results
    # For each case, check the displacements in the bottom layer are as expected
    assert results["interface_high_stiffness_roller"] is not None, "Output file for high stiffness roller case not found"
    assert results["interface_low_stiffness_roller"] is not None, "Output file for low stiffness roller case not found"
    assert results["interface_high_stiffness_no_roller"] is not None, "Output file for high stiffness no roller case not found"
    assert results["interface_low_stiffness_no_roller"] is not None, "Output file for low stiffness no roller case not found"

    # Collect the displacements from the results top nodes
    top_node_disp_high_stiffness_roller_x = results['interface_high_stiffness_roller']['NODE_7']['DISPLACEMENT_X'][0] 
    top_node_disp_low_stiffness_roller_x = results['interface_low_stiffness_roller']['NODE_7']['DISPLACEMENT_X'][0]
    top_node_disp_high_stiffness_no_roller_x = results['interface_high_stiffness_no_roller']['NODE_7']['DISPLACEMENT_X'][0]
    top_node_disp_low_stiffness_no_roller_x = results['interface_low_stiffness_no_roller']['NODE_7']['DISPLACEMENT_X'][0]
    bottom_node_disp_high_stiffness_roller_x = results['interface_high_stiffness_roller']['NODE_8']['DISPLACEMENT_X'][0]
    bottom_node_disp_low_stiffness_roller_x = results['interface_low_stiffness_roller']['NODE_8']['DISPLACEMENT_X'][0]
    bottom_node_disp_high_stiffness_no_roller_x = results['interface_high_stiffness_no_roller']['NODE_8']['DISPLACEMENT_X'][0]
    bottom_node_disp_low_stiffness_no_roller_x = results['interface_low_stiffness_no_roller']['NODE_8']['DISPLACEMENT_X'][0]

    # top node displaces more than bottom node in all cases
    assert top_node_disp_high_stiffness_roller_x > bottom_node_disp_high_stiffness_roller_x
    assert top_node_disp_low_stiffness_roller_x > bottom_node_disp_low_stiffness_roller_x
    assert top_node_disp_high_stiffness_no_roller_x > bottom_node_disp_high_stiffness_no_roller_x
    assert top_node_disp_low_stiffness_no_roller_x > bottom_node_disp_low_stiffness_no_roller_x
    # check that the displacements in the top are higher for the low stiffness cases
    assert top_node_disp_high_stiffness_roller_x < top_node_disp_low_stiffness_roller_x
    assert top_node_disp_high_stiffness_no_roller_x < top_node_disp_low_stiffness_no_roller_x
