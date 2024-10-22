import os
import sys
from shutil import rmtree
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters, GaussPointOutput
from stem.stem import Stem
from benchmark_tests.utils import assert_floats_in_directories_almost_equal


def plot_displacement(data_dict):
    # Create a 1x3 subplot figure for X, Y, Z displacements
    fig = make_subplots(rows=1, cols=3, subplot_titles=("DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"))

    # Create line plots for each displacement
    for key in data_dict.keys():
        fig.add_trace(go.Scatter(x=list(range(len(data_dict[key]['DISPLACEMENT_X']))),
                                 y=data_dict[key]['DISPLACEMENT_X'],
                                 mode='lines',
                                 name=f'{key}_DISPLACEMENT_X'),
                      row=1,
                      col=1)

        fig.add_trace(go.Scatter(x=list(range(len(data_dict[key]['DISPLACEMENT_Y']))),
                                 y=data_dict[key]['DISPLACEMENT_Y'],
                                 mode='lines',
                                 name=f'{key}_DISPLACEMENT_Y'),
                      row=1,
                      col=2)

        fig.add_trace(go.Scatter(x=list(range(len(data_dict[key]['DISPLACEMENT_Z']))),
                                 y=data_dict[key]['DISPLACEMENT_Z'],
                                 mode='lines',
                                 name=f'{key}_DISPLACEMENT_Z'),
                      row=1,
                      col=3)

    # Update layout to show dropdown for selecting datasets
    dropdown_buttons = [
        dict(label=key,
             method='update',
             args=[{
                 'visible': [key in trace.name for trace in fig.data]
             }, {
                 'title': f'Displacements for {key}'
             }]) for key in data_dict.keys()
    ]

    fig.update_layout(updatemenus=[
        dict(active=0,
             buttons=dropdown_buttons,
             direction='down',
             showactive=True,
             x=0.5,
             y=1.15,
             xanchor='center',
             yanchor='top')
    ])

    # Set initial visibility (only the first dataset)
    for i, trace in enumerate(fig.data):
        trace.visible = (i < 3)  # First dataset is visible, rest are hidden

    fig.update_layout(title='Displacement Data Visualization')
    fig.write_html("displacement_plot.html")


if __name__ == "__main__":
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 19.5

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
    layer1_coordinates = [(0.0, 0.0, -5.0), (5.0, 0.0, -5.0), (5.0, 1.0, -5.0), (0.0, 1.0, -5.0)]
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
    layer2_coordinates = [(0.0, 1.0, -5.0), (5.0, 1.0, -5.0), (5.0, 2.5, -5.0), (0.0, 2.5, -5.0)]
    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer2_coordinates, material2, "soil_layer_2")

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

    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    origin_point = np.array([2.5, 2.5, -5.0])
    direction_vector = np.array([0, 0, 1])
    rail_pad_thickness = 0.025

    # create a straight track with rails, sleepers and rail pads
    model.generate_straight_track(sleeper_distance=0.5,
                                  n_sleepers=40,
                                  rail_parameters=rail_parameters,
                                  sleeper_parameters=sleeper_parameters,
                                  rail_pad_parameters=rail_pad_parameters,
                                  rail_pad_thickness=rail_pad_thickness,
                                  origin_point=origin_point,
                                  direction_vector=direction_vector,
                                  name="rail_track_1")

    origin = [float(origin_point[0]), float(origin_point[1] + rail_pad_thickness), float(origin_point[-1])]
    moving_load = MovingLoad(load=[0.0, -10000.0, 0.0],
                             direction=[1, 1, 1],
                             velocity=140 / 3.6,
                             origin=origin,
                             offset=0.0)

    model.add_load_on_line_model_part("rail_track_1", moving_load, "moving_load")

    # model.show_geometry(show_surface_ids=True, show_point_ids=False)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 9, 10, 11], roller_displacement_parameters,
                                                 "roller_fixed")

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
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-0,
                                                            displacement_absolute_tolerance=1.0e-0)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.01,
                                     rayleigh_m=0.0001)

    # Set up problem data
    problem = Problem(problem_name="test_extended_beam", number_of_threads=4, settings=solver_settings)
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
    # gauss_point_results = [GaussPointOutput.CAUCHY_STRESS_VECTOR,GaussPointOutput.CAUCHY_STRESS_TENSOR ]
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
    model.output_settings = [vtk_output_process]
    coordinates = [[2.5, 2.5, 0.0], [2.5, 2.5, 0.5], [2.5, 2.5, 2.5], [2.5, 2.5, 5.0], [2.5, 2.5, 7.5], [2.5, 2.5, 9.0],
                   [2.5, 2.5, 10.0]]
    coordinates_on_beam = [[coord[0], coord[1] + rail_pad_thickness, coord[2]] for coord in coordinates]
    model.add_output_settings_by_coordinates(part_name="subset_outputs",
                                             output_name="json_output",
                                             coordinates=coordinates + coordinates_on_beam,
                                             output_parameters=JsonOutputParameters(
                                                 output_interval=0.001,
                                                 nodal_results=nodal_results,
                                                 gauss_point_results=gauss_point_results))

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.25)

    input_folder = "benchmark_tests/test_extended_beam/inputs_kratos_full_qs"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()
    expected_output_dir_temp = "benchmark_tests/test_extended_beam/inputs_kratos_full_qs"
    # node coordinates for the soil layers
    import json
    node_coordinates_soil_1 = model.body_model_parts[0].mesh.nodes
    node_coordinates_soil_2 = model.body_model_parts[1].mesh.nodes
    element_coordinates_soil_1 = model.body_model_parts[0].mesh.elements
    element_coordinates_soil_2 = model.body_model_parts[1].mesh.elements
    # from stem coordinates to dict
    node_coordinates_soil_1 = {f"NODE_{node_id}": node.coordinates for node_id, node in node_coordinates_soil_1.items()}
    node_coordinates_soil_2 = {f"NODE_{node_id}": node.coordinates for node_id, node in node_coordinates_soil_2.items()}
    element_coordinates_soil_1 = {
        f"ELEMENT_{element_id}": [node for node in element.node_ids]
        for element_id, element in element_coordinates_soil_1.items()
    }
    element_coordinates_soil_2 = {
        f"ELEMENT_{element_id}": [node for node in element.node_ids]
        for element_id, element in element_coordinates_soil_2.items()
    }
    # save the node coordinates
    with open(os.path.join(expected_output_dir_temp, "soil_1_nodes.json"), "w") as f:
        json.dump(node_coordinates_soil_1, f)
    with open(os.path.join(expected_output_dir_temp, "soil_2_nodes.json"), "w") as f:
        json.dump(node_coordinates_soil_2, f)
    with open(os.path.join(expected_output_dir_temp, "soil_1_elements.json"), "w") as f:
        json.dump(element_coordinates_soil_1, f)
    with open(os.path.join(expected_output_dir_temp, "soil_2_elements.json"), "w") as f:
        json.dump(element_coordinates_soil_2, f)

    ## find the maximum z coordinate of the nodes for all model parts
    #max_z = -10000
    #min_z = 10000
    #for part in model.body_model_parts:
    #    for counter, node in part.geometry.points.items():
    #        if node.coordinates[2] > max_z:
    #            max_z = node.coordinates[2]
    #        if node.coordinates[2] < min_z:
    #            min_z = node.coordinates[2]

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    # check that the output is as expected (only the nodal displacements are checked)
    #with open(os.path.join(expected_output_dir_temp, "soil_layer_2.json"), "r") as f:
    #    soil_output = json.load(f)
    #nodes_id = [name.split("_")[-1] for name, item in soil_output.items() if 'TIME' not in name]
    ## collect node coordinates
    #nodes_soil = model.body_model_parts[0].mesh.nodes
    #node_coordinates = np.array([nodes_soil[int(node_id)].coordinates for node_id in nodes_id])
    ## define interest points in the geometry
    #length = np.max(node_coordinates[:, 2]) - np.min(node_coordinates[:, 2])
    #depth = np.max(node_coordinates[:, 1]) - np.min(node_coordinates[:, 1])
    #x_track = origin_point[0]
    #interest_point_factors = [0.1, 0.25, 0.5, 0.75, 0.9]
    #interest_points = []
    #for factor_y in interest_point_factors:
    #    for factor_z in interest_point_factors:
    #        interest_points.append([x_track,
    #                                np.min(node_coordinates[:, 1]) + factor_y * depth,
    #                                np.min(node_coordinates[:, 2]) + factor_z * length])
    ## find the node closest to the interest points
    #interest_points_nodes = []
    #for point in interest_points:
    #    distances = np.linalg.norm(node_coordinates - point, axis=1)
    #    interest_points_nodes.append("NODE_" + nodes_id[np.argmin(distances)])
    ## get the nodal displacements for the interest points
    #interest_points_displacements = {node_id:soil_output.get(node_id) for node_id in interest_points_nodes}
    #plot_displacement(interest_points_displacements)
