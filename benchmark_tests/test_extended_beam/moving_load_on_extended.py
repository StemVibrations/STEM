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
from stem.load import MovingLoad, PointLoad
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters, GaussPointOutput
from stem.stem import Stem

if __name__ == "__main__":
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 10
    point_load = False

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
    layer1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
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
    layer2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.5, 0.0), (0.0, 2.5, 0.0)]
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
    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                                              NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                              NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],
                                              NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    soil_equivalent_parameters = ElasticSpringDamper(
        NODAL_DISPLACEMENT_STIFFNESS=[0, 14285714.29 * 4 / 7, 0],
        NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
        # NODAL_DAMPING_COEFFICIENT=[1, 60946444.21, 1],
        NODAL_DAMPING_COEFFICIENT=[0, 1, 0],
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    origin_point = np.array([2.5, 2.5, -5.0])
    direction_vector = np.array([0, 0, 1])
    rail_pad_thickness = 0.025

    # create a straight track with rails, sleepers and rail pads
    model.generate_extended_straight_track(0.5, 40, rail_parameters, sleeper_parameters, rail_pad_parameters,
                                           rail_pad_thickness, origin_point, soil_equivalent_parameters, 2.5,
                                           direction_vector, "rail_track_1")
    origin = [float(origin_point[0]), float(origin_point[1] + rail_pad_thickness), float(origin_point[-1])]
    if point_load:
        load = PointLoad(active=[False, True, False], value=[0.0, -10000.0, 0.0])
        model.add_load_by_coordinates(coordinates=[[2.5, 2.5, 5.0]], load_parameters=load, name="point_load")
    else:
        moving_load = MovingLoad(load=[0.0, -10000.0, 0.0],
                                 direction=[1, 1, 1],
                                 velocity=140 / 3.6,
                                 origin=origin,
                                 offset=0.0)
        model.add_load_on_line_model_part("rail_track_1", moving_load, "moving_load")

    #model.show_geometry(show_surface_ids=True, show_point_ids=True)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [7, 2, 10, 5, 9, 4, 11, 6], roller_displacement_parameters,
                                                 "roller_fixed")

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    if point_load:
        solution_type = SolutionType.QUASI_STATIC
    else:
        solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.5,
                                       delta_time=0.001,
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
                   [2.5, 2.5, 10.0]],
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

    input_folder = "benchmark_tests/test_extended_beam/inputs_kratos_extended_qs"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    expected_output_dir_temp = "benchmark_tests/test_extended_beam/inputs_kratos_extended_qs"
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
    # max_z = -10000
    # min_z = 10000
    # for part in model.body_model_parts:
    #    for counter, node in part.geometry.points.items():
    #        if node.coordinates[2] > max_z:
    #            max_z = node.coordinates[2]
    #        if node.coordinates[2] < min_z:
    #            min_z = node.coordinates[2]

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()
