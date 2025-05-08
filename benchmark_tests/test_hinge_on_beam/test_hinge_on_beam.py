import os
import json

import matplotlib.pyplot as plt
import pytest

from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import *
from stem.load import PointLoad
from stem.boundary import RotationConstraint
from stem.additional_processes import HingeParameters
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, JsonOutputParameters
from stem.stem import Stem

SHOW_RESULTS = False


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)

    # Specify beam material model
    YOUNG_MODULUS = 210000000000
    POISSON_RATIO = 0.30000
    DENSITY = 7850
    CROSS_AREA = 0.01
    I22 = 0.0001
    I33 = 0.0001

    TORTIONAL_INERTIA = I22 + I33
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I33, I22, TORTIONAL_INERTIA)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    beam_coordinates = [(0, 0, 0), (1, 0, 0)]
    # Create the beam
    gmsh_input = {name: {"coordinates": beam_coordinates, "ndim": 1}}
    # check if extrusion length is specified in 3D
    model.gmsh_io.generate_geometry(gmsh_input, "")
    #
    # create body model part
    body_model_part = BodyModelPart(name)
    body_model_part.material = structural_material

    # set the geometry of the body model part
    body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, name)
    model.body_model_parts.append(body_model_part)

    # Define moving load
    vertical_force = -10000
    point_load = PointLoad(value=[0.0, vertical_force, 0.0], active=[True, True, True])

    model.add_load_by_coordinates([[0.5, 0.0, 0.0]], point_load, "point_load")

    # calculate hinge rotational stiffness based on fixity factor
    fixity_factor = 0.75
    distance_boundary_hinge = 0.5
    hinge_stiffness_y = (3 * YOUNG_MODULUS * I22) / (distance_boundary_hinge * (1 / fixity_factor - 1))
    hinge_stiffness_z = (3 * YOUNG_MODULUS * I33) / (distance_boundary_hinge * (1 / fixity_factor - 1))

    model.add_hinge_on_beam("beam", [(distance_boundary_hinge, 0.0, 0.0)],
                            HingeParameters(hinge_stiffness_y, hinge_stiffness_z), "hinge")

    # Define rotation boundary condition
    rotation_boundaries_parameters = RotationConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    rotation_boundaries_parameters2 = RotationConstraint(active=[True, True, True],
                                                         is_fixed=[True, True, True],
                                                         value=[0, 0, 0])

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # model.show_geometry(show_point_ids=True, show_line_ids=True)
    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(0, [1], rotation_boundaries_parameters, "rotation")
    model.add_boundary_condition_by_geometry_ids(0, [2], rotation_boundaries_parameters2, "rotation2")
    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.01)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=1.0,
                                       delta_time=0.5,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE

    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion)

    # Set up problem data
    problem = Problem(problem_name="hinge_on_beam", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    model.add_output_settings(output_parameters=JsonOutputParameters(output_interval=1.0,
                                                                     nodal_results=nodal_results,
                                                                     gauss_point_results=gauss_point_results),
                              part_name="beam",
                              output_dir="output",
                              output_name="json_output")

    input_folder = "benchmark_tests/test_hinge_on_beam/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    with open(os.path.join(input_folder, "output/json_output.json"), "r") as file:
        calculated_output = json.load(file)

    if SHOW_RESULTS:

        # get coordinates and vertical displacements
        coordinates = []
        displacements = []
        for key, value in calculated_output.items():
            if key != "TIME":
                coordinates.append(value["COORDINATES"])
                displacements.append(value["DISPLACEMENT_Y"])

        # sort based on x coordinate
        sorted_pairs = sorted(zip(coordinates, displacements), key=lambda pair: pair[0][0])

        # Unpack sorted pairs
        sorted_coordinates, sorted_values = zip(*sorted_pairs)

        # Plot the results
        plt.plot([x[0] for x in sorted_coordinates], sorted_values)
        plt.xlabel("x [m]")
        plt.ylabel("Displacement [m]")

        plt.show()

    # analytical solution for expected vertical displacement at hinge
    u_expected = vertical_force * distance_boundary_hinge**3 * (
        4 * YOUNG_MODULUS * I33 + hinge_stiffness_z * distance_boundary_hinge) / (
            24 * YOUNG_MODULUS * I33 * (YOUNG_MODULUS * I33 + hinge_stiffness_z * distance_boundary_hinge))

    assert u_expected == pytest.approx(calculated_output["NODE_3"]["DISPLACEMENT_Y"][0], rel=1e-6)
