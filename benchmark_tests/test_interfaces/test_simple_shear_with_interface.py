import os
import json
from shutil import rmtree

import pytest
import numpy as np

from stem.model import Model
from stem.boundary import DisplacementConstraint
from stem.load import LineLoad, SurfaceLoad
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, Lu)
from stem.output import NodalOutput, Output, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import (SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw,
                                InterfaceMaterial)
from stem.stem import Stem


@pytest.mark.parametrize("element_order", [1, 2])
@pytest.mark.parametrize("interface_type", ["weak", "stiff"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_simple_shear_with_interface_elements(element_order, interface_type, ndim):
    """
    simple shear test with first and second order elements, with a weak and stiff interface, in 2D and 3D.
    The test consists of two soil layers with an interface in between.
    A horizontal load is applied on the top of the model and the displacements at the top and bottom of the
    interface are compared to the expected values.

    """

    test_name = f"test_{interface_type}_interface_{ndim}d_element_order_{element_order}"

    output_dir = "output"
    model = Model(ndim)
    model.extrusion_length = 1.0

    DENSITY_SOLID = 2700
    POROSITY = 0.0
    YOUNG_MODULUS = 1e14
    POISSON_RATIO = 0.1
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    max_x_coord = 1
    layer1_coordinates = [
        (0.0, 0.0, 0),
        (max_x_coord, 0.0, 0),
        (max_x_coord, 1.0, 0),
        (0.0, 1.0, 0),
    ]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "bottom_soil")

    # add another material on top of the first one
    layer2_coordinates = [
        (0.0, 1.0, 0),
        (max_x_coord, 1.0, 0),
        (max_x_coord, 2.0, 0),
        (0.0, 2.0, 0),
    ]
    model.add_soil_layer_by_coordinates(layer2_coordinates, material1, "top_soil")

    if interface_type == "weak":
        interface_young_modulus = 1e2
    else:
        interface_young_modulus = 1e20

    constitutive_law2 = LinearElasticSoil(YOUNG_MODULUS=interface_young_modulus, POISSON_RATIO=POISSON_RATIO)

    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)

    interface_material = InterfaceMaterial(name="interface",
                                           constitutive_law=constitutive_law2,
                                           soil_formulation=soil_formulation,
                                           retention_parameters=retention_parameters1)

    load_value = 200

    if ndim == 2:
        load_coordinates = [(0.0, 2.0, 0), (max_x_coord, 2.0, 0)]
        load = LineLoad(active=[True, True, True], value=[load_value, 0, 0])
    else:
        load_coordinates = [(0.0, 2.0, 0), (max_x_coord, 2.0, 0), (max_x_coord, 2.0, model.extrusion_length),
                            (0.0, 2.0, model.extrusion_length)]
        load = SurfaceLoad(active=[True, True, True], value=[load_value, 0, 0])

    model.add_load_by_coordinates(load_coordinates, load, "load")

    # only get output on the top of the interface
    connections_dictionary = {"calculated_output": [False, True]}
    model.set_interface_between_model_parts(["bottom_soil"], ["top_soil"], interface_material, "interface_top_bot",
                                            connections_dictionary)

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    top_boundary = DisplacementConstraint(is_fixed=[False, True, True], value=[0, 0, 0])

    soil_sides = DisplacementConstraint(is_fixed=[False, True, True], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    if ndim == 2:
        # model.show_geometry(show_line_ids=True)
        model.add_boundary_condition_by_geometry_ids(1, [6], top_boundary, "top_boundary")
        model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(1, [7, 5, 2, 4], soil_sides, "soil_sides")
    else:
        # model.show_geometry(show_surface_ids=True)
        model.add_boundary_condition_by_geometry_ids(2, [8], top_boundary, "top_boundary")
        model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 9, 10, 11, 7], soil_sides, "soil_sides")

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(
        start_time=0.0,
        end_time=1.0,
        delta_time=1.00,
        reduction_factor=1.0,
        increase_factor=1.0,
        max_delta_time_factor=1000,
    )
    convergence_criterion = DisplacementConvergenceCriteria(
        displacement_relative_tolerance=1.0e-6,
        displacement_absolute_tolerance=1.0e-12,
    )
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=NewtonRaphsonStrategy(),
                                     linear_solver_settings=Lu())

    # Set up problem data
    problem = Problem(problem_name=test_name, number_of_threads=4, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    # vtk_output_process = Output(
    #     part_name="porous_computational_model_part",
    #     output_name="vtk_output",
    #     output_dir=output_dir,
    #     output_parameters=VtkOutputParameters(
    #         file_format="binary",
    #         output_interval=1,
    #         nodal_results=nodal_results,
    #         gauss_point_results=gauss_point_results,
    #         output_control_type="step",
    #     ),
    # )
    # model.output_settings.append(vtk_output_process)

    model.add_output_settings_by_coordinates(coordinates=[(0.0, 1.0, 0), (0.0, 2.0, 0)],
                                             output_parameters=JsonOutputParameters(
                                                 output_interval=0.5,
                                                 nodal_results=nodal_results,
                                                 gauss_point_results=gauss_point_results,
                                             ),
                                             part_name="calculated_output",
                                             output_dir=output_dir)
    # Set mesh size
    # --------------------------------

    model.mesh_settings.element_order = element_order
    model.set_mesh_size(element_size=1.0)

    input_folder = "benchmark_tests/test_interfaces/" + test_name
    stem = Stem(model, input_folder)
    stem.write_all_input_files()
    stem.run_calculation()
    # read the json output file
    output_file = os.path.join(input_folder, output_dir, "calculated_output.json")

    with open(output_file, "r") as f:
        output_data = json.load(f)

    output_data.pop("TIME", None)
    calculated_bottom_disp, calculated_top_disp = np.nan, np.nan
    for node_key, results in output_data.items():
        if np.allclose(results["COORDINATES"], [0.0, 1.0, 0]):
            calculated_bottom_disp = results["DISPLACEMENT_X"][0]
        elif np.allclose(results["COORDINATES"], [0.0, 2.0, 0]):
            calculated_top_disp = results["DISPLACEMENT_X"][0]

    if interface_type == "weak":
        shear_modulus_interface = interface_young_modulus / (2 * (1 + POISSON_RATIO))
        expected_top_disp = load_value * 1 / shear_modulus_interface
        expected_bottom_disp = load_value * 1 / shear_modulus_interface

    else:
        shear_modulus = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
        expected_top_disp = load_value * 2 / shear_modulus
        expected_bottom_disp = load_value * 1 / shear_modulus

    assert expected_top_disp == pytest.approx(
        calculated_top_disp, rel=1e-6
    ), f"Calculated top displacement {calculated_top_disp} does not match expected value {expected_top_disp}"
    assert expected_bottom_disp == pytest.approx(
        calculated_bottom_disp, rel=1e-6
    ), f"Calculated bottom displacement {calculated_bottom_disp} does not match expected value {expected_bottom_disp}"

    rmtree(input_folder)
