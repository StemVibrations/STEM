import os
import json
import sys
from shutil import rmtree

import UVEC.uvec_ten_dof_vehicle_2D as uvec
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.default_materials import DefaultMaterial
from stem.load import UvecLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
     LinearNewtonRaphsonStrategy, NewmarkScheme, Cg, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem

from benchmark_tests.utils import assert_files_equal
from tests.utils import TestUtils


def test_train_track_uvec_soil_3d():
    """
    Test the UVEC on a 3D soil model.
    """
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 50

    # Specify materials: 3 Layers
    solid_density_1 = 2650
    porosity_1 = 0.3
    young_modulus_1 = 30e6
    poisson_ratio_1 = 0.2
    soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity_1)
    constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio_1)
    retention_parameters_1 = SaturatedBelowPhreaticLevelLaw()
    material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, retention_parameters_1)

    solid_density_2 = 2550
    porosity_2 = 0.3
    young_modulus_2 = 30e6
    poisson_ratio_2 = 0.2
    soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_2, POROSITY=porosity_2)
    constitutive_law_2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_2, POISSON_RATIO=poisson_ratio_2)
    retention_parameters_2 = SaturatedBelowPhreaticLevelLaw()
    material_soil_2 = SoilMaterial("soil_2", soil_formulation_2, constitutive_law_2, retention_parameters_2)

    solid_density_3 = 2650
    porosity_3 = 0.3
    young_modulus_3 = 10e6
    poisson_ratio_3 = 0.2
    soil_formulation_3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity_3)
    constitutive_law_3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio_3)
    retention_parameters_3 = SaturatedBelowPhreaticLevelLaw()
    material_embankment = SoilMaterial("embankment", soil_formulation_3, constitutive_law_3, retention_parameters_3)

    # Rail and sleeper parameters
    rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
    rail_pad_parameters = ElasticSpringDamper(
        NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
        NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
        NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],  # damping coefficient [Ns/m]
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    # create cross section
    soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]

    # add soil to geometry
    model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
    model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
    model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")

    # Create track
    origin_point = [0.75, 3.0, 0.0]
    direction_vector = [0, 0, 1]
    number_of_sleepers = 101
    sleeper_spacing = 0.5
    rail_pad_thickness = 0.025

    model.generate_straight_track(sleeper_spacing, number_of_sleepers, rail_parameters, sleeper_parameters,
                                  rail_pad_parameters, rail_pad_thickness, origin_point, direction_vector, "rail_track")

    # define uvec parameters
    uvec_parameters = {
        "n_carts": 1,  # number of carts [-]
        "cart_inertia": (1128.8e3) / 2,  # inertia of the cart [kgm2]
        "cart_mass": (50e3) / 2,  # mass of the cart [kg]
        "cart_stiffness": 2708e3,  # stiffness between the cart and bogies [N/m]
        "cart_damping": 64e3,  # damping coefficient between the cart and bogies [Ns/m]
        "bogie_distances": [-9.95, 9.95],  # distances of the bogies from the centre of the cart [m]
        "bogie_inertia": (0.31e3) / 2,  # inertia of the bogie [kgm2]
        "bogie_mass": (6e3) / 2,  # mass of the bogie [kg]
        "wheel_distances": [-1.25, 1.25],  # distances of the wheels from the centre of the bogie [m]
        "wheel_mass": 1.5e3,  # mass of the wheel [kg]
        "wheel_stiffness": 4800e3,  # stiffness between the wheel and the bogie [N/m]
        "wheel_damping": 0.25e3,  # damping coefficient between the wheel and the bogie [Ns/m]
        "gravity_axis": 1,  # axis on which gravity works [x =0, y = 1, z = 2]
        "contact_coefficient": 9.1e-7,  # Hertzian contact coefficient between the wheel and the rail [N/m]
        "contact_power": 1.0,  # Hertzian contact power between the wheel and the rail [-]
        "static_initialisation": False,  # True if the analysis of the UVEC is static
    }

    # define the UVEC load
    uvec_load = UvecLoad(direction=[1, 1, 1],
                         velocity=40,
                         origin=[0.75, 3 + rail_pad_thickness, 5],
                         wheel_configuration=[0.0, 2.5, 19.9, 22.4],
                         uvec_model=uvec,
                         uvec_parameters=uvec_parameters)

    # add the load on the tracks
    model.add_load_on_line_model_part("rail_track", uvec_load, "train_load")

    # define BC
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [4, 10, 16], roller_displacement_parameters, "sides_roller")
    model.add_boundary_condition_by_geometry_ids(2, [2, 5, 6, 7, 11, 12, 15, 17, 18], absorbing_boundaries_parameters,
                                                 "abs")

    # coarse mesh
    model.set_mesh_size(element_size=1.0)

    # analysis parameters
    end_time = 0.5
    delta_time = 1e-03
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC

    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=end_time,
                                       delta_time=delta_time,
                                       reduction_factor=1,
                                       increase_factor=1,
                                       max_delta_time_factor=1000)

    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)

    strategy_type = LinearNewtonRaphsonStrategy()
    scheme_type = NewmarkScheme()
    linear_solver_settings = Cg(tolerance=1e-6)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=strategy_type,
                                     scheme=scheme_type,
                                     linear_solver_settings=linear_solver_settings,
                                     rayleigh_k=0.0002,
                                     rayleigh_m=0.6)

    # Set up problem data
    problem = Problem(problem_name="calculate_uvec_on_embankment_with_absorbing_boundaries",
                      number_of_threads=4,
                      settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
    gauss_point_results = []

    model.add_output_settings(part_name="porous_computational_model_part",
                              output_dir="output",
                              output_name="vtk_output",
                              output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=50,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=gauss_point_results,
                                                                    output_control_type="step"))

    # define nodes for JSON output
    desired_output_points = [(0.0, 3.0, 25.0), (0.75, 3.0, 25.0), (1.5, 3.0, 25.0), (3, 2.0, 25.0), (4, 2.0, 25.0),
                             (5, 2.0, 25.0)]

    model.add_output_settings_by_coordinates(part_name="subset_outputs",
                                             output_dir="output",
                                             output_name="json_output",
                                             coordinates=desired_output_points,
                                             output_parameters=JsonOutputParameters(
                                                 output_interval=delta_time,
                                                 nodal_results=nodal_results,
                                                 gauss_point_results=gauss_point_results))

    # write the files and run the calculation
    input_folder = "benchmark_tests/test_train_track_uvec_soil_3d/inputs_kratos"
    stem = Stem(model, input_folder)

    stem.write_all_input_files()

    stem.run_calculation()

    if sys.platform == "win32":
        expected_output_dir = "benchmark_tests/test_train_track_uvec_soil_3d/output_windows"
    elif sys.platform == "linux":
        expected_output_dir = "benchmark_tests/test_train_track_uvec_soil_3d/output_linux"
    else:
        raise Exception("Unknown platform")
    # compare VTK files
    assert assert_files_equal(os.path.join(expected_output_dir, "output_vtk_porous_computational_model_part"),
                              os.path.join(input_folder, "output/output_vtk_porous_computational_model_part"))

    # compare json files
    with open(os.path.join(expected_output_dir, "json_output.json"), 'r') as f:
        expected_json = json.load(f)
    with open(os.path.join(input_folder, "output/json_output.json"), 'r') as f:
        actual_json = json.load(f)

    TestUtils.assert_dictionary_almost_equal(expected_json, actual_json)

    rmtree(input_folder)
