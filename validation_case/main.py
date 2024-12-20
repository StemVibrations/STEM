import math
import os

import numpy as np

from stem.default_materials import DefaultMaterial
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad, UvecLoad
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.table import Table
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def square_coordinates(width, height, start_x, start_y):

    return [(start_x, start_y, 0), (start_x + width, start_y, 0), (start_x + width, start_y + height, 0),
            (start_x, start_y + height, 0)]


def run_model():

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALISATION AND PARAMETER SPECIFICATION
    # ------------------------------------------------------------------------------------------------------------------
    model_width = 15
    output_distance = 10
    model_length = 50
    extension_tracks_before_model = 50
    extension_tracks_after_model = 5

    ballast_width = 2.2
    ballast_height = 0.7
    ballast_slope = 15
    ballast_bottom_width = ballast_width + ballast_height / math.tan(math.radians(ballast_slope))

    # ballast_to_trench_distance = 0
    # trench_slope = 5
    # trench_depth = 0.20
    # trench_bottom_width = 10
    # trench_top_width = trench_bottom_width + trench_depth / math.tan(math.radians(trench_slope))

    rail_offset = 0.75

    # Specify dimension and initiate the model
    ndim = 3
    model_stage_1 = Model(ndim)
    model_stage_1.extrusion_length = model_length

    # trench_width_slope = (trench_top_width - trench_bottom_width) / 2

    # ------------------------------------------------------------------------------------------------------------------
    # SOIL AND BALLAST GEOMETRY AND MATERIAL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    # soil_layer_1 = [
    #     (0, 0, 0), (ballast_to_trench_distance + ballast_bottom_width, 0, 0),
    #     (ballast_to_trench_distance + ballast_bottom_width + trench_width_slope, -trench_depth, 0),
    #     (ballast_to_trench_distance + ballast_bottom_width + trench_width_slope + trench_bottom_width, -trench_depth, 0),
    #     (ballast_to_trench_distance + ballast_bottom_width + trench_top_width, 0, 0),
    #     (model_width, 0, 0),
    #     (model_width, -1.8, 0),
    #     (0, -1.8, 0)]

    # soil_layer_1 = [
    #     (0, 0, 0), (ballast_to_trench_distance + ballast_bottom_width, 0, 0),
    #     (ballast_to_trench_distance + ballast_bottom_width + trench_width_slope, -trench_depth, 0),
    #     (model_width, -trench_depth, 0),
    #     (ballast_to_trench_distance + ballast_bottom_width + trench_top_width, 0, 0),
    #     (model_width, 0, 0),
    #     (model_width, -1.8, 0),
    #     (0, -1.8, 0)]

    soil_layer_1 = [
        (0, 0, 0),
        (4.4, 0, 0),
        # (6.7, 0, 0),
        # (16.7, -0.2, 0), (19, 0.0, 0),
        (model_width, 0, 0),
        # (model_width, 0, 0),
        (model_width, -1.8, 0),
        (0, -1.8, 0)
    ]

    soil_layer_2 = square_coordinates(
        model_width,
        3.0,
        0,
        -4.8,
    )
    soil_layer_3 = square_coordinates(
        model_width,
        3.6,
        0,
        -8.4,
    )
    soil_layer_4 = square_coordinates(
        model_width,
        0.5,
        0,
        -8.9,
    )
    soil_layer_5 = square_coordinates(
        model_width,
        3.1,
        0,
        -12.0,
    )

    ballast_geometry = [(0, 0, 0), (4.4, 0, 0), (2.4, 0.8, 0), (0, 0.8, 0)]

    soil_properties = {
        "soil_layer_1": [soil_layer_1, 2010, 3.16e8, 0.313],
        "soil_layer_2": [soil_layer_2, 2012, 3.70e8, 0.303],
        # "soil_layer_3": [soil_layer_3, 2063, 4.70e8, 0.309],
        # "soil_layer_4": [soil_layer_4, 1969, 3.64e8, 0.309],
        # "soil_layer_5": [soil_layer_5, 1990, 4.43e8, 0.34]
    }

    all_soil_coordinates = [np.array(ss[0]) for ss in soil_properties.values()]
    all_soil_coordinates = np.concatenate(all_soil_coordinates)
    # soil layers
    POROSITY = 0.3
    for part_name, (soil_coordinates, soil_density, soil_e_modulus, soil_poisson) in soil_properties.items():

        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=soil_density, POROSITY=POROSITY)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=soil_e_modulus, POISSON_RATIO=soil_poisson)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        material = SoilMaterial(part_name, soil_formulation, constitutive_law, retention_parameters)
        model_stage_1.add_soil_layer_by_coordinates(soil_coordinates, material, part_name)

    # Ballast
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2000, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=2.2e7, POISSON_RATIO=0.3)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material6 = SoilMaterial("ballast", soil_formulation1, constitutive_law1, retention_parameters1)

    # Create the soil layer
    model_stage_1.add_soil_layer_by_coordinates(ballast_geometry, material6, "ballast")

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD PARAMETERS AND GEOMETRY
    # ------------------------------------------------------------------------------------------------------------------

    # use the median velocity from the distribution

    rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
    rail_pad_parameters = ElasticSpringDamper(
        NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
        NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
        NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],  # damping coefficient [Ns/m]
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    soil_equivalent_parameters = ElasticSpringDamper(
        NODAL_DISPLACEMENT_STIFFNESS=[0, 14285714.29 * 4 / 7, 0],
        NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
        # NODAL_DAMPING_COEFFICIENT=[1, 60946444.21, 1],
        NODAL_DAMPING_COEFFICIENT=[0, 1, 0],
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
    origin_point = [rail_offset, ballast_height, -extension_tracks_before_model]

    direction_vector = [0, 0, 1]
    sleeper_spacing = 0.5
    rail_pad_thickness = 0.025
    length_soil_equivalent_element = 2.5  #m

    number_of_sleepers = int(
        (model_length + extension_tracks_before_model + extension_tracks_after_model) / sleeper_spacing) + 1

    # model_stage_1.generate_straight_track(sleeper_spacing, number_of_sleepers, rail_parameters,
    #                               sleeper_parameters, rail_pad_parameters,
    #                               rail_pad_thickness, origin_point,
    #                               direction_vector, "rail_track_1")

    model_stage_1.generate_extended_straight_track(sleeper_spacing, number_of_sleepers, rail_parameters,
                                                   sleeper_parameters, rail_pad_parameters, rail_pad_thickness,
                                                   origin_point, soil_equivalent_parameters,
                                                   length_soil_equivalent_element, direction_vector, "rail_track_1")

    # define uvec parameters
    uvec_parameters = {
        "n_carts": 2,  # number of carts [-]
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
        "initialisation_steps": 20,  # number of time steps on which the gravity on the UVEC is
        # gradually increased [-]
    }

    def get_wheel_configuration(start_position_cart: float, cart_to_cart_distance: float, uvec_parameters: dict):

        interdistance_wheels = []
        for boogie_distance in uvec_parameters['bogie_distances']:
            for wheel_distance in uvec_parameters['wheel_distances']:
                interdistance_wheels.append(boogie_distance + wheel_distance)

        interdistance_wheels = np.asarray(interdistance_wheels)
        interdistance_wheels -= interdistance_wheels.min()

        wheel_configuration = []
        progressive_distance_start_position = start_position_cart
        for i in range(uvec_parameters["n_carts"]):

            current_configuration = interdistance_wheels + progressive_distance_start_position
            wheel_configuration.extend(np.round(current_configuration.tolist(), 3))
            progressive_distance_start_position = current_configuration[-1] + cart_to_cart_distance

        return wheel_configuration

    # define the UVEC load
    uvec_load_1 = UvecLoad(direction=[1, 1, 1],
                           velocity=34.4,
                           origin=[rail_offset, ballast_height + rail_pad_thickness, -extension_tracks_before_model],
                           wheel_configuration=get_wheel_configuration(start_position_cart=0,
                                                                       cart_to_cart_distance=5.2,
                                                                       uvec_parameters=uvec_parameters),
                           uvec_file=r"../uvec_ten_dof_vehicle_2D/uvec.py",
                           uvec_function_name="uvec",
                           uvec_parameters=uvec_parameters)

    # uvec_load_2 = UvecLoad(direction=[1, 1, 1], velocity=34.4, origin=[rail_offset, ballast_height + rail_pad_thickness,
    #                                                                    -extension_tracks_before_model + 0.2 + 25],
    #                        wheel_configuration=[0.0, 2.5, 19.9, 22.4],
    #                        uvec_file=r"../uvec_ten_dof_vehicle_2D/uvec.py", uvec_function_name="uvec",
    #                        uvec_parameters=uvec_parameters)
    #
    # uvec_load_3 = UvecLoad(direction=[1, 1, 1], velocity=34.4, origin=[rail_offset, ballast_height + rail_pad_thickness,
    #                                                                    -extension_tracks_before_model + 0.2 + 50],
    #                        wheel_configuration=[0.0, 2.5, 19.9, 22.4],
    #                        uvec_file=r"../uvec_ten_dof_vehicle_2D/uvec.py", uvec_function_name="uvec",
    #                        uvec_parameters=uvec_parameters)

    # add the load on the tracks
    model_stage_1.add_load_on_line_model_part("rail_track_1", uvec_load_1, "train_load1")
    # # add the load on the tracks
    # model_stage_1.add_load_on_line_model_part("rail_track_1", uvec_load_2, "train_load2")
    # add the load on the tracks
    # model_stage_1.add_load_on_line_model_part("rail_track_1", uvec_load_3, "train_load3")

    # ------------------------------------------------------------------------------------------------------------------
    # SUPPORTS, CONSTRAINTS AND ABSORBING BOUNDARIES
    # ------------------------------------------------------------------------------------------------------------------

    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True],
                                                            value=[0, 0, 0])
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

    z_start_vertical_plane = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
    z_end_vertical_plane = [(0, 0, model_length), (0, 1, model_length), (1, 0, model_length)]
    symmetry_plane = [(0, 0, 0), (0, 0, model_length), (0, 1, 0)]
    x_end_vertical_plane = [(model_width, 0, 0), (model_width, 0, model_length), (model_width, 1, 0)]
    y_min = all_soil_coordinates[:, 1].min()
    bottom_plane = [(0, y_min, 0), (1, y_min, 0), (0, y_min, 1)]

    # plane z=0: absorbing
    model_stage_1.add_boundary_condition_on_plane(z_start_vertical_plane, absorbing_boundaries_parameters,
                                                  "absorbing_z_eq_0")
    # plane z=model_length: absorbing
    model_stage_1.add_boundary_condition_on_plane(z_end_vertical_plane, absorbing_boundaries_parameters,
                                                  "absorbing_z_eq_model_length")
    # plane x=model_width: absorbing
    model_stage_1.add_boundary_condition_on_plane(x_end_vertical_plane, absorbing_boundaries_parameters,
                                                  "absorbing_x_eq_model_width")
    # plane x=0: symmetry
    model_stage_1.add_boundary_condition_on_plane(symmetry_plane, roller_displacement_parameters, "symmetry_plane")
    # plane x=0: symmetry
    model_stage_1.add_boundary_condition_on_plane(bottom_plane, no_displacement_parameters, "fixed_base")

    # Set mesh size
    # --------------------------------
    # considering a shear wave velocity of 245 m/s of the first soil layer
    # and 10 elements per wave, to describe up to 100Hz we need a mesh size of approx. 0.25m

    model_stage_1.set_element_size_of_group(0.5, "ballast")
    model_stage_1.set_element_size_of_group(0.5, "soil_layer_1")
    model_stage_1.set_element_size_of_group(1.0, "soil_layer_2")
    # model_stage_1.set_element_size_of_group(0.5, "ballast")
    # model_stage_1.set_mesh_size(element_size=0.25)
    # model_stage_1.generate_mesh(save_file=True, open_gmsh_gui=True)

    # Synchronize geometry
    # model_stage_1.synchronise_geometry()

    # Define project parameters
    # --------------------------------
    # considering a target frequency output of 100Hz, we consider an output frequency of about 500Hz
    # dt = 1/500 = 0.002
    output_frequency = 500  # Hz
    time_step = 1 / output_frequency
    end_time = 3.0

    initial_steps = 10

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC

    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=time_step * initial_steps,
                                       delta_time=time_step,
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
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=1e-03,
                                     rayleigh_m=10.0)

    # Set up problem data
    problem = Problem(problem_name="validation_case", number_of_threads=4, settings=solver_settings)
    model_stage_1.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

    # Define the output process
    model_stage_1.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                            output_interval=1,
                                                                            nodal_results=nodal_results,
                                                                            gauss_point_results=[],
                                                                            output_control_type="step"),
                                      output_dir="output",
                                      output_name="vtk_output")

    # Define the output process
    model_stage_1.add_output_settings_by_coordinates(output_parameters=JsonOutputParameters(
        output_interval=time_step - 1e-08,
        nodal_results=nodal_results,
        gauss_point_results=[],
    ),
                                                     coordinates=[(model_width, 0, 0),
                                                                  (model_width, 0, model_length / 2),
                                                                  (model_width, 0, model_length)],
                                                     part_name="line_output",
                                                     output_dir="output",
                                                     output_name="json_output_line")

    # define the STEM instance
    input_folder = "inputs_kratos"
    stem = Stem(model_stage_1, input_folder)

    # create new stage
    model_stage_2 = stem.create_new_stage(delta_time=time_step, stage_duration=end_time)

    # Set up solver settings for the new stage
    model_stage_2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    model_stage_2.project_parameters.settings.rayleigh_k = 5.787452476068922e-05
    model_stage_2.project_parameters.settings.rayleigh_m = 0.5711986642890533

    # add the new stage to the calculation
    stem.add_calculation_stage(model_stage_2)

    # write the kratos input files
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()


if __name__ == "__main__":
    run_model()
