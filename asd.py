import sys

uvec_module_path = r"benchmark_tests/test_train_uvec_3d"
sys.path.append(uvec_module_path)
#import uvec_ten_dof_vehicle_2D.base_model
#print(uvec_ten_dof_vehicle_2D.base_model.__file__)
#exit()

import os
from shutil import copytree

from stem.model import Model
from stem.soil_material import (
    OnePhaseSoil,
    LinearElasticSoil,
    SoilMaterial,
    SaturatedBelowPhreaticLevelLaw,
)
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.default_materials import DefaultMaterial
from stem.load import  UvecLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.additional_processes import ParameterFieldParameters
from stem.field_generator import RandomFieldGenerator
from stem.solver import (
    AnalysisType,
    SolutionType,
    TimeIntegration,
    DisplacementConvergenceCriteria,
    NewtonRaphsonStrategy,
    NewmarkScheme,
    Amgcl,
    StressInitialisationType,
    SolverSettings,
    Problem,
)
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem

import pandas as pd

input_files_dir = "uvec_train_model"
results_dir = "output_uvec_train_model"


def main(use_metabarrier=False):
    print("Starting the simulation")

    ndim = 3
    model = Model(ndim)
    model.extrusion_length = 30

    prinsebeek_soil_parameters = pd.read_csv("prinsebeek.csv")

    prev_depth = 0.0 if not use_metabarrier else prinsebeek_soil_parameters.iloc[0]["depth"]

    for i, row in prinsebeek_soil_parameters.iterrows():
        # If we implement a metabarrier, the first layer needs different coordinates
        if i > 0 or not use_metabarrier:
            solid_density = row["density"]
            porosity = row["porosity"]
            young_modulus = row["young_modulus"]
            poisson_ratio = row["poissons_ratio"]

            material_soil = create_material_soil(ndim, solid_density, porosity, young_modulus, poisson_ratio)

            add_soil_layer_at_depth(model, prev_depth, row["depth"], material_soil, f"soil_{i}")
            prev_depth = row["depth"]

    concrete_material_soil = create_material_soil(3, 2350, 0.095, 40e9, 0.15)

    # RAILS
    rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
    rail_pad_parameters = ElasticSpringDamper(
        NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
        NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
        NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],  # damping coefficient [Ns/m]
        NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0],
    )
    sleeper_parameters = NodalConcentrated(
        NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
        NODAL_MASS=140,
        NODAL_DAMPING_COEFFICIENT=[0, 0, 0],
    )

    if use_metabarrier:
        top_soil_params = prinsebeek_soil_parameters.iloc[0]
        solid_density = top_soil_params["density"]
        porosity = top_soil_params["porosity"]
        young_modulus = top_soil_params["young_modulus"]
        poisson_ratio = top_soil_params["poissons_ratio"]

        top_material_soil = create_material_soil(ndim, solid_density, porosity, young_modulus, poisson_ratio)

        top_soil_coordinates = [
            (-5.0, -top_soil_params["depth"], 0.0),
            (40.0, -top_soil_params["depth"], 0.0),
            (40.0, 0.0, 0.0),
            (2.98, 0.0, 0.0),
            (2.98, -1.0, 0.0),
            (2.5, -1.0, 0.0),
            (2.5, 0.0, 0.0),
            (-5.0, 0.0, 0.0),
        ]
        model.add_soil_layer_by_coordinates(
            top_soil_coordinates, top_material_soil, "soil_0"
        )

        metabarrier_coordinates = [
            (2.5, 0.0, 0.0),
            (2.5, -1.0, 0.0),
            (2.59, -1.0, 0.0),
            (2.61, -0.19, 0.0),
            (2.87, -0.19, 0.0),
            (2.89, -1.0, 0.0),
            (2.98, -1.0, 0.0),
            (2.98, 0.0, 0.0),
        ]

        model.add_soil_layer_by_coordinates(
            metabarrier_coordinates, concrete_material_soil, "metabarrier"
        )

    embankment_coordinates = [
        (-5.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (1.5, 0.5, 0.0),
        (-5, 0.5, 0.0),
    ]

    # Embankment soil
    solid_density_3 = 2650
    porosity_3 = 0.3
    young_modulus_3 = 10e6
    poisson_ratio_3 = 0.2
    soil_formulation_3 = OnePhaseSoil(
        ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity_3
    )
    constitutive_law_3 = LinearElasticSoil(
        YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio_3
    )
    retention_parameters_3 = SaturatedBelowPhreaticLevelLaw()
    material_embankment = SoilMaterial(
        "embankment", soil_formulation_3, constitutive_law_3, retention_parameters_3
    )

    model.add_soil_layer_by_coordinates(
        embankment_coordinates, material_embankment, "embankment_layer"
    )

    # Add rail materials to the model
    origin_point = [0.0, 0.5, 0.0]
    direction_vector = [0, 0, 1]
    number_of_sleepers = 101
    sleeper_spacing = 0.5
    rail_pad_thickness = 0.025

    model.generate_straight_track(
        sleeper_spacing,
        number_of_sleepers,
        rail_parameters,
        sleeper_parameters,
        rail_pad_parameters,
        rail_pad_thickness,
        origin_point,
        direction_vector,
        "rail_track_1",
    )

    # Add train load
    # the name of the uvec module
    uvec_folder = os.path.join(uvec_module_path, "uvec_ten_dof_vehicle_2D")
    # create input files directory, since it might not have been created yet
    os.makedirs(input_files_dir, exist_ok=True)
    # copy uvec module to input files directory
    copytree(uvec_folder, os.path.join(input_files_dir, "uvec_ten_dof_vehicle_2D"), dirs_exist_ok=True)

    # define uvec parameters
    uvec_load = create_uvec_load(rail_pad_thickness)

    # add the load on the tracks
    model.add_load_on_line_model_part("rail_track_1", uvec_load, "train_load")

    # TODO: Hebben we random field generators nodig?
    # random_field_generator = RandomFieldGenerator(
    #     n_dim=3, cov=0.1, v_scale_fluctuation=1,
    #     anisotropy=[20.0], angle=[0],
    #     model_name="Gaussian", seed=14
    # )

    # field_parameters_json = ParameterFieldParameters(
    #     property_name="YOUNG_MODULUS",
    #     function_type="json_file",
    #     field_generator=random_field_generator
    # )
    # add the random field to the model
    # model.add_field(part_name="soil_layer_2", field_parameters=field_parameters_json)

    model.show_geometry(show_surface_ids=True)

    # Add border constraints
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

    if use_metabarrier:
        model.add_boundary_condition_by_geometry_ids(2, [42], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(2, [8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 44, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 67, 68, 69, 74, 75, 76, 79, 80, 81], absorbing_boundaries_parameters, "abs")
    else:
        model.add_boundary_condition_by_geometry_ids(2, [47], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(2, [8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 51, 52, 55, 56, 57, 60, 61, 62], absorbing_boundaries_parameters, "abs")

    # Set the size of the mesh to be generated
    model.set_mesh_size(element_size=1.0)

    delta_time = 1e-02 # 100Hz

    solver_settings = create_solver_settings(delta_time)

    # Set up problem data
    problem = Problem(
        problem_name="calculate_uvec_on_embankment_with_absorbing_boundaries",
        number_of_threads=4,
        settings=solver_settings,
    )
    model.project_parameters = problem

    # Output parameters
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
    gauss_point_results = []

    model.add_output_settings(
        part_name="porous_computational_model_part",
        output_dir=results_dir,
        output_name="vtk_output",
        output_parameters=VtkOutputParameters(
            file_format="ascii",
            output_interval=1,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
            output_control_type="step"
        )
    )

    desired_output_points = [
        (8.0, 0.0, 25.0),
        (16, 0.0, 25.0),
        (24, 0.0, 25.0),
        (32, 0.0, 25.0),
    ]

    model.add_output_settings_by_coordinates(
        part_name="subset_outputs",
        output_dir=results_dir,
        output_name="json_output",
        coordinates=desired_output_points,
        output_parameters=JsonOutputParameters(
            output_interval=delta_time - 1e-10,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
        ),
    )

    model.show_geometry()

    stem = Stem(model, input_files_dir)

    stem.write_all_input_files()

    stem.run_calculation()

def create_material_soil(ndim, solid_density, porosity, young_modulus, poisson_ratio):
    soil_formulation = OnePhaseSoil(
            ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density, POROSITY=porosity
        )
    constitutive_law = LinearElasticSoil(
            YOUNG_MODULUS=young_modulus, POISSON_RATIO=poisson_ratio
        )
    retention_parameters = SaturatedBelowPhreaticLevelLaw()
    material_soil = SoilMaterial(
            "soil_1", soil_formulation, constitutive_law, retention_parameters
        )

    return material_soil

def add_soil_layer_at_depth(model, depth_top, depth_bottom, material_soil, name):
    soil_coordinates = [
        (-5.0, -depth_bottom, 0.0),
        (40.0, -depth_bottom, 0.0),
        (40.0, -depth_top, 0.0),
        (-5.0, -depth_top, 0.0),
    ]

    model.add_soil_layer_by_coordinates(
        soil_coordinates, material_soil, name
    )

def create_uvec_load(rail_pad_thickness):
    uvec_parameters = {"n_carts": 1, # number of carts [-]
                    "cart_inertia": (1128.8e3) / 2, # inertia of the cart [kgm2]
                    "cart_mass": (50e3) / 2, # mass of the cart [kg]
                    "cart_stiffness": 2708e3, # stiffness between the cart and bogies [N/m]
                    "cart_damping": 64e3, # damping coefficient between the cart and bogies [Ns/m]
                    "bogie_distances": [-9.95, 9.95], # distances of the bogies from the centre of the cart [m]
                    "bogie_inertia": (0.31e3) / 2, # inertia of the bogie [kgm2]
                    "bogie_mass": (6e3) / 2, # mass of the bogie [kg]
                    "wheel_distances": [-1.25, 1.25], # distances of the wheels from the centre of the bogie [m]
                    "wheel_mass": 1.5e3, # mass of the wheel [kg]
                    "wheel_stiffness": 4800e3, # stiffness between the wheel and the bogie [N/m]
                    "wheel_damping": 0.25e3, # damping coefficient between the wheel and the bogie [Ns/m]
                    "gravity_axis": 1, # axis on which gravity works [x =0, y = 1, z = 2]
                    "contact_coefficient": 9.1e-7, # Hertzian contact coefficient between the wheel and the rail [N/m]
                    "contact_power": 1.0, # Hertzian contact power between the wheel and the rail [-]
                    "initialisation_steps": 20, # number of time steps on which the gravity on the UVEC is
                                                    # gradually increased [-]
                    }

    # define the UVEC load
    uvec_load = UvecLoad(direction=[1, 1, 1], velocity=40, origin=[0.0, 0.5+rail_pad_thickness, 5],
                        wheel_configuration=[0.0, 2.5, 19.9, 22.4],
                        uvec_file=r"uvec_ten_dof_vehicle_2D/uvec.py", uvec_function_name="uvec",
                        uvec_parameters=uvec_parameters)

    return uvec_load

def create_solver_settings(delta_time):
    # Simulation parameters
    end_time = 0.2
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC

    time_integration = TimeIntegration(
        start_time=0.0,
        end_time=end_time,
        delta_time=delta_time,
        reduction_factor=1,
        increase_factor=1,
        max_delta_time_factor=1000,
    )

    convergence_criterion = DisplacementConvergenceCriteria(
        displacement_relative_tolerance=1.0e-4, displacement_absolute_tolerance=1.0e-12
    )

    strategy_type = NewtonRaphsonStrategy()
    scheme_type = NewmarkScheme()
    linear_solver_settings = Amgcl()
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(
        analysis_type=analysis_type,
        solution_type=solution_type,
        stress_initialisation_type=stress_initialisation_type,
        time_integration=time_integration,
        is_stiffness_matrix_constant=True,
        are_mass_and_damping_constant=True,
        convergence_criteria=convergence_criterion,
        strategy_type=strategy_type,
        scheme=scheme_type,
        linear_solver_settings=linear_solver_settings,
        rayleigh_k=0.12,
        rayleigh_m=0.0001,
    )

    return solver_settings


if __name__ == "__main__":
    main(True)
