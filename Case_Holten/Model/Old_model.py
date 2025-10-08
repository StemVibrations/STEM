import UVEC.uvec_ten_dof_vehicle_2D as uvec
from stem.model import Model
# from stem.additional_processes import HingeParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.default_materials import DefaultMaterial
from stem.load import UvecLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
    LinearNewtonRaphsonStrategy, NewmarkScheme, Cg, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem

ndim = 3
model = Model(ndim)

# ground
solid_density_l5 = 1850
porosity_l5 = 0.25
young_modulus_l5 = 80e6
poisson_ratio_l5 = 0.25
soil_formulation_l5 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_l5, POROSITY=porosity_l5)
constitutive_law_l5 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_l5, POISSON_RATIO=poisson_ratio_l5)
retention_parameters_l5 = SaturatedBelowPhreaticLevelLaw()
material_soil_l5 = SoilMaterial("soil_l5", soil_formulation_l5, constitutive_law_l5, retention_parameters_l5)

solid_density_l4 = 1850
porosity_l4 = 0.25
young_modulus_l4 = 80e6
poisson_ratio_l4 = 0.25
soil_formulation_l4 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_l4, POROSITY=porosity_l4)
constitutive_law_l4 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_l4, POISSON_RATIO=poisson_ratio_l4)
retention_parameters_l4 = SaturatedBelowPhreaticLevelLaw()
material_soil_l4 = SoilMaterial("soil_l4", soil_formulation_l4, constitutive_law_l4, retention_parameters_l4)

solid_density_l3 = 2050
porosity_l3 = 0.25
young_modulus_l3 = 150e6
poisson_ratio_l3 = 0.25
soil_formulation_l3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_l3, POROSITY=porosity_l3)
constitutive_law_l3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_l3, POISSON_RATIO=poisson_ratio_l3)
retention_parameters_l3 = SaturatedBelowPhreaticLevelLaw()
material_soil_l3 = SoilMaterial("soil_l3", soil_formulation_l3, constitutive_law_l3, retention_parameters_l3)

solid_density_l2 = 1900
porosity_l2 = 0.25
young_modulus_l2 = 100e6
poisson_ratio_l2 = 0.25
soil_formulation_l2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_l2, POROSITY=porosity_l2)
constitutive_law_l2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_l2, POISSON_RATIO=poisson_ratio_l2)
retention_parameters_l2 = SaturatedBelowPhreaticLevelLaw()
material_soil_l2 = SoilMaterial("soil_l2", soil_formulation_l2, constitutive_law_l2, retention_parameters_l2)

solid_density_l1 = 2100
porosity_l1 = 0.25
young_modulus_l1 = 125e6
poisson_ratio_l1 = 0.25
soil_formulation_l1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_l1, POROSITY=porosity_l1)
constitutive_law_l1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_l1, POISSON_RATIO=poisson_ratio_l1)
retention_parameters_l1 = SaturatedBelowPhreaticLevelLaw()
material_soil_l1 = SoilMaterial("soil_l1", soil_formulation_l1, constitutive_law_l1, retention_parameters_l1)

# ballast van Deltares
solid_density_ballast = 1850
porosity_ballast = 0.3
young_modulus_ballast = 150e6
poisson_ratio_ballast = 0.20
soil_formulation_ballast = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_ballast, POROSITY=porosity_ballast)
constitutive_law_ballast = LinearElasticSoil(YOUNG_MODULUS=young_modulus_ballast, POISSON_RATIO=poisson_ratio_ballast)
retention_parameters_ballast = SaturatedBelowPhreaticLevelLaw()
material_ballast = SoilMaterial("ballast", soil_formulation_ballast, constitutive_law_ballast, retention_parameters_ballast)

# solid_density_ballast = 1800
# porosity_ballast = 0.3
# young_modulus_ballast = 67.5e6
# poisson_ratio_ballast = 0.20
# soil_formulation_ballast = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_ballast, POROSITY=porosity_ballast)
# constitutive_law_ballast = LinearElasticSoil(YOUNG_MODULUS=young_modulus_ballast, POISSON_RATIO=poisson_ratio_ballast)
# retention_parameters_ballast = SaturatedBelowPhreaticLevelLaw()
# material_ballast = SoilMaterial("ballast", soil_formulation_ballast, constitutive_law_ballast, retention_parameters_ballast)

rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                                          NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                          NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0], # damping coefficient [Ns/m]
                                          NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                       NODAL_MASS=140,
                                       NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

extrusion_length = 72
soil_l5_coordinates = [(0.0, 3.77, 0.0), (10.0, 3.77, 0.0), (10.0, 5.77, 0.0), (0.0, 5.77, 0.0)]
soil_l4_coordinates = [(0.0, 5.77, 0.0), (10.0, 5.77, 0.0), (10.0, 7.27, 0.0), (0.0, 7.27, 0.0)]
soil_l3_coordinates = [(0.0, 7.27, 0.0), (10.0, 7.27, 0.0), (10.0, 9.77, 0.0), (0.0, 9.77, 0.0)]
soil_l2_coordinates = [(0.0, 9.77, 0.0), (10.0, 9.77, 0.0), (10.0, 11.77, 0.0), (0.0, 11.77, 0.0)]
soil_l1_coordinates = [(0.0, 11.77, 0.0), (10.0, 11.77, 0.0), (10.0, 14.77, 0.0), (0.0, 14.77, 0.0)]
# subballast_coordinates = [(0.0, 14.77, 0.0), (10.0, 14.77, 0.0), (10.0, 15.77, 0.0), (0.0, 15.77, 0.0)]
ballast_coordinates = [(0.0, 14.77, 0.0), (2.5, 14.77, 0.0), (1.5, 15.07, 0.0), (0.75, 15.07, 0.0), (0, 15.07, 0.0)]
model.extrusion_length = extrusion_length

model.add_soil_layer_by_coordinates(soil_l5_coordinates, material_soil_l5, "soil_layer_5")
model.add_soil_layer_by_coordinates(soil_l4_coordinates, material_soil_l4, "soil_layer_4")
model.add_soil_layer_by_coordinates(soil_l3_coordinates, material_soil_l3, "soil_layer_3")
model.add_soil_layer_by_coordinates(soil_l2_coordinates, material_soil_l2, "soil_layer_2")
model.add_soil_layer_by_coordinates(soil_l1_coordinates, material_soil_l1, "soil_layer_1")
# model.add_soil_layer_by_coordinates(subballast_coordinates, material_subballast, "subballast_layer")
model.add_soil_layer_by_coordinates(ballast_coordinates, material_ballast, "ballast_layer")

# create track with extension outside the 3D soil domain
rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                                          NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                          NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],
                                          NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                       NODAL_MASS=140,
                                       NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

soil_equivalent_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 110e6, 0],
                                                 NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                                 NODAL_DAMPING_COEFFICIENT=[0, 110e3, 0],
                                                 NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

# rails
origin_point = [0.75, 15.07, 0]
direction_vector = [0, 0, 1]
number_of_sleepers = 241
sleeper_spacing = 0.6
rail_pad_thickness = 0.025

# create a straight track with rails, sleepers, rail pads and a 1D soil extension
model.generate_extended_straight_track(sleeper_distance=sleeper_spacing,
                                       n_sleepers=number_of_sleepers,
                                       rail_parameters=rail_parameters,
                                       sleeper_parameters=sleeper_parameters,
                                       rail_pad_parameters=rail_pad_parameters,
                                       rail_pad_thickness=rail_pad_thickness,
                                       origin_point=[0.75, 15.07, -36],
                                       soil_equivalent_parameters=soil_equivalent_parameters,
                                       length_soil_equivalent_element=11.3,
                                       direction_vector=[0, 0, 1],
                                       name="rail_track")

# model.generate_straight_track(sleeper_spacing, number_of_sleepers, rail_parameters,
#                               sleeper_parameters, rail_pad_parameters,
#                               rail_pad_thickness, origin_point,
#                               direction_vector, "rail_track")

# train
# define uvec parameters
wheel_configuration=[0.0, 2.5, 19.9, 22.4] # wheel configuration [m]
velocity = 0 # velocity of the UVEC [m/s]
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
                   "static_initialisation": True, # True if the analysis of the UVEC is static
                   "wheel_configuration": wheel_configuration,
                   "velocity": velocity,
                   }

# define the UVEC load
uvec_load = UvecLoad(direction=[1, 1, 1], velocity=velocity, origin=[0.75, 15.07+rail_pad_thickness, 1],
                     wheel_configuration=wheel_configuration,
                     uvec_model=uvec,
                     uvec_parameters=uvec_parameters)

# add the load on the tracks
model.add_load_on_line_model_part("rail_track", uvec_load, "train_load")

# model.show_geometry(show_surface_ids=True)
model.show_geometry()

# boundary conditions
no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [4, 9, 14, 19, 24, 30], roller_displacement_parameters, "sides_roller")
model.add_boundary_condition_by_geometry_ids(2, [2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27, 31, 32], absorbing_boundaries_parameters, "abs")

# mesh voor max 100 Hz
model.set_mesh_size(element_size=1.0)
model.set_element_size_of_group(element_size=0.34, group_name="soil_layer_5")
model.set_element_size_of_group(element_size=0.25, group_name="soil_layer_4")
model.set_element_size_of_group(element_size=0.34, group_name="soil_layer_3")
model.set_element_size_of_group(element_size=0.28, group_name="soil_layer_2")
model.set_element_size_of_group(element_size=0.34, group_name="soil_layer_1")
# ballast van Deltares
model.set_element_size_of_group(element_size=0.30, group_name="ballast_layer")

# settings run
end_time = 1e-1
delta_time = 1e-2
analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.QUASI_STATIC

time_integration = TimeIntegration(start_time=0.0, end_time=end_time, delta_time=delta_time,
                                   reduction_factor=1, increase_factor=1, max_delta_time_factor=1000)

convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                    displacement_absolute_tolerance=1.0e-12)

strategy_type = LinearNewtonRaphsonStrategy()
scheme_type = NewmarkScheme()
linear_solver_settings = Cg()
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                stress_initialisation_type=stress_initialisation_type,
                                time_integration=time_integration,
                                is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                convergence_criteria=convergence_criterion,
                                strategy_type=strategy_type, scheme=scheme_type,
                                linear_solver_settings=linear_solver_settings, rayleigh_k=0,
                                rayleigh_m=0)

problem = Problem(problem_name="HRM_30", number_of_threads=8,
                  settings=solver_settings)
model.project_parameters = problem
nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
gauss_point_results = []

results_dir = "results" # reduceert
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
                         (2.50, 14.77, 36),
                         (4.00, 14.77, 36),
                         (8.00, 14.77, 36)
                         ]

model.add_output_settings_by_coordinates( # fijne stappen
    part_name="subset_outputs",
    output_dir=results_dir,
    output_name="json_output",
    coordinates=desired_output_points,
    output_parameters=JsonOutputParameters(
        output_interval=1e-3,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results
    )
)

input_files_dir = "Holten_rm_test2"
stem = Stem(model, input_files_dir)

delta_time_stage_2 = 1e-3
duration_stage_2 = 1
stage2 = stem.create_new_stage(delta_time_stage_2, duration_stage_2)

velocity = 40
stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
stage2.project_parameters.settings.strategy_type = LinearNewtonRaphsonStrategy()
stage2.project_parameters.settings.rayleigh_k = 0.0002
stage2.project_parameters.settings.rayleigh_m = 0.6
stage2.get_model_part_by_name("train_load").parameters.velocity = velocity
stage2.get_model_part_by_name("train_load").parameters.uvec_parameters["velocity"] = velocity
stage2.get_model_part_by_name("train_load").parameters.uvec_parameters["static_initialisation"] = False

stem.add_calculation_stage(stage2)
stem.write_all_input_files()
stem.run_calculation()
