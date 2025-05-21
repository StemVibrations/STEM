import UVEC.uvec_ten_dof_vehicle_2D as uvec
from stem.model import Model
#from stem.additional_processes import HingeParameters
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

# Ballast bed
solid_density_ballast = 2100
porosity_ballast = 0.3
young_modulus_ballast = 125e6
poisson_ratio_ballast = 0.25
soil_formulation_ballast = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_ballast, POROSITY=porosity_ballast)
constitutive_law_ballast = LinearElasticSoil(YOUNG_MODULUS=young_modulus_ballast, POISSON_RATIO=poisson_ratio_ballast)
retention_parameters_ballast = SaturatedBelowPhreaticLevelLaw()
material_ballast = SoilMaterial("baanlichaam", soil_formulation_ballast, constitutive_law_ballast, retention_parameters_ballast)

# Laag 1
solid_density_laag_1 = 2100
porosity_laag_1 = 0.3
young_modulus_laag_1 = 125e6
poisson_ratio_laag_1 = 0.25
soil_formulation_laag_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_laag_1, POROSITY=porosity_laag_1)
constitutive_law_laag_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_laag_1, POISSON_RATIO=poisson_ratio_laag_1)
retention_parameters_laag_1 = SaturatedBelowPhreaticLevelLaw()
material_laag_1 = SoilMaterial("baanlichaam", soil_formulation_laag_1, constitutive_law_laag_1, retention_parameters_laag_1)

# Laag 2
solid_density_laag_2 = 2100
porosity_laag_2 = 0.3
young_modulus_laag_2 = 125e6
poisson_ratio_laag_2 = 0.25
soil_formulation_laag_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_laag_2, POROSITY=porosity_laag_2)
constitutive_law_laag_2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_laag_2, POISSON_RATIO=poisson_ratio_laag_2)
retention_parameters_laag_2 = SaturatedBelowPhreaticLevelLaw()
material_laag_2 = SoilMaterial("baanlichaam", soil_formulation_laag_2, constitutive_law_laag_2, retention_parameters_laag_2)

# Laag 3
solid_density_laag_3 = 2100
porosity_laag_3 = 0.3
young_modulus_laag_3 = 125e6
poisson_ratio_laag_3 = 0.25
soil_formulation_laag_3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_laag_3, POROSITY=porosity_laag_3)
constitutive_law_laag_3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_laag_3, POISSON_RATIO=poisson_ratio_laag_3)
retention_parameters_laag_3 = SaturatedBelowPhreaticLevelLaw()
material_laag_3 = SoilMaterial("baanlichaam", soil_formulation_laag_3, constitutive_law_laag_3, retention_parameters_laag_3)

# Laag 4
solid_density_laag_4 = 2100
porosity_laag_4 = 0.3
young_modulus_laag_4 = 125e6
poisson_ratio_laag_4 = 0.25
soil_formulation_laag_4 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_laag_4, POROSITY=porosity_laag_4)
constitutive_law_laag_4 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_laag_4, POISSON_RATIO=poisson_ratio_laag_4)
retention_parameters_laag_4 = SaturatedBelowPhreaticLevelLaw()
material_laag_4 = SoilMaterial("baanlichaam", soil_formulation_laag_4, constitutive_law_laag_4, retention_parameters_laag_4)

# Laag 5
solid_density_laag_5 = 2100
porosity_laag_5 = 0.3
young_modulus_laag_5 = 125e6
poisson_ratio_laag_5 = 0.25
soil_formulation_laag_5 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_laag_5, POROSITY=porosity_laag_5)
constitutive_law_laag_5= LinearElasticSoil(YOUNG_MODULUS=young_modulus_laag_5, POISSON_RATIO=poisson_ratio_laag_5)
retention_parameters_laag_5 = SaturatedBelowPhreaticLevelLaw()
material_laag_5 = SoilMaterial("baanlichaam", soil_formulation_laag_5, constitutive_law_laag_5, retention_parameters_laag_5)

rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                                          NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                          NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0], # damping coefficient [Ns/m]
                                          NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                       NODAL_MASS=140,
                                       NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

extrusion_length = 50 # Hoe lang de track is waarover de trein gaat

# coordinaten grondlagen
ballast_coordinates = [(0.0, 14.77, 0.0), (10.0, 14.77, 0.0), (10.0, 15.07, 0.0), (0.75, 15.07, 0.0), (0, 15.07, 0.0)]
laag_1_coordinates = [(0.0, 11.77, 0.0), (10.0, 11.77, 0.0), (10.0, 14.77, 0.0), (0.0, 14.77, 0.0)]
laag_2_coordinates = [(0.0, 9.77, 0.0), (10.0, 9.77, 0.0), (10.0, 11.77, 0.0), (0.0, 11.77, 0.0)]
laag_3_coordinates = [(0.0, 7.27, 0.0), (10.0, 7.27, 0.0), (10.0, 9.77, 0.0), (0.0, 9.77, 0.0)]
laag_4_coordinates = [(0.0, 7.27, 0.0), (10.0, 7.27, 0.0), (10.0, 9.77, 0.0), (0.0, 9.77, 0.0)]


soil_onderste_coordinates = [(0.0, -7.0, 0.0), (10.0, -7.0, 0.0), (10.0, -2.0, 0.0), (0.0, -2.0, 0.0)] #onderste laag
soil_complexsat_coordinates = [(0.0, -2.0, 0.0), (10.0, -2.0, 0.0), (10.0, 8.0, 0.0), (0.0, 8.0, 0.0)] #2 laag
soil_complexunsat_coordinates = [(0.0, 8.0, 0.0), (10.0, 8.0, 0.0), (10.0, 18.0, 0.0), (0.0, 18.0, 0.0)] #3 laag
soil_boxtel_coordinates = [(0.0, 18.0, 0.0), (10.0, 18.0, 0.0), (10.0, 19.4, 0.0), (0.0, 19.4, 0.0)] #middelste laag
baanlichaam_coordinates = [(0.0, 19.4, 0.0), (10.0, 19.4, 0.0), (10.0, 20.0, 0.0), (0.75, 20.0, 0.0), (0, 20.0, 0.0)] #bovenste laag
model.extrusion_length = extrusion_length

model.add_soil_layer_by_coordinates(soil_onderste_coordinates, material_soil_onderste, "soil_layer_onderste")
model.add_soil_layer_by_coordinates(soil_complexsat_coordinates, material_soil_complexsat, "soil_layer_complexsat")
model.add_soil_layer_by_coordinates(soil_complexunsat_coordinates, material_soil_complexunsat, "soil_layer_complexunsat")
model.add_soil_layer_by_coordinates(soil_boxtel_coordinates, material_soil_boxtel, "soil_layer_boxtel")
model.add_soil_layer_by_coordinates(baanlichaam_coordinates, material_baanlichaam, "baanlichaam_layer")

origin_point = [0.75, 20.0, 0] # [x, y = hoogte, z]
direction_vector = [0, 0, 1]
number_of_sleepers = 101
sleeper_spacing = 0.5
rail_pad_thickness = 0.025

model.generate_straight_track(sleeper_spacing, number_of_sleepers, rail_parameters,
                              sleeper_parameters, rail_pad_parameters,
                              rail_pad_thickness, origin_point,
                              direction_vector, "rail_track")

wheel_configuration=[0.0, 2.5, 19.9, 22.4] # wheel configuration [m]
velocity = 0 # velocity of the UVEC [m/s]
# treinsnelheid / 1/2 (bij ovale vorm) omtrek = frequentie
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
uvec_load = UvecLoad(direction=[1, 1, 1], velocity=velocity, origin=[0.75, 20+rail_pad_thickness, 0], #die 20 is ook y directie
                     wheel_configuration=wheel_configuration,
                     uvec_model=uvec,
                     uvec_parameters=uvec_parameters)

# add the load on the tracks
model.add_load_on_line_model_part("rail_track", uvec_load, "train_load")

# show model
model.show_geometry(show_surface_ids=True)

# define BC
no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [4, 9, 14, 19, 25], roller_displacement_parameters, "sides_roller")
model.add_boundary_condition_by_geometry_ids(2, [2, 5, 6, 7, 10, 11, 12, 15, 17, 20, 21, 22, 25, 26, 27], absorbing_boundaries_parameters, "abs")

model.set_mesh_size(element_size=1.0)
# 30 Hz
model.set_element_size_of_group(element_size=0.55, group_name="baanlichaam_layer")
model.set_element_size_of_group(element_size=0.32, group_name="soil_layer_boxtel")
model.set_element_size_of_group(element_size=0.44, group_name="soil_layer_complexunsat")
model.set_element_size_of_group(element_size=0.36, group_name="soil_layer_complexsat")
model.set_element_size_of_group(element_size=0.39, group_name="baanlichaam_layer")

end_time = 0.1
delta_time = 5e-3
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

results_dir = "results"
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
# De locaties waar de metingen zijn gedaan (nog aanpassen) (make a sketch were the starting positions are.
desired_output_points = [
                         (0.75, 20, 25),
                         (0.75, 20 + rail_pad_thickness, 25)
                         ]

model.add_output_settings_by_coordinates(
    part_name="subset_outputs",
    output_dir=results_dir,
    output_name="json_output",
    coordinates=desired_output_points,
    output_parameters=JsonOutputParameters(
        output_interval=delta_time,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results
    )
)

input_files_dir = "Holten_rm"
stem = Stem(model, input_files_dir)
delta_time_stage_2 = 1e-3
duration_stage_2 = 0.5
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
stem.run_calculation() # Duration 2.5 hours....

