input_files_dir = "stiff_files_dir"

results_dir = "output"

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.default_materials import DefaultMaterial
from stem.load import MovingLoad, UvecLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg, Amgcl
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem

import UVEC.uvec_ten_dof_vehicle_2D as uvec

ndim = 3
model = Model(ndim)
model.extrusion_length = 90

bottom_coordinate = -10.0
max_x_coordinate = 50.0


# Define all parameters for all materials
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850, POROSITY=0.0)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=150e6, POISSON_RATIO=0.2)
material_ballast = SoilMaterial("ballast", soil_formulation_1, constitutive_law_1, SaturatedBelowPhreaticLevelLaw())


# # fill in PPS parameters
# soil_formulation_pps = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=None, POROSITY=None)
# constitutive_law_pps = LinearElasticSoil(YOUNG_MODULUS=None, POISSON_RATIO=None)
# material_pps = SoilMaterial("pps", soil_formulation_pps, constitutive_law_pps, SaturatedBelowPhreaticLevelLaw())

# set the undrained Poisson ratio to almost 0.5 in order to prevent volumetric strains and simulate undrained behaviour
poisson_ratio_undrained = 0.495

soil_formulation_sand = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1900, POROSITY=0.0)
constitutive_law_sand = LinearElasticSoil(YOUNG_MODULUS=200e6, POISSON_RATIO=poisson_ratio_undrained)
material_sand = SoilMaterial("sand", soil_formulation_sand, constitutive_law_sand, SaturatedBelowPhreaticLevelLaw())


# fill in concrete parameters
# soil_formulation_concrete = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2350, POROSITY=0.0)
# constitutive_law_concrete = LinearElasticSoil(YOUNG_MODULUS=25e9, POISSON_RATIO=0.2)
# material_concrete = SoilMaterial("concrete", soil_formulation_concrete, constitutive_law_concrete, SaturatedBelowPhreaticLevelLaw())

# fill in styrofoam parameters
soil_formulation_styrofoam = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=32, POROSITY=0.0)
constitutive_law_styrofoam = LinearElasticSoil(YOUNG_MODULUS=5e6, POISSON_RATIO=poisson_ratio_undrained)
material_styrofoam = SoilMaterial("styrofoam", soil_formulation_styrofoam, constitutive_law_styrofoam, SaturatedBelowPhreaticLevelLaw())


thickness_ballast = 0.4
surface_level = 0.7

top_ballast = surface_level + thickness_ballast

ballast_coordinates = [(0.0, top_ballast, 0.0), (1.5, top_ballast, 0.0),
                       (2.5, surface_level, 0.0), (0.0, surface_level, 0.0)]


thickness_foundation_top = 0.5
foundation_coordinates_top = [(0.0, surface_level, 0.0), (2.5, surface_level, 0.0),
                              (2.5, surface_level-thickness_foundation_top, 0.0),
                              (0.0, surface_level-thickness_foundation_top, 0.0)]

thickness_foundation_bottom = 0.5
foundation_coordinates_bottom = [(0.0, surface_level-thickness_foundation_top, 0.0),
                                 (2.5, surface_level-thickness_foundation_top, 0.0),
                                 (2.5, surface_level-thickness_foundation_top-thickness_foundation_bottom, 0.0),
                                 (0.0, surface_level-thickness_foundation_top-thickness_foundation_bottom, 0.0)]

x_coord_deep_wall = 12.5
thickness_deep_wall = 1.0
layer_1_boundary_depth = -5.0
layer_2_boundary_depth = -6.0
length_deep_wall = 7.7


deep_wall_part_coordinates = [(x_coord_deep_wall, surface_level, 0.0), (x_coord_deep_wall, -length_deep_wall, 0.0),
                              (x_coord_deep_wall+thickness_deep_wall, -length_deep_wall, 0.0),
                              (x_coord_deep_wall+thickness_deep_wall, surface_level, 0.0)]




soil_1_coordinates = [(0.0, surface_level-thickness_foundation_bottom-thickness_foundation_top, 0.0),
                             (2.5, surface_level-thickness_foundation_bottom-thickness_foundation_top, 0.0),
                             (2.5, surface_level-thickness_foundation_top, 0.0),
                             (2.5, surface_level, 0.0)] + deep_wall_part_coordinates +  [(max_x_coordinate, surface_level, 0.0),
                             (max_x_coordinate,bottom_coordinate,0.0), (0.0, bottom_coordinate, 0.0)]


model.add_soil_layer_by_coordinates(ballast_coordinates, material_ballast, "ballast")

model.add_soil_layer_by_coordinates(foundation_coordinates_top, material_sand, "foundation_top")
model.add_soil_layer_by_coordinates(foundation_coordinates_bottom, material_sand, "foundation_bot")

model.add_soil_layer_by_coordinates(soil_1_coordinates, material_sand, "soil_layer_1")
model.add_soil_layer_by_coordinates(deep_wall_part_coordinates, material_sand, "deep_wall")


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

sleeper_distance =0.6
n_sleepers = 334
rail_pad_thickness = 0.025

# create a straight track with rails, sleepers, rail pads and a 1D soil extension
model.generate_extended_straight_track(sleeper_distance=sleeper_distance,
                                       n_sleepers=n_sleepers,
                                       rail_parameters=rail_parameters,
                                       sleeper_parameters=sleeper_parameters,
                                       rail_pad_parameters=rail_pad_parameters,
                                       rail_pad_thickness=rail_pad_thickness,
                                       origin_point=[0.7, top_ballast, -45],
                                       soil_equivalent_parameters=soil_equivalent_parameters,
                                       length_soil_equivalent_element=2,
                                       direction_vector=[0, 0, 1],
                                       name="rail_track_1")

# define uvec parameters
wheel_configuration = [0, 2.5, 19.9, 22.4, 26.6, 29.1, 46.5, 49] # distances of the wheels from the origin point [m]
uvec_parameters = {"n_carts": 2, # number of carts [-]
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
                   "wheel_configuration": wheel_configuration, # initial position of the wheels [m]
                   "velocity": 0.0, # velocity of the UVEC [m/s]
                   "irr_parameters": {
                            "Av": 2.095e-05,
                            "seed": 14
                            },
                   }
uvec_load = UvecLoad(direction=[1,1,1], velocity = 0.0, origin=[0.7, top_ballast + rail_pad_thickness, -43],
                     wheel_configuration=wheel_configuration, uvec_model= uvec, uvec_parameters=uvec_parameters)
# uvec_load = MovingLoad(load=[0,-10000,0],direction=[1,1,1], velocity = 0.0, origin=[0.7, top_ballast + rail_pad_thickness, 45])

# add the load on the track
model.add_load_on_line_model_part("rail_track_1", uvec_load, "uvec_load")


# define the boundary conditions
roller_displacement_parameters = DisplacementConstraint(active=[True, False, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])
absorbing_boundaries_parameters_bottom = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=10.0)
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=0.1)


# add the boundary conditions to the model
model.add_boundary_condition_on_plane([(0,bottom_coordinate,0), (0,bottom_coordinate,1), (1,bottom_coordinate,0)],
                                      absorbing_boundaries_parameters_bottom,"bottom_abs")

model.add_boundary_condition_on_plane([(0,0,0), (0,1,0), (0,0,1)],
                                      roller_displacement_parameters, "sides_roller")

model.add_boundary_condition_on_plane([(0,0,0), (1,0,0), (0,1,0)],absorbing_boundaries_parameters,"abs")
model.add_boundary_condition_on_plane([(0,0,model.extrusion_length), (1,0,model.extrusion_length), (0,1,model.extrusion_length)],
                                      absorbing_boundaries_parameters,"abs")
model.add_boundary_condition_on_plane([(max_x_coordinate,0,0), (max_x_coordinate,1,0), (max_x_coordinate,0,1)],
                                      absorbing_boundaries_parameters, "abs")


# aux code to calculate the required element size and time step for proper integration
# import numpy as np
# E = 200e6
# nu = 0.495
#
# M = E *(1 - nu)/(1 + nu)/(1 - 2*nu)
# rho = 1900
# vp = np.sqrt(M/rho)
# f = 100
#
# lambda_ = vp/f
#
# required_el_size = lambda_ / 10
# required_dt = required_el_size/vp

model.set_element_size_of_group(0.3,"ballast")
# model.set_element_size_of_group(0.5, "embankment")
model.set_element_size_of_group(0.75, "foundation_top")
model.set_element_size_of_group(0.75, "foundation_bot")

model.set_element_size_of_group(0.75, "deep_wall")


model.set_mesh_size(element_size=1.3)


# define at which points the json output should be written
delta_time = 0.0005
json_output_parameters = JsonOutputParameters(delta_time-1e-10, [NodalOutput.VELOCITY],[])
model.add_output_settings_by_coordinates([(x_coord_deep_wall+thickness_deep_wall, surface_level, 45.0), (25, surface_level, 45.0), (max_x_coordinate,surface_level,45)],
                                         json_output_parameters, "json_output")

# set time integration parameters
end_time = 0.002
time_integration = TimeIntegration(start_time=0.0, end_time=end_time, delta_time=delta_time,
                                   reduction_factor=1, increase_factor=1, max_delta_time_factor=1000)

# set convergence criteria
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-12)

# set solver settings stage 1
solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                 solution_type=SolutionType.QUASI_STATIC,
                                 stress_initialisation_type=StressInitialisationType.NONE,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False, are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=NewtonRaphsonStrategy(),
                                 linear_solver_settings=Cg(scaling=True))

# Set up problem data
problem = Problem(problem_name="stiff", number_of_threads=16,
                  settings=solver_settings)
model.project_parameters = problem


# define the output settings in vtk format
model.add_output_settings(
    part_name="porous_computational_model_part",
    output_dir=results_dir,
    output_name="vtk_output",
    output_parameters=VtkOutputParameters(
        file_format="binary",
        output_interval=15,
        nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION],
        gauss_point_results=[],
        output_control_type="step"
    )
)

# show the geometry, to check if everything is correct
model.show_geometry()

# create the stem object
stem = Stem(model, input_files_dir)

# create a new stage, and set the differences compared to stage 1
duration_stage_2 = 3.45
stage2 = stem.create_new_stage(delta_time,duration_stage_2)
stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
stage2.project_parameters.settings.linear_solver_settings = Cg(scaling=False)
stage2.project_parameters.settings.is_stiffness_matrix_constant = True
stage2.project_parameters.settings.are_mass_and_damping_constant = True
stage2.project_parameters.settings.strategy_type = LinearNewtonRaphsonStrategy()

# add rayleigh damping parameters
stage2.project_parameters.settings.rayleigh_k = 7.86e-5
stage2.project_parameters.settings.rayleigh_m = 0.248

# change the uvec parameters for the second stage
velocity = 38.9 - 1e-5 # 140 km/h minus a small value to prevent numerical issues (solved in upcoming version)
stage2.get_model_part_by_name("uvec_load").parameters.velocity = velocity
stage2.get_model_part_by_name("uvec_load").parameters.uvec_parameters["velocity"] = velocity
stage2.get_model_part_by_name("uvec_load").parameters.uvec_parameters["static_initialisation"] = False

# increase the virtual thickness of the side absorbing boundary conditions for proper damping
stage2.get_model_part_by_name("abs").parameters.virtual_thickness = 50

# add the new stage to the stem calculation
stem.add_calculation_stage(stage2)

# write the input files
stem.write_all_input_files()

# run the calculation
stem.run_calculation()