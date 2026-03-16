input_files_dir = "sheetpile_with_anchor"
results_dir = "output"

import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated, EulerBeam, Anchor
from stem.default_materials import DefaultMaterial
from stem.load import LineLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg, Amgcl, Lu
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem

from stem.sheetpile_utils import SheetPileUtils

ndim = 2
model = Model(ndim)
model.mesh_settings.element_order = 2

bottom_coordinate = -10.0

sheetpile_x_coordinate = 0
min_x_coordinate = -10
max_x_coordinate = 10.0

surface_level = 5

concrete_cover_thickness = 0.5
concrete_cover_length = 2

top_soil_depth = 5
clay_depth = 7

excavation_depth_left = 7

sheetpile_begin_level = surface_level - 1
sheetpile_length = 12

anchor_end_coordinates = (6, -4, 0)

excavation_depth_right = concrete_cover_length

top_soil_coords_left_part_1 = [(min_x_coordinate, surface_level, 0.0),
                               (sheetpile_x_coordinate - concrete_cover_thickness / 2, surface_level, 0.0),
                               (sheetpile_x_coordinate - concrete_cover_thickness / 2,
                                surface_level - concrete_cover_length, 0.0),
                               (min_x_coordinate, surface_level - concrete_cover_length, 0.0)]

top_soil_coords_left_part_2 = [(sheetpile_x_coordinate - concrete_cover_thickness / 2,
                                surface_level - concrete_cover_length, 0.0),
                               (sheetpile_x_coordinate, surface_level - concrete_cover_length, 0.0),
                               (sheetpile_x_coordinate, surface_level - top_soil_depth, 0.0),
                               (min_x_coordinate, surface_level - top_soil_depth, 0.0),
                               (min_x_coordinate, surface_level - concrete_cover_length, 0.0)]

top_soil_coords_right_part1 = [(sheetpile_x_coordinate + concrete_cover_thickness / 2, surface_level, 0.0),
                               (5, surface_level, 0.0), (3, surface_level - excavation_depth_right, 0.0),
                               (sheetpile_x_coordinate + concrete_cover_thickness / 2,
                                surface_level - excavation_depth_right, 0.0)]

top_soil_coords_right_part2 = [(5, surface_level, 0.0), (max_x_coordinate, surface_level, 0.0),
                               (max_x_coordinate, surface_level - top_soil_depth, 0.0),
                               (sheetpile_x_coordinate, surface_level - top_soil_depth, 0.0),
                               (sheetpile_x_coordinate, surface_level - excavation_depth_right, 0.0),
                               (3, surface_level - excavation_depth_right, 0.0)]

concrete_cover_coordinates = [
    (sheetpile_x_coordinate - concrete_cover_thickness / 2, surface_level, 0.0),
    (sheetpile_x_coordinate + concrete_cover_thickness / 2, surface_level, 0.0),
    (sheetpile_x_coordinate + concrete_cover_thickness / 2, surface_level - concrete_cover_length, 0.0),
    (sheetpile_x_coordinate - concrete_cover_thickness / 2, surface_level - concrete_cover_length, 0.0)
]

# Define all parameters for all materials
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850, POROSITY=0.0)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=150e6, POISSON_RATIO=0.2)
material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, SaturatedBelowPhreaticLevelLaw())

soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2350, POROSITY=0.0)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=25e9, POISSON_RATIO=0.2)
material_concrete = SoilMaterial("concrete", soil_formulation_1, constitutive_law_1, SaturatedBelowPhreaticLevelLaw())

model.add_soil_layer_by_coordinates(top_soil_coords_left_part_1, material_soil_1, "top_soil_left_part_1")
model.add_soil_layer_by_coordinates(top_soil_coords_left_part_2, material_soil_1, "top_soil_left_part_2")
model.add_soil_layer_by_coordinates(top_soil_coords_right_part1, material_soil_1, "top_soil_right_part1")
model.add_soil_layer_by_coordinates(top_soil_coords_right_part2, material_soil_1, "top_soil_right_part2")
model.add_soil_layer_by_coordinates(concrete_cover_coordinates, material_concrete, "concrete_cover")

clay_coordinates_left_part1 = [(min_x_coordinate, surface_level - top_soil_depth, 0.0),
                               (sheetpile_x_coordinate, surface_level - top_soil_depth, 0.0),
                               (sheetpile_x_coordinate, surface_level - excavation_depth_left, 0.0),
                               (min_x_coordinate, surface_level - excavation_depth_left, 0.0)]

clay_coordinates_left_part2 = [(min_x_coordinate, surface_level - excavation_depth_left, 0.0),
                               (sheetpile_x_coordinate, surface_level - excavation_depth_left, 0.0),
                               (sheetpile_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0),
                               (min_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0)]

clay_coordinates_right = [(sheetpile_x_coordinate, surface_level - top_soil_depth, 0.0),
                          (max_x_coordinate, surface_level - top_soil_depth, 0.0),
                          (max_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0),
                          (sheetpile_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0)]

soil_formulation_clay = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850, POROSITY=0.0)
constitutive_law_clay = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=0.2)
material_clay = SoilMaterial("clay", soil_formulation_clay, constitutive_law_clay, SaturatedBelowPhreaticLevelLaw())

model.add_soil_layer_by_coordinates(clay_coordinates_left_part1, material_clay, "clay_left_part1")
model.add_soil_layer_by_coordinates(clay_coordinates_left_part2, material_clay, "clay_left_part2")
model.add_soil_layer_by_coordinates(clay_coordinates_right, material_clay, "clay_right")

deep_sand_coordinates_left = [(min_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0),
                              (sheetpile_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0),
                              (sheetpile_x_coordinate, bottom_coordinate, 0.0),
                              (min_x_coordinate, bottom_coordinate, 0.0)]

deep_sand_coordinates_right = [(sheetpile_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0),
                               (max_x_coordinate, surface_level - top_soil_depth - clay_depth, 0.0),
                               (max_x_coordinate, bottom_coordinate, 0.0),
                               (sheetpile_x_coordinate, bottom_coordinate, 0.0)]

deep_sand_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1900, POROSITY=0.0)
deep_sand_constitutive_law = LinearElasticSoil(YOUNG_MODULUS=200e6, POISSON_RATIO=0.2)
material_deep_sand = SoilMaterial("deep_sand", deep_sand_formulation, deep_sand_constitutive_law,
                                  SaturatedBelowPhreaticLevelLaw())
model.add_soil_layer_by_coordinates(deep_sand_coordinates_left, material_deep_sand, "deep_sand_left")
model.add_soil_layer_by_coordinates(deep_sand_coordinates_right, material_deep_sand, "deep_sand_right")

sheetpile_material = EulerBeam(2, YOUNG_MODULUS=25e9, POISSON_RATIO=0.2, CROSS_AREA=0.02, I33=1e-4, DENSITY=2350)

sheetpile_coordinates = [(sheetpile_x_coordinate, sheetpile_begin_level, 0.0),
                         (sheetpile_x_coordinate, sheetpile_begin_level - sheetpile_length, 0.0)]
SheetPileUtils.add_sheetpile_by_coordinates(coordinates=sheetpile_coordinates,
                                            material_parameters=sheetpile_material,
                                            name="sheetpile",
                                            gmsh_io_instance=model.gmsh_io,
                                            body_model_parts=model.body_model_parts)

anchor_material = Anchor(YOUNG_MODULUS=25e9, CROSS_AREA=0.01)

anchor_coordinates = [(sheetpile_x_coordinate, sheetpile_begin_level, 0.0), anchor_end_coordinates]
SheetPileUtils.add_anchor_by_coordinates(anchor_coordinates, anchor_material, 0.0, "anchor", model.gmsh_io,
                                         model.body_model_parts)

anchor_orrientation_vector = np.array(anchor_coordinates[1]) - np.array(anchor_coordinates[0])
anchor_orrientation_vector = anchor_orrientation_vector / np.linalg.norm(anchor_orrientation_vector)

grout_length = 2

anchor_grout_coordinates = [
    anchor_end_coordinates, anchor_orrientation_vector * grout_length + np.array(anchor_end_coordinates)
]

grout_material = EulerBeam(2, YOUNG_MODULUS=25e9, POISSON_RATIO=0.2, CROSS_AREA=0.01, I33=1e-4, DENSITY=2350)
SheetPileUtils.add_grout_by_coordinates(anchor_grout_coordinates, grout_material, None, "grout", model.gmsh_io,
                                        model.body_model_parts)

point_stiffness = 1e9
point_stiffness_parameters = NodalConcentrated([0, point_stiffness, 0], 0, [0, 0, 0])
point_stiffness_coordinates = [(sheetpile_x_coordinate, sheetpile_begin_level - sheetpile_length, 0.0)]

SheetPileUtils.add_point_element_by_coordinates(point_stiffness_coordinates, point_stiffness_parameters,
                                                "point_stiffness", model.gmsh_io, model.body_model_parts)

line_load = LineLoad(active=[False, False, False], value=[0, -15000, 0])
model.add_load_by_coordinates([(sheetpile_x_coordinate, surface_level, 0.0),
                               (sheetpile_x_coordinate + 2, surface_level, 0.0)], line_load, "line_load")

model.show_geometry()

# add the load on the track
model.add_load_on_line_model_part("rail_track_1", uvec_load, "uvec_load")

# define the boundary conditions
roller_displacement_parameters = DisplacementConstraint(active=[True, False, True],
                                                        is_fixed=[True, False, True],
                                                        value=[0, 0, 0])
absorbing_boundaries_parameters_bottom = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=10.0)
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=0.1)

no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True],
                                                    value=[0, 0, 0])

# add the boundary conditions to the model
# model.add_boundary_condition_on_plane([(0,bottom_coordinate,0), (0,bottom_coordinate,1), (1,bottom_coordinate,0)],
#                                       absorbing_boundaries_parameters_bottom,"bottom_abs")

model.add_boundary_condition_on_plane([(0, bottom_coordinate, 0), (0, bottom_coordinate, 1), (1, bottom_coordinate, 0)],
                                      no_displacement_parameters, "bottom_abs")

model.add_boundary_condition_on_plane([(0, 0, 0), (0, 1, 0), (0, 0, 1)], roller_displacement_parameters, "sides_roller")

model.add_boundary_condition_on_plane([(0, 0, 0), (1, 0, 0), (0, 1, 0)], absorbing_boundaries_parameters, "abs")
model.add_boundary_condition_on_plane([(0, 0, model.extrusion_length), (1, 0, model.extrusion_length),
                                       (0, 1, model.extrusion_length)], absorbing_boundaries_parameters, "abs")
model.add_boundary_condition_on_plane([(max_x_coordinate, 0, 0), (max_x_coordinate, 1, 0), (max_x_coordinate, 0, 1)],
                                      absorbing_boundaries_parameters, "abs")

# aux code to calculate the required element size and time step for proper integration
# import numpy as np
# E = 150e6
# nu = 0.2
#
# M = E *(1 - nu)/(1 + nu)/(1 - 2*nu)
# rho = 1850
# vp = np.sqrt(M/rho)
# f = 100
#
# lambda_ = vp/f
#
# required_el_size = lambda_ / 10
# required_dt = required_el_size/vp

model.set_element_size_of_group(0.3 * 2, "ballast")
# model.set_element_size_of_group(0.5, "embankment")
model.set_element_size_of_group(0.75 * 2, "foundation_top")
model.set_element_size_of_group(0.75 * 2, "foundation_bot")

model.set_element_size_of_group(0.75 * 2, "deep_wall")

model.set_mesh_size(element_size=1.5 * 2)

# define at which points the json output should be written
delta_time = 0.001
json_output_parameters = JsonOutputParameters(delta_time - 1e-10, [NodalOutput.VELOCITY], [])
model.add_output_settings_by_coordinates([(x_coord_deep_wall + thickness_deep_wall, surface_level, 45.0),
                                          (25, surface_level, 45.0), (max_x_coordinate, surface_level, 45)],
                                         json_output_parameters, "json_output")

# set time integration parameters
end_time = 0.002
time_integration = TimeIntegration(start_time=0.0,
                                   end_time=end_time,
                                   delta_time=delta_time,
                                   reduction_factor=1,
                                   increase_factor=1,
                                   max_delta_time_factor=1000)

# set convergence criteria
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-5,
                                                        displacement_absolute_tolerance=1.0e-12)

# linear_solver = Lu()
linear_solver = Cg(scaling=True)
# set solver settings stage 1
solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                 solution_type=SolutionType.QUASI_STATIC,
                                 stress_initialisation_type=StressInitialisationType.NONE,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False,
                                 are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=NewtonRaphsonStrategy(),
                                 linear_solver_settings=linear_solver)

# Set up problem data
problem = Problem(problem_name="holten", number_of_threads=8, settings=solver_settings)
model.project_parameters = problem

# define the output settings in vtk format
model.add_output_settings(part_name="porous_computational_model_part",
                          output_dir=results_dir,
                          output_name="vtk_output",
                          output_parameters=VtkOutputParameters(
                              file_format="binary",
                              output_interval=15,
                              nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION],
                              gauss_point_results=[],
                              output_control_type="step"))

# show the geometry, to check if everything is correct
# model.show_geometry()

# create the stem object
stem = Stem(model, input_files_dir)

# create a new stage, and set the differences compared to stage 1
duration_stage_2 = 3.2
stage2 = stem.create_new_stage(delta_time, duration_stage_2)
stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
# stage2.project_parameters.settings.linear_solver_settings = Cg(scaling=False)
stage2.project_parameters.settings.linear_solver_settings = Cg(scaling=False)
stage2.project_parameters.settings.is_stiffness_matrix_constant = True
stage2.project_parameters.settings.are_mass_and_damping_constant = True
stage2.project_parameters.settings.strategy_type = LinearNewtonRaphsonStrategy()

# add rayleigh damping parameters
stage2.project_parameters.settings.rayleigh_k = 7.86e-5
stage2.project_parameters.settings.rayleigh_m = 0.248

# change the uvec parameters for the second stage
velocity = 38.9 - 1e-5  # 140 km/h minus a small value to prevent numerical issues (solved in upcoming version)
stage2.get_model_part_by_name("uvec_load").parameters.velocity = velocity
stage2.get_model_part_by_name("uvec_load").parameters.uvec_parameters["velocity"] = velocity
stage2.get_model_part_by_name("uvec_load").parameters.uvec_parameters["static_initialisation"] = False

# increase the virtual thickness of the side absorbing boundary conditions for proper damping
stage2.get_model_part_by_name("abs").parameters.virtual_thickness = 50

# add the new stage to the stem calculation
stem.add_calculation_stage(stage2)

stage2.project_parameters.settings._inititalize_acceleration = True

# write the input files
stem.write_all_input_files()

# run the calculation
stem.run_calculation()
