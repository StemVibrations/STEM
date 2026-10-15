input_files_dir = "sheetpile_with_anchor2"
results_dir = "output2"

import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw, SmallStrainUmatLaw, MohrCoulombLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated, EulerBeam, Anchor
from stem.additional_processes import Excavation, ApplyFinalStressesOfPreviousStageToInitialState
# from stem.default_materials import DefaultMaterial
from stem.load import LineLoad, WaterLineLoad
from stem.water_processes import PhreaticLineWaterPressure
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg, Amgcl, Lu, LineSearchStrategy, ResidualConvergenceCriteria
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters, GaussPointOutput, GiDOutputParameters
from stem.table import Table
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
# constitutive_law_1 = SmallStrainUmatLaw(UMAT_NAME=r"C:\software_development\STEM\const_models\MohrCoulombUMAT.dll", IS_FORTRAN_UMAT=True,
#                                         UMAT_PARAMETERS= [6.25e7,0.2, 5000,45,45,1e10],
#                                         STATE_VARIABLES= [0])
"GeoMohrCoulombWithTensionCutOff2D"
# constitutive_law_1 = MohrCoulombLaw(YOUNG_MODULUS=150e6,
#                                     POISSON_RATIO=0.2,
#                                     GEO_FRICTION_ANGLE=45,
#                                     GEO_DILATANCY_ANGLE=45,
#                                     GEO_COHESION=5000,
#                                     GEO_TENSILE_STRENGTH=5000)

umat_name = r"C:\software_development\ConstitutiveModels2\build_C\lib\matsuoka_nakai.dll"
# umat_name = r"C:\software_development\STEM\const_models\matsuoka_nakai.dll"
constitutive_law_1 = SmallStrainUmatLaw(UMAT_NAME=umat_name,
                                        IS_FORTRAN_UMAT=False,
                                        UMAT_PARAMETERS=[150e6, 0.2, 5000, 45, 45],
                                        STATE_VARIABLES=[0])

#
# constitutive_law_1 = SmallStrainUmatLaw(UMAT_NAME=r"C:\software_development\STEM\const_models\linear_elastic.dll", IS_FORTRAN_UMAT=True,
#                                         UMAT_PARAMETERS= [450e6,0.2],
#                                         STATE_VARIABLES= [0])

# constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=150e6, POISSON_RATIO=0.2)
material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, SaturatedBelowPhreaticLevelLaw())

# soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850, POROSITY=0.0)
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2350, POROSITY=0.0)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=25e9, POISSON_RATIO=0.2)
# constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=150e6, POISSON_RATIO=0.2)
material_concrete = SoilMaterial("concrete", soil_formulation_1, constitutive_law_1, SaturatedBelowPhreaticLevelLaw())

model.add_soil_layer_by_coordinates(top_soil_coords_left_part_1, material_soil_1, "top_soil_left_part_1")
model.add_soil_layer_by_coordinates(top_soil_coords_left_part_2, material_soil_1, "top_soil_left_part_2")
model.add_soil_layer_by_coordinates(top_soil_coords_right_part1, material_soil_1, "top_soil_right_part1")
model.add_soil_layer_by_coordinates(top_soil_coords_right_part2, material_soil_1, "top_soil_right_part2")
model.add_soil_layer_by_coordinates(concrete_cover_coordinates, material_soil_1, "concrete_cover")

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
# sheetpile_material = EulerBeam(2, YOUNG_MODULUS=2.05e8, POISSON_RATIO=0.2, CROSS_AREA=0.2449, I33=0.001225, DENSITY=2350)
# sheetpile_material = EulerBeam(2, YOUNG_MODULUS=25e9, POISSON_RATIO=0.2, DENSITY=2350, THICKNESS=0.2449, THICKNESS_EFFECTIVE_Y=1)

sheetpile_coordinates = [(sheetpile_x_coordinate, sheetpile_begin_level - sheetpile_length, 0.0),
                         (sheetpile_x_coordinate, sheetpile_begin_level, 0.0)]
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
# grout_material = EulerBeam(2, YOUNG_MODULUS=25e9, POISSON_RATIO=0.2, DENSITY=2350, THICKNESS=0.3464, THICKNESS_EFFECTIVE_Y=1)
SheetPileUtils.add_grout_by_coordinates(anchor_grout_coordinates, grout_material, None, "grout", model.gmsh_io,
                                        model.body_model_parts)

point_stiffness = 1e9
point_stiffness_parameters = NodalConcentrated([0, point_stiffness, 0], 0, [0, 0, 0])
point_stiffness_coordinates = [(sheetpile_x_coordinate, sheetpile_begin_level - sheetpile_length, 0.0)]

SheetPileUtils.add_point_element_by_coordinates(point_stiffness_coordinates, point_stiffness_parameters,
                                                "point_stiffness", model.gmsh_io, model.body_model_parts)

table = Table(values=[0, 0, -15000, -15000], times=[0, 6, 7, 8])

line_load = LineLoad(active=[False, False, False], value=[0, table, 0])
# line_load = LineLoad(active=[False, False, False], value=[0, -0, 0])
model.add_load_by_coordinates([(sheetpile_x_coordinate, surface_level, 0.0),
                               (sheetpile_x_coordinate + 2, surface_level, 0.0)], line_load, "line_load")

# model.show_geometry(show_line_ids=True)
water_load = WaterLineLoad(active=True, reference_coordinate=[-10, -10, 0])
model.add_load_by_geometry_ids([18], water_load, "water_normal_load_soil")
model.add_load_by_geometry_ids([19, 7], water_load, "water_normal_load_beam")

phreatic_level = 3
phreatic_line = PhreaticLineWaterPressure([(-10, phreatic_level, 0), (0, phreatic_level, 0), (10, phreatic_level, 0)],
                                          is_fixed=True)

model.add_phreatic_line(phreatic_line)

# model.show_geometry(show_line_ids=True)

# define the boundary conditions
roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])

no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])

model.add_boundary_condition_by_geometry_ids(1, [4, 5, 20, 23, 28, 15, 25, 30], roller_displacement_parameters,
                                             "sides_roller")
model.add_boundary_condition_by_geometry_ids(1, [26, 29], no_displacement_parameters, "bottom_no_disp")

excavation_building_pit_left_1 = Excavation(deactivate_body_model_part=False)
excavation_building_pit_left_2 = Excavation(deactivate_body_model_part=False)
excavation_building_pit_right = Excavation(deactivate_body_model_part=False)
excavation_sheetpile = Excavation(deactivate_body_model_part=True)
excavation_anchor = Excavation(deactivate_body_model_part=True)
excavation_concrete_cover = Excavation(deactivate_body_model_part=False)

model.apply_additional_process(excavation_building_pit_left_1, "top_soil_left_part_1")
model.apply_additional_process(excavation_building_pit_left_2, "top_soil_left_part_2")
model.apply_additional_process(excavation_building_pit_left_2, "clay_left_part1")

model.apply_additional_process(excavation_building_pit_right, "top_soil_right_part1")
model.apply_additional_process(excavation_sheetpile, "sheetpile")
model.apply_additional_process(excavation_sheetpile, "point_stiffness")
model.apply_additional_process(excavation_anchor, "anchor")
model.apply_additional_process(excavation_anchor, "grout")
model.apply_additional_process(excavation_concrete_cover, "concrete_cover")

move_stress = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["sheetpile", "anchor", "grout"])
# move_stress = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=[ "grout"])
# model.apply_additional_process(move_stress)

model.set_element_size_of_group(0.5, "top_soil_left_part_1")
# model.set_element_size_of_group(0.75 * 2, "top_soil_left_part_2")
model.set_element_size_of_group(0.4, "top_soil_right_part1")
model.set_element_size_of_group(0.4 , "top_soil_right_part2")
model.set_element_size_of_group(0.4, "concrete_cover")
model.set_element_size_of_group(0.4, "sheetpile")
# model.set_element_size_of_group(0.75 * 2, "clay_left_part1")
# model.set_element_size_of_group(0.75 * 2, "clay_left_part2")
# model.set_element_size_of_group(0.75 * 2, "clay_right")

model.set_mesh_size(element_size=1.0)

# define at which points the json output should be written
delta_time = 0.5
# json_output_parameters = JsonOutputParameters(delta_time - 1e-10, [NodalOutput.VELOCITY], [])
# model.add_output_settings_by_coordinates([(x_coord_deep_wall + thickness_deep_wall, surface_level, 45.0),
#                                           (25, surface_level, 45.0), (max_x_coordinate, surface_level, 45)],
#                                          json_output_parameters, "json_output")

# set time integration parameters
end_time = 1
time_integration = TimeIntegration(start_time=0.0,
                                   end_time=end_time,
                                   delta_time=delta_time,
                                   reduction_factor=0.5,
                                   increase_factor=1.5,
                                   max_delta_time_factor=1000)

# set convergence criteria
convergence_criterion = ResidualConvergenceCriteria(residual_relative_tolerance=1.0e-2,
                                                        residual_absolute_tolerance=1.0e-2)

linear_solver = Lu()
# linear_solver = Cg(scaling=False, preconditioner_type="none")
# linear_solver = Amgcl(smoother_type="damped_jacobi", krylov_type="cg", tolerance=1e-12, max_iteration=1000)
# set solver settings stage 1
solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                 solution_type=SolutionType.QUASI_STATIC,
                                 stress_initialisation_type=StressInitialisationType.GRAVITY_LOADING,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False,
                                 are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=NewtonRaphsonStrategy(),
                                 linear_solver_settings=linear_solver)

# Set up problem data
problem = Problem(problem_name="sheetpile", number_of_threads=16, settings=solver_settings)
model.project_parameters = problem

# define the output settings in vtk format
model.add_output_settings(
    part_name="porous_computational_model_part",
    output_dir=results_dir,
    output_name="vtk_output",
    output_parameters=VtkOutputParameters(
        file_format="ascii",
        output_interval=1,
        nodal_results=[NodalOutput.TOTAL_DISPLACEMENT, NodalOutput.DISPLACEMENT, NodalOutput.WATER_PRESSURE],
        gauss_point_results=[GaussPointOutput.CAUCHY_STRESS_VECTOR],
        output_control_type="step"))

# model.add_output_settings(part_name="porous_computational_model_part",
#                           output_dir=results_dir,
#                           output_name="vtk_output",
#                           output_parameters=GiDOutputParameters(
#                               file_format="ascii",
#                               output_interval=1,
#                               nodal_results=[NodalOutput.TOTAL_DISPLACEMENT, NodalOutput.DISPLACEMENT],
#                               gauss_point_results=[GaussPointOutput.CAUCHY_STRESS_TENSOR],
#                               output_control_type="step"))

# show the geometry, to check if everything is correct
# model.show_geometry()

# create the stem object
stem = Stem(model, input_files_dir)
#
# # create a new stage, excavate top part
duration_stage_2 = 1
stage2 = stem.create_new_stage(0.2, duration_stage_2)
# stage2.project_parameters.settings.reset_displacements = True
top_soil_exc_left_part_1 = stage2.get_additional_process_part_by_name_and_type("top_soil_left_part_1", Excavation)
top_soil_exc_left_part_1.parameters.deactivate_body_model_part = True
top_soil_exc_left_part_1.parameters.changed_phase = True

top_soil_exc_right_part_1 = stage2.get_additional_process_part_by_name_and_type("top_soil_right_part1", Excavation)
top_soil_exc_right_part_1.parameters.deactivate_body_model_part = True
top_soil_exc_right_part_1.parameters.changed_phase = True

top_soil_exc_right_part_1 = stage2.get_additional_process_part_by_name_and_type("concrete_cover", Excavation)
top_soil_exc_right_part_1.parameters.deactivate_body_model_part = True
top_soil_exc_right_part_1.parameters.changed_phase = True

stage2.project_parameters.settings._read_force = True

stem.add_calculation_stage(stage2)

# create a new stage, add sheetpile and anchor
stage_3 = stem.create_new_stage(0.01, 1)
stage_3.project_parameters.settings.reset_displacements = False

sheet_pile_exc = stage_3.get_additional_process_part_by_name_and_type("sheetpile", Excavation)
sheet_pile_exc.parameters.deactivate_body_model_part = False

anchor_exc = stage_3.get_additional_process_part_by_name_and_type("anchor", Excavation)
anchor_exc.parameters.deactivate_body_model_part = False

grout_exc = stage_3.get_additional_process_part_by_name_and_type("grout", Excavation)
grout_exc.parameters.deactivate_body_model_part = False
#
# point_stiffness_exc = stage_3.get_additional_process_part_by_name_and_type("point_stiffness", Excavation)
# point_stiffness_exc.parameters.deactivate_body_model_part = False

# top_soil_exc_left_part_1 = stage_3.get_additional_process_part_by_name_and_type("top_soil_left_part_1", Excavation)
# top_soil_exc_left_part_1.parameters.changed_phase = False
#
# top_soil_exc_right_part_1 = stage_3.get_additional_process_part_by_name_and_type("top_soil_right_part1", Excavation)
# top_soil_exc_right_part_1.parameters.changed_phase = False
#
# top_soil_exc_right_part_1 = stage_3.get_additional_process_part_by_name_and_type("concrete_cover", Excavation)
# top_soil_exc_right_part_1.parameters.changed_phase = False



stem.add_calculation_stage(stage_3)

# create a new stage, fill the excavation on the right side
stage_4 = stem.create_new_stage(0.01, 1)

top_soil_exc_right_part_1 = stage_4.get_additional_process_part_by_name_and_type("top_soil_right_part1", Excavation)
top_soil_exc_right_part_1.parameters.deactivate_body_model_part = False
top_soil_exc_right_part_1.parameters.changed_phase = False

stage_4.get_model_part_by_name("concrete_cover").material = material_concrete
top_soil_exc_right_part_1 = stage_4.get_additional_process_part_by_name_and_type("concrete_cover", Excavation)
top_soil_exc_right_part_1.parameters.deactivate_body_model_part = False
top_soil_exc_right_part_1.parameters.changed_phase = False

move_stress = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["sheetpile", "anchor", "grout"])
# move_stress = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=[ "grout"])
stage_4.apply_additional_process(move_stress)

stem.add_calculation_stage(stage_4)

# create a new stage, fill the excavation on the right side
stage_5 = stem.create_new_stage(0.01, 1)
anchor_model_part = stage_5.get_model_part_by_name("anchor")
anchor_model_part.material.material_parameters.PRESTRESS = 100e3 / anchor_model_part.material.material_parameters.CROSS_AREA
stem.add_calculation_stage(stage_5)

# # create a new stage, further excavate the left side, which is now supported by the sheetpile and anchor
# stage_6 = stem.create_new_stage(0.5, 1)
#
#
# soil_formulation_clay = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850/50, POROSITY=0.0)
# constitutive_law_clay = LinearElasticSoil(YOUNG_MODULUS=50e6/50, POISSON_RATIO=0.49)
# material_clay_new = SoilMaterial("clay_new", soil_formulation_clay, constitutive_law_clay, SaturatedBelowPhreaticLevelLaw())
#
# soil_formulation_1_new = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850/50, POROSITY=0.0)
# constitutive_law_new = LinearElasticSoil(YOUNG_MODULUS=150e6/50, POISSON_RATIO=0.49)
#
# material_soil_1_new = SoilMaterial("soil_1_new", soil_formulation_1_new, constitutive_law_new, SaturatedBelowPhreaticLevelLaw())
#
# stage_6.get_model_part_by_name("top_soil_left_part_2").material = material_soil_1
# stage_6.get_model_part_by_name("clay_left_part1").material = material_clay_new
#
# anchor_model_part = stage_6.get_model_part_by_name("anchor")
# anchor_model_part.material.material_parameters.PRESTRESS = 0.0
#
# stem.add_calculation_stage(stage_6)

stage_7 = stem.create_new_stage(2.5e-5, 1)

top_soil_exc_left_part_2 = stage_7.get_additional_process_part_by_name_and_type("top_soil_left_part_2", Excavation)
top_soil_exc_left_part_2.parameters.deactivate_body_model_part = True

clay_exc_left_part_1 = stage_7.get_additional_process_part_by_name_and_type("clay_left_part1", Excavation)
clay_exc_left_part_1.parameters.deactivate_body_model_part = True

anchor_model_part = stage_7.get_model_part_by_name("anchor")
anchor_model_part.material.material_parameters.PRESTRESS = 0.0

new_phreatic_level = -2.0
stage_7.get_model_part_by_name("water_normal_load_soil").parameters.reference_coordinate = [-10, new_phreatic_level, 0]
stage_7.get_model_part_by_name("water_normal_load_beam").parameters.reference_coordinate = [-10, new_phreatic_level, 0]
water_process_part = stage_7.water_process_parts[0]
phreatic_line = PhreaticLineWaterPressure([(-10, new_phreatic_level, 0), (0, new_phreatic_level, 0),(0.0, phreatic_level, 0), (10, phreatic_level, 0)],
                                          is_fixed=True)
water_process_part.parameters = phreatic_line

stem.add_calculation_stage(stage_7)

# apply loading
stage_8 = stem.create_new_stage(0.1, 1)
line_load_mp  = stage_8.get_model_part_by_name("line_load")
line_load_mp.parameters.active = [True, True, True]

stem.add_calculation_stage(stage_8)

# write the input files
stem.write_all_input_files()

import time

t0 = time.time()
# run the calculation
stem.run_calculation()
print(f"Total calculation time: {time.time() - t0} seconds")
