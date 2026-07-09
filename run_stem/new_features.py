from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw, InterfaceMaterial
from stem.structural_material import ElasticSpringDamper, NodalConcentrated
from stem.load import UvecLoad, TrainType
import UVEC.uvec_ten_dof_vehicle_2D as uvec
from stem.default_materials import DefaultMaterial
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    LinearNewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Cg, Problem
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem
from stem.logger import enable_writing_to_log_file

import pathlib

input_files_dir = "new_features_example"
enable_writing_to_log_file(pathlib.Path(input_files_dir) / "stem.log")

ndim = 3
model = Model(ndim)
model.extrusion_length = 60

## -----define materials

poisson_ratio_undrained = 0.495
soil_formulation_clay = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1500, POROSITY=0.0)
constitutive_law_clay = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=poisson_ratio_undrained)
material_clay = SoilMaterial("clay", soil_formulation_clay, constitutive_law_clay, SaturatedBelowPhreaticLevelLaw())

soil_formulation_sand = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2000, POROSITY=0.0)
constitutive_law_sand = LinearElasticSoil(YOUNG_MODULUS=350e6, POISSON_RATIO=poisson_ratio_undrained)
material_sand = SoilMaterial("sand", soil_formulation_sand, constitutive_law_sand, SaturatedBelowPhreaticLevelLaw())

soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1850, POROSITY=0.0)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=150e6, POISSON_RATIO=0.2)
material_ballast = SoilMaterial("ballast", soil_formulation_1, constitutive_law_1, SaturatedBelowPhreaticLevelLaw())

soil_formulation_embankment = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1900, POROSITY=0.0)
constitutive_law_embankment = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
material_embankment = SoilMaterial("embankment", soil_formulation_embankment, constitutive_law_embankment,
                                   SaturatedBelowPhreaticLevelLaw())

## -----define geometry
surface_level = 0.7
ballast_coordinates = [(0.0, 2.6, 0.0), (1.5, 2.6, 0.0), (2.5, 2.2, 0.0), (0.0, 2.2, 0.0)]
embankment_coordinates = [(0.0, 2.2, 0.0), (3.5, 2.2, 0.0), (7.5, surface_level, 0.0), (0.0, surface_level, 0.0)]

layer_1_thickness = 3
bottom_layer_1 = surface_level - layer_1_thickness

soil1_coordinates_1 = [(0.0, surface_level, 0.0), (10.0, surface_level, 0.0), (10.0, bottom_layer_1, 0.0),
                       (0.0, bottom_layer_1, 0.0)]

layer_2_thickness = 2
bottom_layer_2 = bottom_layer_1 - layer_2_thickness

height_embankment = surface_level + 0.5
soil2_coordinates_1 = [(0.0, bottom_layer_1, 0.0), (10.0, bottom_layer_1, 0.0), (10.0, bottom_layer_2, 0.0),
                       (0.0, bottom_layer_2, 0.0)]

## -----add soil layers to the model
model.add_soil_layer_by_coordinates(soil1_coordinates_1, material_clay, "soil_layer_1")

model.add_soil_layer_by_coordinates(soil2_coordinates_1, material_sand, "soil_layer_2")

model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment")

model.add_soil_layer_by_coordinates(ballast_coordinates, material_ballast, "ballast")
# model.show_geometry()

rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters
rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                                          NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                          NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],
                                          NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2400, POROSITY=0.1)
constitutive_law = LinearElasticSoil(YOUNG_MODULUS=30e9, POISSON_RATIO=0.2)
sleeper_parameters = SoilMaterial(name="concrete",
                                  soil_formulation=soil_formulation,
                                  constitutive_law=constitutive_law,
                                  retention_parameters=SaturatedBelowPhreaticLevelLaw())

origin_point_track = [0.75, 2.6, 0]
direction_vector = [0, 0, 1]
# dimensions of the sleeper
sleeper_height = 0.3
rail_pad_thickness = 0.02
sleeper_length = 2.5 / 2
sleeper_width = 0.234
sleeper_distance = 0.6
sleeper_dimensions = [sleeper_width, sleeper_height, sleeper_length]
distance_middle_sleeper_to_rail = origin_point_track[0]

# create a straight track with rails, sleepers and rail pads
model.generate_straight_track(sleeper_distance=sleeper_distance,
                              n_sleepers=101,
                              rail_parameters=rail_parameters,
                              sleeper_parameters=sleeper_parameters,
                              rail_pad_parameters=rail_pad_parameters,
                              rail_pad_thickness=rail_pad_thickness,
                              origin_point=origin_point_track,
                              direction_vector=direction_vector,
                              sleeper_dimensions=sleeper_dimensions,
                              name="rail_track_1",
                              distance_middle_sleeper_to_rail=distance_middle_sleeper_to_rail)

# model.show_geometry()

interface_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
interface_const_law = LinearElasticSoil(YOUNG_MODULUS=120e6, POISSON_RATIO=0.3)

interface_material = InterfaceMaterial(name="interface_concrete_soil",
                                       constitutive_law=interface_const_law,
                                       soil_formulation=interface_formulation,
                                       retention_parameters=SaturatedBelowPhreaticLevelLaw())

model.set_interface_between_model_parts(["sleeper_rail_track_1"], ["ballast"], interface_material,
                                        "interface_sleeper_soil")

origin_point_load = origin_point_track.copy()
origin_point_load[1] += sleeper_height + rail_pad_thickness
uvec_load = UvecLoad(direction_signs=[1, 1, 1],
                     velocity=0.0,
                     origin=origin_point_load,
                     uvec_model=uvec,
                     train_type=TrainType.PASSENGERS_HEAVY,
                     irregularities=None,
                     rail_joint=None,
                     static_vehicle_calculation=True,
                     nb_carts=1)

model.add_load_on_line_model_part("rail_track_1", uvec_load, "uvec_load")

no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=10.0)

model.add_boundary_condition_on_plane([(0, bottom_layer_2, 0), (0, bottom_layer_2, 1), (1, bottom_layer_2, 0)],
                                      no_displacement_parameters, "base_fixed")
model.add_boundary_condition_on_plane([(0, 0, 0), (0, 0, 1), (0, 1, 0)], roller_displacement_parameters, "sides_roller")

model.add_boundary_condition_on_plane([(0, 0, 0), (1, 0, 0), (1, 1, 0)], absorbing_boundaries_parameters, "abs_z_min")
model.add_boundary_condition_on_plane([(0, 0, model.extrusion_length), (1, 0, model.extrusion_length),
                                       (1, 1, model.extrusion_length)], absorbing_boundaries_parameters, "abs_z_max")
model.add_boundary_condition_on_plane([(10, 0, 0), (10, 1, 0), (10, 0, 1)], absorbing_boundaries_parameters,
                                      "abs_x_max")

model.set_mesh_size(element_size=1.0)
model.mesh_settings.element_order = 1

# Set up start and end time of calculation, time step
time_integration = TimeIntegration(start_time=0.0,
                                   end_time=1.0,
                                   delta_time=0.5,
                                   reduction_factor=1.0,
                                   increase_factor=1.0)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-9)

solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                 solution_type=SolutionType.QUASI_STATIC,
                                 stress_initialisation_type=StressInitialisationType.NONE,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=True,
                                 are_mass_and_damping_constant=True,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=LinearNewtonRaphsonStrategy(),
                                 linear_solver_settings=Cg(),
                                 rayleigh_k=7.86e-5,
                                 rayleigh_m=0.248)

problem = Problem(problem_name="new_features_example", number_of_threads=8, settings=solver_settings)
model.project_parameters = problem

nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]

gauss_point_results = []
model.add_output_settings(part_name="porous_computational_model_part",
                          output_name="vtk_output",
                          output_dir="output",
                          output_parameters=VtkOutputParameters(output_interval=1,
                                                                nodal_results=nodal_results,
                                                                gauss_point_results=gauss_point_results,
                                                                output_control_type="step"))

stem = Stem(model, input_files_dir)

stage_2 = stem.create_new_stage(0.001, 0.5)
uvec_load = stage_2.get_model_part_by_name("uvec_load")
uvec_load.parameters.static_vehicle_calculation = False
uvec_load.parameters.velocity = 40

stage_2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
stem.add_calculation_stage(stage_2)

stem.write_all_input_files()
stem.run_calculation()
