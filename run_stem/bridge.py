import os
import sys
from shutil import rmtree

import pytest
import numpy as np

from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.load import MovingLoad
from stem.model import Model
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    StressInitialisationType, SolverSettings, Problem
from stem.stem import Stem


# Define geometry, conditions and material parameters
# --------------------------------

# Specify dimension and initiate the model
ndim = 3
model = Model(ndim)
# add groups for extrusions
model.add_group_for_extrusion("Group 1", reference_depth=0, extrusion_length=25)
model.add_group_for_extrusion("Group 2", reference_depth=25, extrusion_length=10)
model.add_group_for_extrusion("Group 3", reference_depth=35, extrusion_length=25)
model.add_group_for_extrusion("pillar_group", reference_depth=29.6, extrusion_length=0.6)

# Specify material model
solid_density = 2650
porosity = 0.3
young_modulus = 30e6
poisson_ratio = 0.2
soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density, POROSITY=porosity)
constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus, POISSON_RATIO=poisson_ratio)
retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
material_soil = SoilMaterial("soil1", soil_formulation1, constitutive_law1, retention_parameters1)

# material for the bridging section
bridge_density = 7450
bridge_porosity = 0.0
bridge_young_modulus = 21e9
bridge_poisson_ratio = 0.3
bridge_soil_formulation = OnePhaseSoil(ndim,
                                       IS_DRAINED=True,
                                       DENSITY_SOLID=bridge_density,
                                       POROSITY=bridge_porosity)

bridge_constitutive_law = LinearElasticSoil(YOUNG_MODULUS=bridge_young_modulus, POISSON_RATIO=bridge_poisson_ratio)
bridge_retention_parameters = SaturatedBelowPhreaticLevelLaw()
material_bridge = SoilMaterial("steel", bridge_soil_formulation, bridge_constitutive_law,
                               bridge_retention_parameters)

# material for a pillar
pillar_density = 2650
pillar_porosity = 0.0
pillar_young_modulus = 25e9
pillar_poisson_ratio = 0.2
pillar_soil_formulation = OnePhaseSoil(ndim,
                                        IS_DRAINED=True,
                                        DENSITY_SOLID=pillar_density,
                                        POROSITY=pillar_porosity)

pillar_constitutive_law = LinearElasticSoil(YOUNG_MODULUS=pillar_young_modulus, POISSON_RATIO=pillar_poisson_ratio)
pillar_retention_parameters = SaturatedBelowPhreaticLevelLaw()
material_pillar = SoilMaterial("pillar", pillar_soil_formulation, pillar_constitutive_law,
                               pillar_retention_parameters)

# Specify the coordinates for the shapes to extrude: x, y, z [m]

embankment_coordinates_1 = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
soil1_coordinates = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
soil2_coordinates = [(0.0, 1.0, 0.0), (10.0, 1.0, 0.0), (10.0, 2.0, 0.0), (0.0, 2.0, 0.0)]

bridge_coordinates = [(0.0, 2.5, 25.0),(1.0, 2.5, 25.0), (1.5, 2.5, 25.0), (1.5, 3.0, 25.0), (0., 3.0, 25.0)]

embankment_coordinates_2 = [(0.0, 2.0, 35.0), (3.0, 2.0, 35.0), (1.5, 3.0, 35.0), (0.75, 3.0, 35.0), (0, 3.0, 35.0)]
soil1_coordinates_2 = [(0.0, 0.0, 35.0), (10.0, 0.0, 35.0), (10.0, 1.0, 35.0), (0.0, 1.0, 35.0)]
soil2_coordinates_2 = [(0.0, 1.0, 35.0), (10.0, 1.0, 35.0), (10.0, 2.0, 35.0), (0.0, 2.0, 35.0)]

pillar_coordinates = [(0.6, 0.0, 29.6), (1, 0.0, 29.6), (1, 2.5, 29.6), (0.6, 2.5, 29.6)]

# Create the soil layer
model.add_soil_layer_by_coordinates(embankment_coordinates_1, material_soil, "embankment1", "Group 1")
model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil, "soil1_group1", "Group 1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil, "soil2_group1", "Group 1")

model.add_soil_layer_by_coordinates(bridge_coordinates, material_bridge, "bridge", "Group 2")

model.add_soil_layer_by_coordinates(embankment_coordinates_2, material_soil, "embankment2", "Group 3")
model.add_soil_layer_by_coordinates(soil1_coordinates_2, material_soil, "soil1_group2", "Group 3")
model.add_soil_layer_by_coordinates(soil2_coordinates_2, material_soil, "soil2_group2", "Group 3")

model.add_soil_layer_by_coordinates(pillar_coordinates, material_pillar, "pillar", "pillar_group")

# add the track
rail_parameters = EulerBeam(ndim=ndim,
                            YOUNG_MODULUS=30e9,
                            POISSON_RATIO=0.2,
                            DENSITY=7200,
                            CROSS_AREA=0.01,
                            I33=1e-4,
                            I22=1e-4,
                            TORSIONAL_INERTIA=2e-4)
rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 750e6, 1],
                                          NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                          NODAL_DAMPING_COEFFICIENT=[1, 750e3, 1],
                                          NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                       NODAL_MASS=140,
                                       NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

origin_point = np.array([0.75, 3.0, 0.0])
direction_vector = np.array([0, 0, 1])
rail_pad_thickness = 0.025

model.generate_straight_track(0.5,121, rail_parameters, sleeper_parameters, rail_pad_parameters, rail_pad_thickness,
                                origin_point, direction_vector, "track")

# model.show_geometry(show_surface_ids=True)

# Define moving load
# load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 60.0)]

moving_load = MovingLoad(load=[0.0, -10.0e3, 0.0],
                         direction=[1, 1, 1],
                         velocity=0,
                         origin=[0.75, 3.0 + rail_pad_thickness, 0.0],
                         offset=5.0)

model.add_load_on_line_model_part("track", moving_load, "moving_load")
# model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

moving_load2 = MovingLoad(load=[0.0, -10.0e3, 0.0],
                         direction=[1, 1, 1],
                         velocity=0,
                         origin=[0.75, 3.0 + rail_pad_thickness, 0.0],
                         offset=7.0)

model.add_load_on_line_model_part("track", moving_load2, "moving_load2")
# model.add_load_by_coordinates(load_coordinates, moving_load2, "moving_load2")


output_coordinates = [(3.0, 2.0, 20), (6.5, 2.0, 20), (10, 2.0, 20)]

delta_time = 0.0005
model.add_output_settings_by_coordinates(output_coordinates,JsonOutputParameters(output_interval=delta_time-1e-7, nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]),"output_line")


# Define boundary conditions
no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True],
                                                    value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True],
                                                        value=[0, 0, 0])

absorbing_boundaries = AbsorbingBoundary(absorbing_factors=[1, 1], virtual_thickness=10)

model.synchronise_geometry()
model.show_geometry(show_surface_ids=True)

# Add boundary conditions to the model (geometry ids are shown in the show_geometry)
model.add_boundary_condition_by_geometry_ids(2, [8,39,51], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [11, 16, 22, 42, 47, 36,60,13,18,24,43,48,37], roller_displacement_parameters,
                                             "sides_roller"),
model.add_boundary_condition_by_geometry_ids(2, [44, 49, 38, 40, 45, 9, 14, 12, 17, 23], absorbing_boundaries, "absorbing_boundaries")

# Synchronize geometry
model.synchronise_geometry()

model.set_element_size_of_group(0.5,"embankment1")
model.set_element_size_of_group(0.5,"bridge")
# model.set_element_size_of_group(0.5,"embankment2")
# model.set_element_size_of_group(0.5,"soil1_group1")
# model.set_element_size_of_group(1,"soil1_group2")
model.set_element_size_of_group(0.2,"moving_load")
model.set_element_size_of_group(0.2,"output_line")
model.set_element_size_of_group(1,"pillar")

# Set mesh size
# --------------------------------
model.set_mesh_size(element_size=3)

# Define project parameters
# --------------------------------

# Set up solver settings
analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
solution_type = SolutionType.QUASI_STATIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0,
                                   end_time=0.1,
                                   delta_time=delta_time,
                                   reduction_factor=1.0,
                                   increase_factor=1.0,
                                   max_delta_time_factor=1000)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-9)
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type,
                                 solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=True,
                                 are_mass_and_damping_constant=True,
                                 convergence_criteria=convergence_criterion,
                                 rayleigh_k=0.0,
                                 rayleigh_m=0.0)

# Set up problem data
problem = Problem(problem_name="calculate_moving_load_on_3_groups_3d",
                  number_of_threads=2,
                  settings=solver_settings)
model.project_parameters = problem

# Define the results to be written to the output file

# Nodal results
nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]
# Gauss point results
gauss_point_results = []

# Define the output process
model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                output_interval=1,
                                                                nodal_results=nodal_results,
                                                                gauss_point_results=gauss_point_results,
                                                                output_control_type="step"),
                          part_name="porous_computational_model_part",
                          output_dir="output",
                          output_name="vtk_output")

input_folder = "moving_load_on_bridge"

# Write KRATOS input files
# --------------------------------
stem = Stem(model, input_folder)

stage2 = stem.create_new_stage(delta_time, 2.0)


stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
stage2.project_parameters.settings.rayleigh_k = 0.001
stage2.project_parameters.settings.rayleigh_m = 0.01

stage2.output_settings[0].output_parameters.output_interval = 1

stage2.get_model_part_by_name("moving_load").parameters.velocity = 30
stage2.get_model_part_by_name("moving_load2").parameters.velocity = 30

stem.add_calculation_stage(stage2)

stem.write_all_input_files()

# Run Kratos calculation
# --------------------------------
stem.run_calculation()

