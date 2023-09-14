import sys
import os

path_kratos = r"D:\Kratos_without_compiling"
material_name = "MaterialParameters.json"
project_name = "ProjectParameters.json"
mesh_name = "calculate_moving_load_on_embankment_3d.mdpa"

sys.path.append(os.path.join(path_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(path_kratos, r"KratosGeoMechanics\libs"))

import KratosMultiphysics.GeoMechanicsApplication
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (GeoMechanicsAnalysis)
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import MovingLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, GiDOutputParameters, Output
from stem.IO.kratos_io import KratosIO


# Define geometry, conditions and material parameters
# --------------------------------

# Specify dimension and initiate the model
ndim = 3
model = Model(ndim)

# Specify material model
solid_density = 2650
porosity = 0.3
young_modulus = 10e6
poisson_ratio = 0.2
soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density, POROSITY=porosity)
constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus, POISSON_RATIO=poisson_ratio)
retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
material_soil1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)
material_soil2 = SoilMaterial("soil2", soil_formulation1, constitutive_law1, retention_parameters1)
material_embankment = SoilMaterial("embankment", soil_formulation1, constitutive_law1, retention_parameters1)

# Specify the coordinates for the column: x:2m x y:2m x z:10m
soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
soil2_coordinates = [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0), (5.0, 0.0, 2.0), (0.0, 0.0, 2.0)]
embankment_coordinates = [(0.0, 0.0, 2.0), (3.0, 0.0, 2.0), (1.5, 0.0, 3.0), (0, 0.0, 3.0)]
model.extrusion_length = [0, 10, 0]

# Create the soil layer
model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil1, "soil1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil2, "soil2")
model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment")

# Define moving load
load_coordinates = [(0.75, 0.0, 3.0), (0.75, 10.0, 3.0)]
moving_load = MovingLoad(load=[0.0, -10.0, 0.0], direction=[1, 1, 1], velocity=5, origin=[0.75, 0.0, 3.0], offset=0.0)
model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

# Define boundary conditions
no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, False, False], value=[0, 0, 0])

# Add boundary conditions to the model (geometry ids are shown in the show_geometry)
model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
# model.add_boundary_condition_by_geometry_ids(1, [2, 4, 5, 6], roller_displacement_parameters, "roller_fixed")

# Synchronize geometry
model.synchronise_geometry()

# Show geometry and geometry ids
#TODO: show_geometry is not responding
model.show_geometry(show_surface_ids=True)
# input()

# Set mesh size and generate mesh
# --------------------------------

model.set_mesh_size(element_size=1)
model.generate_mesh()

# Define project parameters
# --------------------------------

# Set up solver settings
analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
solution_type = SolutionType.QUASI_STATIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=1.0, delta_time=0.01, reduction_factor=1.0,
                                   increase_factor=1.0, max_delta_time_factor=1000)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-9)
strategy_type = NewtonRaphsonStrategy(min_iterations=6, max_iterations=15, number_cycles=100)
scheme_type = NewmarkScheme(newmark_beta=0.25, newmark_gamma=0.5, newmark_theta=0.5)
linear_solver_settings = Amgcl(tolerance=1e-8, max_iteration=500, scaling=True)
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False, are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=strategy_type, scheme=scheme_type,
                                 linear_solver_settings=linear_solver_settings, rayleigh_k=0.0,
                                 rayleigh_m=0.0)

# Set up problem data
problem = Problem(problem_name="calculate_moving_load_on_embankment_3d", number_of_threads=1, settings=solver_settings)
model.project_parameters = problem

# Define the results to be written to the output file

# Nodal results
nodal_results = [NodalOutput.DISPLACEMENT,
                 NodalOutput.TOTAL_DISPLACEMENT]
# Gauss point results
gauss_point_results = [
]

# Define the output process
gid_output = Output(
    part_name="porous_computational_model_part",
    output_name="gid_output",
    output_dir="output",
    output_parameters=GiDOutputParameters(
        file_format="binary",
        output_interval=1,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
# vtk_output_process = Output(
#     part_name="porous_computational_model_part",
#     output_name="vtk_output",
#     output_dir="output",
#     output_parameters=VtkOutputParameters(
#         file_format="binary",
#         output_interval=1,
#         nodal_results=nodal_results,
#         gauss_point_results=gauss_point_results,
#         output_control_type="step"
    )
)

# Write KRATOS input files
# --------------------------------

kratos_io = KratosIO(ndim=model.ndim)
# Define the output folder
output_folder = "inputs_kratos"

# Write project settings to ProjectParameters.json file
kratos_io.write_project_parameters_json(
    model=model,
    outputs=[gid_output],
    mesh_file_name="calculate_moving_load_on_embankment_3d.mdpa",
    materials_file_name="MaterialParameters.json",
    output_folder=output_folder
)

# Write mesh to .mdpa file
kratos_io.write_mesh_to_mdpa(
    model=model,
    mesh_file_name="calculate_moving_load_on_embankment_3d.mdpa",
    output_folder=output_folder
)

# Write materials to MaterialParameters.json file
kratos_io.write_material_parameters_json(
    model=model,
    output_folder=output_folder
)

# Run Kratos calculation
# --------------------------------

project_folder = "inputs_kratos"
os.chdir(project_folder)

with open(project_name, "r") as parameter_file:
    parameters = KratosMultiphysics.Parameters(parameter_file.read())

model = KratosMultiphysics.Model()
simulation = GeoMechanicsAnalysis(model, parameters)
simulation.Run()
