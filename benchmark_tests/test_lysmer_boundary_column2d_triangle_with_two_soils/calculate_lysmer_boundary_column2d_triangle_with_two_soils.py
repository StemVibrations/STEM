import sys
import os

path_kratos = r"D:\Kratos_without_compiling"
material_name = "MaterialParameters.json"
project_name = "ProjectParameters.json"
mesh_name = "calculate_lysmer_boundary_column2d_triangle_with_two_soils.mdpa"

sys.path.append(os.path.join(path_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(path_kratos, r"KratosGeoMechanics\libs"))

import KratosMultiphysics.GeoMechanicsApplication
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (GeoMechanicsAnalysis)
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.boundary import AbsorbingBoundary
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.IO.kratos_io import KratosIO



# Define geometry, conditions and material parameters
# --------------------------------

# Specify dimension and initiate the model
ndim = 2
model = Model(ndim)

# Specify material model
# Linear elastic drained soil with a Density of 2650, a Young's modulus of 10.0e7,
# a Poisson ratio of 0.2 & a Porosity of 0.3 is specified.
DENSITY_SOLID = 2650
POROSITY = 0.3
YOUNG_MODULUS = 10.0e7
POISSON_RATIO = 0.2
soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
material1 = SoilMaterial("soil1", soil_formulation1, constitutive_law1, retention_parameters1)
material2 = SoilMaterial("soil2", soil_formulation1, constitutive_law1, retention_parameters1)

# Specify the coordinates for the column: x:1m x y:10m
layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 5, 0), (0, 5, 0)]
layer2_coordinates = [(0, 5, 0), (1, 5, 0), (1, 10, 0), (0, 10, 0)]

# Create the soil layers
model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "layer1")
model.add_soil_layer_by_coordinates(layer2_coordinates, material2, "layer2")

# Boundary conditions and Loads
load_coordinates = [(0.0, 10.0, 0), (1.0, 10.0, 0)]

# Add line load
line_load = LineLoad(active=[False, True, False], value=[0, -10, 0])
model.add_load_by_coordinates(load_coordinates, line_load, "load")

# Define absorbing boundary condition
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=1000.0)

# Define displacement conditions
displacement_parameters = DisplacementConstraint(active=[True, False, False], is_fixed=[True, False, False], value=[0, 0, 0])

# Add boundary conditions to the model (geometry ids are shown in the show_geometry)

model.add_boundary_condition_by_geometry_ids(1, [1], absorbing_boundaries_parameters, "abs")
model.add_boundary_condition_by_geometry_ids(1, [2, 4, 5, 7], displacement_parameters, "sides")

# Synchronize geometry
model.synchronise_geometry()

# Show geometry and geometry ids
model.show_geometry(show_line_ids=True)

# Add gravity to the geometry, set mesh size and generate mesh
# --------------------------------
model._Model__add_gravity_load()
model.set_mesh_size(element_size=2)
model.generate_mesh()

# Define project parameters
# --------------------------------

# Set up solver settings
analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
solution_type = SolutionType.DYNAMIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=0.3, delta_time=0.01, reduction_factor=1.0,
                                   increase_factor=1.0, max_delta_time_factor=1000)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-9)
strategy_type = NewtonRaphsonStrategy(min_iterations=6, max_iterations=30, number_cycles=100)
scheme_type = NewmarkScheme(newmark_beta=0.25, newmark_gamma=0.5, newmark_theta=0.5)
linear_solver_settings = Amgcl(tolerance=1.0e-6, max_iteration=1000, scaling=True)
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False, are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=strategy_type, scheme=scheme_type,
                                 linear_solver_settings=linear_solver_settings, rayleigh_k=0.001,
                                 rayleigh_m=0.1)

# Set up problem data
problem = Problem(problem_name="calculate_lysmer_boundary_column2d_triangle_with_two_soils",
                  number_of_threads=1, settings=solver_settings)
model.project_parameters = problem

# Define the results to be written to the output file

# Nodal results
nodal_results = [NodalOutput.DISPLACEMENT,
                 NodalOutput.VELOCITY]
# Gauss point results
gauss_point_results = []

# Define the output process
vtk_output_process = Output(
    part_name="porous_computational_model_part",
    output_name="vtk_output",
    output_dir="output",
    output_parameters=VtkOutputParameters(
        file_format="binary",
        output_interval=1,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
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
    outputs=[vtk_output_process],
    mesh_file_name="calculate_lysmer_boundary_column2d_triangle_with_two_soils.mdpa",
    materials_file_name="MaterialParameters.json",
    output_folder=output_folder
)

# Write mesh to .mdpa file
kratos_io.write_mesh_to_mdpa(
    model=model,
    mesh_file_name="calculate_lysmer_boundary_column2d_triangle_with_two_soils.mdpa",
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
