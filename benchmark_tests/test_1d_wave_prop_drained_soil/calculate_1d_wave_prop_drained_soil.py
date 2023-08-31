import sys
import os

path_kratos = r"D:\Kratos_without_compiling"
material_name = "MaterialParameters.json"
project_name = "ProjectParameters.json"
mesh_name = "test_1d_wave_prop_drained_soil.mdpa"

sys.path.append(os.path.join(path_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(path_kratos, r"KratosGeoMechanics\libs"))

import KratosMultiphysics.GeoMechanicsApplication
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (GeoMechanicsAnalysis)
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.table import Table
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
# Linear elastic drained soil with a Density of 2700, a Young's modulus of 50e6,
# a Poisson ratio of 0.3 & a Porosity of 0.3 is specified.
rho1 = 2700
por1 = 0.3
E1 = 50e6
v1 = 0.3
soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=rho1, POROSITY=por1)
constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=E1, POISSON_RATIO=v1)
retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

# Specify the coordinates for the column: x:1m x y:10m
layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 10, 0), (0, 10, 0)]

# Create the soil layer
model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_column")

# Boundary conditions and Loads
load_coordinates = [(0.0, 10.0, 0), (1.0, 10.0, 0)]
# Add table for the load in the mdpa file
t = (0.0, 0.0025, 1)
values = (0.0, -1000.0, -1000.0)
LINE_LOAD_Y = Table(times=t, values=values)
# Add line load
line_load = LineLoad(active=[False, True, False], value=[0, LINE_LOAD_Y, 0])
model.add_load_by_coordinates(load_coordinates, line_load, "load")

# Define boundary conditions
no_displacement_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                    value=[0, 0, 0])
sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

# Add boundary conditions to the model (geometry ids are shown in the show_geometry)
model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(1, [2, 4], sym_parameters, "side_rollers")

# Synchronize geometry
model.synchronise_geometry()

# Show geometry and geometry ids
# TODO: bug in show_geometry
model.show_geometry(show_line_ids=True, show_point_ids=True)

# Set mesh size and generate mesh
# --------------------------------

model.set_mesh_size(element_size=0.45)
model.generate_mesh()

# Define project parameters
# --------------------------------

# Set up solver settings
analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
solution_type = SolutionType.DYNAMIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=0.15, delta_time=0.0025, reduction_factor=1.0,
                                   increase_factor=1.0, max_delta_time_factor=1000)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-12,
                                                        displacement_absolute_tolerance=1.0E-6)
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
                                 linear_solver_settings=linear_solver_settings, rayleigh_k=6e-6,
                                 rayleigh_m=0.02)

# Set up problem data
problem = Problem(problem_name="test_1d_wave_prop_drained_soil", number_of_threads=2, settings=solver_settings)
model.project_parameters = problem

# Define the results to be written to the output file

# Nodal results
nodal_results = [NodalOutput.VELOCITY]
# Gauss point results
gauss_point_results = [
]

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
    mesh_file_name="test_1d_wave_prop_drained_soil.mdpa",
    materials_file_name="MaterialParameters.json",
    output_folder=output_folder
)

# Write mesh to test_1d_wave_prop_drained_soil.mdpa file
kratos_io.write_mesh_to_mdpa(
    model=model,
    mesh_file_name="test_1d_wave_prop_drained_soil.mdpa",
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
