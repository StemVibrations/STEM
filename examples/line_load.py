import sys
import os

path_kratos = r"./StemKratos"
material_name = "MaterialParameters.json"
project_name = "ProjectParameters.json"
mesh_name = "calculate_load_on_embankment_3d.mdpa"

input_files_dir = "line_load"
results_dir = "output_line_load"


sys.path.append(os.path.join(path_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(path_kratos, r"KratosGeoMechanics\libs"))

import KratosMultiphysics.GeoMechanicsApplication
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (GeoMechanicsAnalysis)

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
     NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.IO.kratos_io import KratosIO

ndim = 3
model = Model(ndim)


solid_density_1 = 2650
porosity_1 = 0.3
young_modulus_1 = 30e6
poisson_ratio_1 = 0.2
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity_1)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio_1)
retention_parameters_1 = SaturatedBelowPhreaticLevelLaw()
material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, retention_parameters_1)


solid_density_2 = 2550
porosity_2 = 0.3
young_modulus_2 = 30e6
poisson_ratio_2 = 0.2
soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_2, POROSITY=porosity_2)
constitutive_law_2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_2, POISSON_RATIO=poisson_ratio_2)
retention_parameters_2 = SaturatedBelowPhreaticLevelLaw()
material_soil_2 = SoilMaterial("soil_2", soil_formulation_2, constitutive_law_2, retention_parameters_2)

solid_density_3 = 2650
porosity_3 = 0.3
young_modulus_3 = 10e6
poisson_ratio_3 = 0.2
soil_formulation_3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity_3)
constitutive_law_3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio_3)
retention_parameters_3 = SaturatedBelowPhreaticLevelLaw()
material_embankment = SoilMaterial("embankment", soil_formulation_3, constitutive_law_3, retention_parameters_3)

soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
model.extrusion_length = [0, 0, 50]

model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")

load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]
line_load = LineLoad(active=[False, True, False], value=[0, -1000, 0])
model.add_load_by_coordinates(load_coordinates, line_load, "line_load")

model.synchronise_geometry()

model.show_geometry(show_surface_ids=True)

no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                             roller_displacement_parameters, "sides_roller")
                                             
model.set_mesh_size(element_size=1)

model.generate_mesh()


analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.DYNAMIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=0.099, delta_time=0.01, reduction_factor=1.0,
                                   increase_factor=1.0)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-9)
strategy_type = NewtonRaphsonStrategy()
scheme_type = NewmarkScheme()
linear_solver_settings = Amgcl()
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=strategy_type, scheme=scheme_type,
                                 linear_solver_settings=linear_solver_settings, rayleigh_k=0.12,
                                 rayleigh_m=0.0001)
                                 
# Set up problem data
problem = Problem(problem_name="calculate_load_on_embankment_3d", number_of_threads=1,
                  settings=solver_settings)
model.project_parameters = problem

nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
gauss_point_results = []

vtk_output_process = Output(
    part_name="porous_computational_model_part",
    output_name="vtk_output",
    output_dir="output",
    output_parameters=VtkOutputParameters(
        output_interval=1,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
   )
)

kratos_io = KratosIO(ndim=model.ndim)

kratos_io.write_input_files_for_kratos(
    model=model,
    outputs=[vtk_output_process],
    mesh_file_name=mesh_name, output_folder=input_files_dir
)

os.chdir(input_files_dir)

with open(project_name, "r") as parameter_file:
    kratos_parameters = KratosMultiphysics.Parameters(parameter_file.read())

kratos_model = KratosMultiphysics.Model()
simulation = GeoMechanicsAnalysis(kratos_model, kratos_parameters)
simulation.Run()

