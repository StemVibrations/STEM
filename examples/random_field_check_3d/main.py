import os
import sys
from pathlib import Path

# set working directory in STEM!

path_kratos = r"C:/Users/morettid/OneDrive - TNO/Desktop/projects/STEM"
material_name = "MaterialParameters.json"
project_name = "ProjectParameters.json"
mesh_name = "simple_3d_mesh_rf.mdpa"

working_folder = os.getcwd()
main_folder = os.path.join(working_folder,"examples/random_field_check_3d")
input_files_dir = "kratos_inputs"
results_dir = "kratos_output"


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
from stem.output import NodalOutput, VtkOutputParameters, GaussPointOutput
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
young_modulus_3 = 30e6
poisson_ratio_3 = 0.2
soil_formulation_3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity_3)
constitutive_law_3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio_3)
retention_parameters_3 = SaturatedBelowPhreaticLevelLaw()
material_embankment = SoilMaterial("embankment", soil_formulation_3, constitutive_law_3, retention_parameters_3)

soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
model.extrusion_length = 1

model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")
model.synchronise_geometry()

depth = model.extrusion_length
load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, depth)]
line_load = LineLoad(active=[False, True, False], value=[0, -1000, 0])
model.add_load_by_coordinates(load_coordinates, line_load, "line_load")
# model.show_geometry(show_surface_ids=True)

cov = 0.2
v_scale_fluctuation = 1
anisotropy=[10.0, 10.0]
angle=[0, 0]
seed = 42
model_name="Gaussian"

model.add_random_field(part_name="soil_layer_2", property_name="YOUNG_MODULUS", cov=cov,
                       v_scale_fluctuation=v_scale_fluctuation, anisotropy=anisotropy,
                       angle=angle, seed=seed, model_name=model_name)


model.synchronise_geometry()
model.show_geometry(show_surface_ids=True, show_line_ids=True)

no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])


model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 16, 17, 18],
                                             roller_displacement_parameters, "sides_roller")
# generate mesh
model.set_mesh_size(element_size=0.2)
model.generate_mesh(save_file=True, open_gmsh_gui=True)

analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.QUASI_STATIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=0.01, delta_time=0.01, reduction_factor=1.0,
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
gauss_point_results = [ GaussPointOutput.YOUNG_MODULUS]

model.add_model_part_output(
    part_name="porous_computational_model_part",
    output_name="vtk_output",
    output_dir=results_dir,
    output_parameters=VtkOutputParameters(
        output_interval=1,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
   )
)

kratos_io = KratosIO(ndim=model.ndim)
kratos_io.set_project_folder(os.path.join(main_folder, input_files_dir))

kratos_io.write_input_files_for_kratos(
    model=model,
    mesh_file_name=mesh_name,
)

current_folder = Path(os.getcwd())
os.chdir(os.path.join(main_folder, input_files_dir))

with open(project_name, "r") as parameter_file:
    kratos_parameters = KratosMultiphysics.Parameters(parameter_file.read())

kratos_model = KratosMultiphysics.Model()
simulation = GeoMechanicsAnalysis(kratos_model, kratos_parameters)
simulation.Run()
os.chdir(current_folder)

#  ------- test random field figure generated directly from python using the same centroids ---------------------- #

centroids = model.get_centroids_elements_model_part(model.body_model_parts[0].name)

import matplotlib as mpl
from matplotlib import pyplot as plt
from random_fields.generate_field import RandomFields, ModelName


mean = material_soil_2.get_property_in_material("YOUNG_MODULUS")
# dimensions of anisotropy and angle are adjusted automatically in model.add_random_field()

rf = RandomFields(mean=mean, variance=(cov*mean)**2,
                  model_name=ModelName[model_name],
                  v_scale_fluctuation=v_scale_fluctuation,
                  anisotropy=anisotropy, angle=angle, n_dim=3, seed=seed)

rf.generate(centroids)
random_field = list(rf.random_field)

# make plot
fig, ax = plt.subplots(figsize=(6, 5))
ax = fig.add_subplot(projection='3d')
ax.set_position([0.1, 0.1, 0.7, 0.8])
ax.view_init(45, -90, 0)

vmin = 3.0e07
vmax = 4.3e07

ax.set_xlabel('x coordinate')
# ax.set_ylabel('y coordinate')
ax.set_zlabel('z coordinate')

cax = ax.inset_axes([1.1, 0., 0.05, 1])
ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c=random_field[0], vmin=vmin, vmax=vmax, cmap="coolwarm",
           edgecolors=None,
           marker="o", s=4)

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, cax=cax)

# ax.set_aspect('equal')
fig.suptitle("Random field generated from Python:")
plt.savefig(os.path.join(main_folder, "random_field_from_python.png"))
plt.close()