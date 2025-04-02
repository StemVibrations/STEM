from stem.model import Model
from stem.boundary import DisplacementConstraint
from stem.load import SurfaceLoad
from stem.solver import SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    LinearNewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, Cg, NewtonRaphsonStrategy
from stem.output import NodalOutput, Output, VtkOutputParameters
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.stem import Stem
from stem.model_part import BodyModelPart
from stem.structural_material import *


ndim = 3
model = Model(ndim)
model.extrusion_length = 10.0


#DENSITY_SOLID = 2650
#POROSITY = 0.3
#YOUNG_MODULUS = 50e2
#POISSON_RATIO = 0.3
#soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
#constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
#retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
#material1 = SoilMaterial("soil_1", soil_formulation1, constitutive_law1, retention_parameters1)
## Specify the coordinates for the column: x:5m x y:1m
#layer1_coordinates = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 2.0, 0.0), (0.0, 2.0, 0.0)]


# Define boundary conditions
no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True],
                                                    value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True],
                                                        value=[0, 0, 0])
#model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_layer_1")

# add shell plate
points_shell = [(0.0, 2.0, 0.0), (10.0, 2.0, 0), (10.0, 2.0, 10.0), (0.0, 2.0, 10.0)]
geo_settings = {"shell": {"coordinates": points_shell, "ndim": 2}}
model.gmsh_io.generate_geometry(geo_settings, "")
# add middle line
geo_settings = {"line": {"coordinates": [(5, 2.0, 10.0), (5, 2.0, 0.0)], "ndim": 1}}
model.gmsh_io.generate_geometry(geo_settings, "")

shell = Shell(YOUNG_MODULUS=25e6, POISSON_RATIO=0.25, THICKNESS=0.1, DENSITY=2650)
body_model_part = BodyModelPart("shell")
body_model_part.material =  StructuralMaterial(name="shell", material_parameters=shell)

# set the geometry of the body model part
body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "shell")

model.body_model_parts.append(body_model_part)
surface_load = SurfaceLoad(active=[False, True, False], value=[0, -10, 0])
model.add_load_by_coordinates([(0,2,10), (0,2,0), (10,2,0), (10,2,10)], surface_load, "load")
model.show_geometry(show_surface_ids=True, show_line_ids=True)
# Add boundary conditions to the model (geometry ids are shown in the show_geometry)
#model.add_boundary_condition_by_geometry_ids(2, [2], no_displacement_parameters, "base_fixed")
#model.add_boundary_condition_by_geometry_ids(2, [1,3,5,6],
#                                             roller_displacement_parameters, "roller_fixed")
model.add_boundary_condition_by_geometry_ids(1, [10,17,15,8, 14, 16],
                                             no_displacement_parameters, "shell_fixed")
# Set up solver settings
analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.QUASI_STATIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0,
                                   end_time=0.5,
                                   delta_time=0.01,
                                   reduction_factor=1.0,
                                   increase_factor=1.0,
                                   max_delta_time_factor=1000)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-6,
                                                        displacement_absolute_tolerance=1.0e-12)
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type,
                                 solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False,
                                 are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=NewtonRaphsonStrategy(),
                                 linear_solver_settings=Cg(),
                                 rayleigh_k=0.00,
                                 rayleigh_m=0.0000)

# Set up problem data
problem = Problem(problem_name="test_extended_beam", number_of_threads=4, settings=solver_settings)
model.project_parameters = problem
model.set_mesh_size(element_size=0.5)
input_folder = "shell_elements/"
# Nodal results
nodal_results = [
    NodalOutput.DISPLACEMENT,
    NodalOutput.VELOCITY_X,
    NodalOutput.VELOCITY_Y,
    NodalOutput.VELOCITY_Z,
]
# Gauss point results
gauss_point_results = []
vtk_output_process = Output(part_name="shell",
                            output_name="vtk_output",
                            output_dir="output" ,
                            output_parameters=VtkOutputParameters(file_format="binary",
                                                                  output_interval=1,
                                                                  nodal_results=nodal_results,
                                                                  gauss_point_results=gauss_point_results,
                                                                  output_control_type="step"))
model.output_settings.append(vtk_output_process)
# Write KRATOS input files
# --------------------------------
stem = Stem(model, input_folder)
stem.write_all_input_files()
# open the MaterialParameters.json file and change the material parameters
import json

with open("shell_elements/MaterialParameters_stage_1.json", "r") as file:
    data = json.load(file)
    data["properties"][-1]["Material"]["constitutive_law"] = {"name": "LinearElasticPlaneStrain2DLaw"}
    data["properties"][-1]["Material"]["Variables"] = {
        "YOUNG_MODULUS": 200e9,
        "POISSON_RATIO": 0.3,
        "THICKNESS": 0.1,
        "DENSITY": 2650
    }
# save the changes
with open("shell_elements/MaterialParameters_stage_1.json", "w") as file:
    json.dump(data, file, indent=4)

stem.run_calculation()