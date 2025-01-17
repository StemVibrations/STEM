import sys
import os

path_kratos = os.path.dirname(os.path.abspath(__file__))
material_name = "MaterialParameters.json"
project_name = "ProjectParameters.json"
mesh_name = "calculate_moving_load_on_embankment_3d.mdpa"

sys.path.append(os.path.join(path_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(path_kratos, r"KratosGeoMechanics\libs"))


from stem.IO.kratos_io import KratosIO
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.model import Model
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (GeoMechanicsAnalysis)
import KratosMultiphysics.GeoMechanicsApplication


ndim = 2
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
model.extrusion_length = [0, 0, 20]

model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")

load_coordinates = [(0.75, 0.0, 0.0), (0.75, 3.0, 00.0)]
moving_load = MovingLoad(load=[0.0, -10.0, 0.0], direction=[1, 1, 1], velocity=5, origin=[0.75, 0.0, 0.0],
                         offset=0.0)
model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")


no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                             roller_displacement_parameters, "sides_roller")

model.synchronise_geometry()

model.show_geometry(show_surface_ids=True)

model.set_mesh_size(element_size=0.1)

model.generate_mesh()

elements = []
nodes = []
for body in model.body_model_parts:
    elements.append([[body.mesh.elements[k].id, body.mesh.elements[k].node_ids] for k in body.mesh.elements.keys()])
    nodes.append([[body.mesh.nodes[k].id, body.mesh.nodes[k].coordinates] for k in body.mesh.nodes.keys()])

with open("geometry_2D.pickle", "wb") as f:
    import pickle
    pickle.dump([elements, nodes], f)



analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
solution_type = SolutionType.QUASI_STATIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=1.0, delta_time=0.005, reduction_factor=1.0,
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
problem = Problem(problem_name="calculate_moving_load_on_embankment_3d", number_of_threads=1,
                  settings=solver_settings)
model.project_parameters = problem

nodal_results = [NodalOutput.DISPLACEMENT,
                 NodalOutput.TOTAL_DISPLACEMENT]
gauss_point_results = []

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

kratos_io = KratosIO(ndim=model.ndim)
output_folder = "inputs_kratos"


kratos_io.write_project_parameters_json(
    model=model,
    outputs=[vtk_output_process],
    mesh_file_name="calculate_moving_load_on_embankment_3d.mdpa",
    materials_file_name="MaterialParameters.json",
    output_folder=output_folder
)

kratos_io.write_mesh_to_mdpa(
    model=model,
    mesh_file_name="calculate_moving_load_on_embankment_3d.mdpa",
    output_folder=output_folder
)

kratos_io.write_material_parameters_json(
    model=model,
    output_folder=output_folder
)


project_folder = "inputs_kratos"
os.chdir(project_folder)

with open(project_name, "r") as parameter_file:
    parameters = KratosMultiphysics.Parameters(parameter_file.read())

model = KratosMultiphysics.Model()
simulation = GeoMechanicsAnalysis(model, parameters)
simulation.Run()


