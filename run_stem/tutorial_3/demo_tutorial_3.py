# ----------------------------------------------------------------------------------------------------------------------
import os
from shutil import copytree

import numpy as np

from stem.structural_material import ElasticSpringDamper, NodalConcentrated

input_files_dir = "run_stem/tutorial_3/check_1"
results_dir = "output_uvec_load"
uvec_folder = "benchmark_tests/test_train_uvec_3d/uvec_ten_dof_vehicle_2D"

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.default_materials import DefaultMaterial
from stem.load import MovingLoad, UvecLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.additional_processes import ParameterFieldParameters
from stem.field_generator import RandomFieldGenerator
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
     NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem
# ----------------------------------------------------------------------------------------------------------------------
ndim = 3
model = Model(ndim)
# ----------------------------------------------------------------------------------------------------------------------
solid_density_1 = 2650
porosity_1 = 0.3
young_modulus_1 = 30e6
poisson_ratio_1 = 0.2
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity_1)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio_1)
retention_parameters_1 = SaturatedBelowPhreaticLevelLaw()
material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, retention_parameters_1)
# ----------------------------------------------------------------------------------------------------------------------
solid_density_2 = 2550
porosity_2 = 0.3
young_modulus_2 = 30e6
poisson_ratio_2 = 0.2
soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_2, POROSITY=porosity_2)
constitutive_law_2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_2, POISSON_RATIO=poisson_ratio_2)
retention_parameters_2 = SaturatedBelowPhreaticLevelLaw()
material_soil_2 = SoilMaterial("soil_2", soil_formulation_2, constitutive_law_2, retention_parameters_2)
# ----------------------------------------------------------------------------------------------------------------------
solid_density_3 = 2650
porosity_3 = 0.3
young_modulus_3 = 10e6
poisson_ratio_3 = 0.2
soil_formulation_3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity_3)
constitutive_law_3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio_3)
retention_parameters_3 = SaturatedBelowPhreaticLevelLaw()
material_embankment = SoilMaterial("embankment", soil_formulation_3, constitutive_law_3, retention_parameters_3)
# ----------------------------------------------------------------------------------------------------------------------
# In het Nederlandse spoornet worden verschillen typen spoorstaven gebruikt. De meest voorkomende
# zijn 46 E3 (oud NP46), 54 E1 (oud UIC 54) en 60 E1 (oud UIC 60). De naamgeving van deze typen spoorstaven bestaat uit
# een afkorting + het nominale gewicht per meter spoorstaaf in kilogram.
#
# De NP46 komt vooral nog voor op emplacementen en op een aantal regionale spoorlijnen.
# De 54 E1 (in het verleden met UIC 54 aangeduid) ligt op de meeste spoorlijnen van het kernnet.
# De 60 E1 (oud UIC 60) vinden we op een aantal proefstukken, o.a. bij Deurne op de lijn Eindhoven – Venlo, de
# hogesnelheidslijn (HSL-Zuid) en het A15 tracé (Betuweroute).
#
# Buiten Nederland is de 60 E1 spoorstaaf waarschijnlijk de meest toegepaste spoorstaaf.
# In de tramwereld worden groefrails gebruikt.

# Define the material and geometry for the rails:
rail_parameters = DefaultMaterial.Rail_54E1_3D.value

# Define point spring/damper for the pad
rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 750e6, 1],
                                          NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                          NODAL_DAMPING_COEFFICIENT=[1, 750e3, 1],
                                          NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                       NODAL_MASS=140,
                                       NODAL_DAMPING_COEFFICIENT=[0, 0, 0])
rail_pad_thickness = 0.025
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0, 3.0, 0.0)]
model.extrusion_length = 50

# define the origin point of the track and the direction for the extrusion
origin_point = np.array([0.75, 3.0, 0.0])
direction_vector = np.array([0, 0, 1])
# ----------------------------------------------------------------------------------------------------------------------
model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")
# ----------------------------------------------------------------------------------------------------------------------
# create a straight track with rails, sleepers and rail pads
model.generate_straight_track(0.5, 101, rail_parameters.material_parameters, sleeper_parameters, rail_pad_parameters,
                              rail_pad_thickness, origin_point,
                              direction_vector, "rail_track_1")
# ----------------------------------------------------------------------------------------------------------------------
# Define UVEC load
load_coordinates = [(0.75, 3.0+rail_pad_thickness, 0.0), (0.75, 3.0+rail_pad_thickness, 50.0)]

uvec_parameters = {"n_carts": 1,
                   "cart_inertia": (1128.8e3) / 2,
                   "cart_mass": (50e3) / 2,
                   "cart_stiffness": 2708e3,
                   "cart_damping": 64e3,
                   "bogie_distances": [-9.95, 9.95],
                   "bogie_inertia": (0.31e3) / 2,
                   "bogie_mass": (6e3) / 2,
                   "wheel_distances": [-1.25, 1.25],
                   "wheel_mass": 1.5e3,
                   "wheel_stiffness": 4800e3,
                   "wheel_damping": 0.25e3,
                   "gravity_axis": 1,
                   "contact_coefficient": 9.1e-7,
                   "contact_power": 1,
                   "initialisation_steps": 100,
                   }

uvec_load = UvecLoad(direction=[1, 1, 1], velocity=40, origin=[0.75, 3+rail_pad_thickness, 5],
                     wheel_configuration=[0.0, 2.5, 19.9, 22.4],
                     uvec_file=r"uvec_ten_dof_vehicle_2D\uvec.py", uvec_function_name="uvec",
                     uvec_parameters=uvec_parameters)
# add the load on the tracks
model.add_load_on_line_model_part("rail_track_1", uvec_load, "train_load")
# ----------------------------------------------------------------------------------------------------------------------
model.synchronise_geometry()

# model.show_geometry(show_surface_ids=True)
# ----------------------------------------------------------------------------------------------------------------------
no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])

# Define absorbing boundary condition
absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

# Add boundary conditions to the model (geometry ids are shown in the show_geometry)
model.add_boundary_condition_by_geometry_ids(2, [2, 5, 6, 7, 11, 12, 15, 16, 17], absorbing_boundaries_parameters, "abs")

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [4, 10], roller_displacement_parameters, "sides_roller")
# ----------------------------------------------------------------------------------------------------------------------
# Define the field generator
# random_field_generator = RandomFieldGenerator(
#     n_dim=3, cov=0.1, v_scale_fluctuation=1,
#     anisotropy=[20.0], angle=[0],
#     model_name="Gaussian", seed=14
# )
#
# field_parameters_json = ParameterFieldParameters(
#     property_name="YOUNG_MODULUS",
#     function_type="json_file",
#     field_generator=random_field_generator
# )
# # add the random field to the model
# model.add_field(part_name="soil_layer_2", field_parameters=field_parameters_json)
# ----------------------------------------------------------------------------------------------------------------------
model.set_mesh_size(element_size=1.0)
# ----------------------------------------------------------------------------------------------------------------------
# analysis_type = AnalysisType.MECHANICAL
# solution_type = SolutionType.DYNAMIC
analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.DYNAMIC
# Set up start and end time of calculation, time step and etc
end_time = 0.5
# nsteps = 3000
delta_time = 1e-03
# 30m/s -> 50/30 = 1.67 seconds
nsteps = int(end_time/delta_time)
time_integration = TimeIntegration(start_time=0.0, end_time=end_time, delta_time=delta_time,
                                   reduction_factor=1, increase_factor=1, max_delta_time_factor=1000)

convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-12)
strategy = NewtonRaphsonStrategy(max_iterations=50)
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                 convergence_criteria=convergence_criterion, strategy_type=strategy,
                                 rayleigh_k=0.0001,
                                 rayleigh_m=0.01)
# ----------------------------------------------------------------------------------------------------------------------
# Set up problem data
problem = Problem(problem_name="calculate_uvec_on_embankment_random_fields_and_abs_boundaries",
                  number_of_threads=8,
                  settings=solver_settings)
model.project_parameters = problem
# ----------------------------------------------------------------------------------------------------------------------
nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]
gauss_point_results = []
# ----------------------------------------------------------------------------------------------------------------------
# Define the output process
output_fs = 100  # hz
output_step = int(np.round(1/delta_time/output_fs))

model.add_output_settings(
    part_name="porous_computational_model_part",
    output_dir=results_dir,
    output_name="vtk_output",
    output_parameters=VtkOutputParameters(
        file_format="ascii",
        output_interval=output_step,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
    )
)

# ----------------------------------------------------------------------------------------------------------------------
desired_output_points = [
    (0.0, 3.0, 25.0), (0.75, 3.0, 25.0), (1.5, 3.0, 25.0),
    (3, 2.0, 25.0), (4, 2.0, 25.0),
    (5, 2.0, 25.0)
]
output_fs = 1000  # hz
output_dt = 1/output_fs

model.add_output_settings_by_coordinates(
    part_name="subset_outputs",
    output_dir=results_dir,
    output_name="json_output",
    coordinates=desired_output_points,
    output_parameters=JsonOutputParameters(
        output_interval=output_dt-1e-10,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results
    )
)
model.synchronise_geometry()

# ----------------------------------------------------------------------------------------------------------------------
# copy uvec to input folder
os.makedirs(input_files_dir, exist_ok=True)
copytree(
    uvec_folder,
    os.path.join(input_files_dir, "uvec_ten_dof_vehicle_2D"),
    dirs_exist_ok=True
)
# ----------------------------------------------------------------------------------------------------------------------
stem = Stem(model, input_files_dir)

# ----------------------------------------------------------------------------------------------------------------------
stem.write_all_input_files()

# ----------------------------------------------------------------------------------------------------------------------
stem.run_calculation()

# ----------------------------------------------------------------------------------------------------------------------
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open(
        os.path.join(input_files_dir, results_dir,"json_output.json"), "r"
) as outfile:
    point_outputs = json.load(outfile)

otp_nodes = list(point_outputs.keys())[1:]
time = point_outputs["TIME"]
dt = np.mean(np.diff(time))

fig, ax = plt.subplots(3, 1, sharex="all", sharey="all")
comp = "DISPLACEMENT"
unit = " [m]"
for otp in otp_nodes:

    for ii, res in enumerate(["X", "Y", "Z"]):
        ax[ii].plot(time, point_outputs[otp][comp + "_" + res])
        ax[ii].set_ylabel(comp + "_" + res + unit)

plt.xlabel("time [s]")

fig, ax = plt.subplots(3, 1, sharex="all", sharey="all")
comp = "VELOCITY"
unit = " [m]"
for otp in otp_nodes:

    for ii, res in enumerate(["X", "Y", "Z"]):

        x = point_outputs[otp][comp + "_" + res]

        # freq =
        xf = np.fft.rfft(x, n=2**12)
        freq = np.fft.rfftfreq(n=2**12)*1/dt
        # ax[ii].plot(time, x)
        ax[ii].plot(freq, np.abs(xf))
        ax[ii].set_ylabel(comp + "_" + res + unit)
        ax[ii].set_yscale('log')

plt.xlabel("frequency [s]")


fig, ax = plt.subplots(3, 1, sharex="all", sharey="all")
comp = "VELOCITY"
unit = " [m/s]"
for otp in otp_nodes:

    for ii, res in enumerate(["X", "Y", "Z"]):

        x = point_outputs[otp][comp + "_" + res]

        # freq =
        # xf = np.fft.rfft(x, n=2**10)
        # freq = np.fft.rfftfreq(n=2**10)*1/dt
        ax[ii].plot(time, x)
        # ax[ii].plot(freq, np.abs(xf))
        ax[ii].set_ylabel(comp + "_" + res + unit)
        # ax[ii].set_yscale('log')

plt.xlabel("time [s]")

# fig, ax = plt.subplots(3, 1)
# comp = "ACCELERATION"
# unit = " [m/s^2]"
# for otp in otp_nodes:
#
#     for ii, res in enumerate(["X", "Y", "Z"]):
#
#         x = point_outputs[otp][comp + "_" + res]
#
#         # freq =
#         # xf = np.fft.rfft(x, n=2**10)
#         # freq = np.fft.rfftfreq(n=2**10)*1/dt
#         ax[ii].plot(time, x)
#         # ax[ii].plot(freq, np.abs(xf))
#         ax[ii].set_ylabel(comp + "_" + res + unit)
#         # ax[ii].set_yscale('log')


plt.show()