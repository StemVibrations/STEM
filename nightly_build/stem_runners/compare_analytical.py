import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from benchmark_tests.analytical_solutions.strip_load import StripLoad
from benchmark_tests.analytical_solutions.pekeris import Pekeris, LoadType
from benchmark_tests.analytical_solutions.analytical_wave_prop import OneDimWavePropagation
from benchmark_tests.analytical_solutions.linear_spring_damper_mass import LinearSpringDamperMass

# from benchmark_tests.analytical_solutions.point_load_moving import MovingLoadElasticHalfSpace

import nightly_build.stem_runners.read_VTK as read_VTK


def compare_wave_propagation(path_model, output_file):

    # Based on: benchmark_tests/test_1d_wave_prop_drained_soil_3D/test_1d_wave_prop_drained_soil_3d.py
    # with:
    # - model.set_mesh_size(element_size=0.15)
    # - model.mesh_settings.element_order = 2
    # - rayleigh_k=3.929751681281367e-05
    # - rayleigh_m=0.12411230236404121

    # load data from STEM
    with open(path_model, "r") as f:
        data_kratos = json.load(f)

    young_modulus = 50e6  # Pa
    poisson_ratio = 0.3
    density_solid = 2700  # kg/m3
    porosity = 0.3
    load_value = -1e3  # Pa
    lenght = 10  # m
    nb_elements = 20

    p_modulus = (young_modulus * (1 - poisson_ratio)) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    p = OneDimWavePropagation(nb_terms=100)
    p.properties(density_solid * (1 - porosity), p_modulus, load_value, lenght, nb_elements)
    p.solution()
    p.write_results()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)
    ax.plot(p.time, p.v[10, :] * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax.plot(data_kratos["TIME"], np.array(data_kratos['NODE_9']['VELOCITY_Y']) * 1000, color="b", label="STEM")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [mm/s]")
    ax.grid()
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-4, 4)
    ax.legend(loc='upper right')
    plt.savefig(output_file)
    plt.close()


def compare_pekeris(path_model, output_file):
    # load Pekeris data from STEM
    with open(path_model, "r") as f:
        pekeris_data_kratos = json.load(f)

    keys_nodes = [k for k in pekeris_data_kratos.keys() if k.startswith("NODE_")]

    young_modulus = 30e6  # Pa
    poisson_ratio = 0.2
    density_solid = 2000  # kg/m3
    load_value = -1e6  # Pa
    coords = [1, 2, 3]

    lmb = Pekeris(tau_max=8)
    lmb.material_properties(poisson_ratio, density_solid, young_modulus)
    lmb.loading(load_value * 4, LoadType.Heaviside)
    lmb.solution(coords)
    lmb.results(output_folder="./", file_name="Heaviside", plots=False)

    fig, ax = plt.subplots(nrows=len(coords), ncols=1, figsize=(6, 10), sharex=True, sharey=True)
    for j, c in enumerate(coords):
        # find index in STEM data
        node_key = [k for k in keys_nodes if pekeris_data_kratos[k]["COORDINATES"][0] == c][0]
        ax[j].plot(np.array(lmb.time)[:, j],
                   np.array(lmb.u)[:, j],
                   color="r",
                   label="Analytical",
                   marker="x",
                   markevery=10)
        ax[j].plot(np.array(pekeris_data_kratos["TIME"]),
                   np.array(pekeris_data_kratos[node_key]["DISPLACEMENT_Y"]),
                   color="b",
                   label="STEM")
        ax[j].set_ylabel(f'Displacement at x={c} m [m]')
        ax[j].grid()
        ax[j].legend(loc=1)
        ax[j].text(0.5, -0.15, f'({chr(97+j)})', transform=ax[j].transAxes, ha='center', va='top', fontsize=12)

    ax[2].set_xlabel('Time [s]')
    ax[1].set_xlim(0, 0.08)
    ax[1].set_ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def compare_strip_load_2D(path_model, output_file):

    # based on: benchmark_tests/test_strip_load_2D/test_strip_load_2D.py
    # with:
    # - model.set_mesh_size(element_size=0.15)
    # - element_order=2

    coordinate_x_coords = np.linspace(0, 20, 41)
    coordinate_y_coords = np.ones(len(coordinate_x_coords)) * 9
    coordinate_z_coords = np.zeros(len(coordinate_x_coords))
    coordinate_points = np.array([coordinate_x_coords, coordinate_y_coords, coordinate_z_coords]).T.tolist()

    read_VTK.read(path_model, coordinate_points, "CAUCHY_STRESS_VECTOR", "results.json")

    with open("results.json", "r") as f:
        data = json.load(f)

    time_step = 0.001

    x = [coord[0] for coord in data["COORDINATES"]]
    stress_zz_kratos = []
    for i in range(len(data["DATA"])):
        aux = [data_point[1] for data_point in data["DATA"][i]]
        stress_zz_kratos.append(aux)

    young_modulus = 30e6  # Pa
    poisson_ratio = 0.2
    density_solid = 2000  # kg/m3
    porosity = 0
    load_value = 1e6  # Pa
    line_load_length = 1  # m

    strip_load = StripLoad(young_modulus, poisson_ratio, (1 - porosity) * density_solid, load_value)

    # x coordinates
    start_x = 0
    end_x = 20
    n_steps = 300
    x_coordinates = [start_x + (end_x - start_x) * i / n_steps for i in range(n_steps)]

    # depth coordinate
    depth = 1 * line_load_length

    # calculate vertical stress at different x coordinates at different times
    t_calc = [0.05, 0.075, 0.1]
    fig, ax = plt.subplots(nrows=len(t_calc), ncols=1, figsize=(6, 10), sharex=True, sharey=True)
    for j, t in enumerate(t_calc):
        all_sigma_zz = [
            strip_load.calculate_vertical_stress(x, depth, t, line_load_length, load_value) for x in x_coordinates
        ]

        # plot vertical stress: Figure 12.14 in Verruijt
        idx_k = np.where(t == np.array(data['TIME_INDEX']).astype(float) * time_step)[0][0]
        ax[j].plot([i for i in x], [stress_zz_kratos[idx_k][i] / 1e3 for i in range(len(x))],
                   color="r",
                   marker="x",
                   label="STEM")
        ax[j].plot([i for i in x_coordinates], [i / 1e3 for i in all_sigma_zz], color="b", label="Analytical")
        ax[j].set_ylabel(f'Vertical stress at t={t} s [kPa]')
        ax[j].grid()
        ax[j].legend()

    ax[2].set_xlabel('Distance [m]')
    ax[2].set_xlim(0, 20)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def compare_strip_load_3D(path_model, output_file):

    # based on: benchmark_tests/test_strip_load_3D/test_strip_load_3D.py
    # with:
    # - model.set_mesh_size(element_size=0.15)
    # - element_order=2

    coordinate_x_coords = np.linspace(0, 20, 41)
    coordinate_y_coords = np.ones(len(coordinate_x_coords)) * 9
    coordinate_z_coords = np.zeros(len(coordinate_x_coords))
    coordinate_points = np.array([coordinate_x_coords, coordinate_y_coords, coordinate_z_coords]).T.tolist()

    read_VTK.read(path_model, coordinate_points, "CAUCHY_STRESS_VECTOR", "results.json")

    with open("results.json", "r") as f:
        data = json.load(f)

    time_step = 0.001

    x = [coord[0] for coord in data["COORDINATES"]]
    time = np.array(data['STEP']) * time_step
    stress_zz_kratos = []
    for i in range(len(data["DATA"])):
        aux = [data_point[1] for data_point in data["DATA"][i]]
        stress_zz_kratos.append(aux)

    young_modulus = 30e6  # Pa
    poisson_ratio = 0.2
    density_solid = 2000  # kg/m3
    porosity = 0
    load_value = 1e6  # Pa
    line_load_length = 1  # m

    strip_load = StripLoad(young_modulus, poisson_ratio, (1 - porosity) * density_solid, load_value)

    # time to calculate the vertical stress at
    end_time = time[-1]

    # x coordinates
    start_x = 0
    end_x = 20
    n_steps = 300
    x_coordinates = [start_x + (end_x - start_x) * i / n_steps for i in range(n_steps)]

    # depth coordinate
    depth = 1 * line_load_length

    # calculate vertical stress at different x coordinates at different times

    t_calc = [0.05, 0.075, 0.1]
    fig, ax = plt.subplots(nrows=len(t_calc), ncols=1, figsize=(6, 10), sharex=True, sharey=True)
    for j, t in enumerate(t_calc):
        all_sigma_zz = [
            strip_load.calculate_vertical_stress(x, depth, t, line_load_length, load_value) for x in x_coordinates
        ]

        # plot vertical stress: Figure 12.14 in Verruijt
        idx_k = np.where(t == np.array(data['TIME_INDEX']).astype(float) * time_step)[0][0]
        ax[j].plot([i for i in x], [stress_zz_kratos[idx_k][i] / 1e3 for i in range(len(x))],
                   color="r",
                   marker="x",
                   label="STEM")
        ax[j].plot([i for i in x_coordinates], [i / 1e3 for i in all_sigma_zz], color="b", label="Analytical")
        ax[j].set_ylabel(f'Vertical stress at t={t} s [kPa]')
        ax[j].grid()
        ax[j].legend()

    ax[1].set_xlabel('Distance [m]')
    ax[1].set_xlim(0, 20)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def compare_sdof(path_model, output_file):

    # load data from STEM
    with open(path_model, "r") as f:
        data_kratos = json.load(f)

    # calculate spring damper mass system analytically
    end_time = 1
    nsteps = 1000
    analytical_solution = LinearSpringDamperMass(k=10000, c=100, m=10, g=9.81, end_time=end_time, n_steps=nsteps)

    analytical_solution.solve()

    # start at 0 displacement
    amplitude = analytical_solution.displacement[0]
    analytical_solution.displacement -= amplitude

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)
    ax.plot(analytical_solution.time,
            analytical_solution.displacement * 1000,
            label="Analytical",
            marker="x",
            color='r',
            markevery=25)
    ax.plot(data_kratos["TIME"], np.array(data_kratos["NODE_2"]["DISPLACEMENT_Y"]) * 1000, label="STEM", color='b')
    ax.grid()
    ax.set_xlim(0, end_time)
    ax.set_ylim(-20, 5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [mm]")
    ax.legend(loc='upper right')
    plt.savefig(output_file)
    plt.close()


def compare_moving_load(path_model, output_file):

    # Based on:
    # with:
    # - model.set_mesh_size(element_size=0.15)
    # - model.mesh_settings.element_order = 2

    # load data from STEM
    with open(path_model, "r") as f:
        data_kratos = json.load(f)

    plt.plot(data_kratos["TIME"], data_kratos['NODE_13']['DISPLACEMENT_Y'], color="r", marker="x", label="STEM")
    # plt.show()


def compare_vibrating_dam(path_model, output_file):

    # load data from STEM
    with open(path_model, "r") as f:
        data_kratos = json.load(f)

    feet_to_m = 0.3048
    shear_wave_velocity = 1200 * feet_to_m  # ft/s to m/s
    y_max = 150 * feet_to_m

    time_step = data_kratos["TIME"][1] - data_kratos["TIME"][0]
    calculated_horizontal_displacement = data_kratos["NODE_2"]["DISPLACEMENT_X"]
    f, Pxx = welch(calculated_horizontal_displacement,
                   fs=1 / time_step,
                   nfft=20000,
                   nperseg=len(calculated_horizontal_displacement))

    # the beta values follow from literature for the shear beam natural frequencies
    betas = [2.404, 5.520, 8.654, 11.792, 14.931]
    expected_natural_frequencies = [shear_wave_velocity / y_max * beta / (2 * np.pi) for beta in betas]

    # plot PSD
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)
    ax.plot(f, Pxx)

    # plot expected natural frequencies
    for i, expected_natural_frequency in enumerate(expected_natural_frequencies):
        ax.axvline(expected_natural_frequency,
                   color='g',
                   linestyle='--',
                   label=f'Expected natural frequency' if i == 0 else None)

    ax.set_xlim(0, 20)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [m^2/Hz]')
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()


def compare_abs_boundary(path_model, output_file):

    # load data from STEM
    with open(path_model, "r") as f:
        data_kratos = json.load(f)

    young_modulus = 50e6  # Pa
    poisson_ratio = 0.3
    density_solid = 2700  # kg/m3
    porosity = 0.3
    load_value = -1e3  # Pa
    lenght = 10  # m
    nb_elements = 20

    p_modulus = (young_modulus * (1 - poisson_ratio)) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    # firstly calculate without absorption
    p = OneDimWavePropagation(nb_terms=100)
    p.properties(density_solid * (1 - porosity), p_modulus, load_value, lenght, nb_elements)
    p.solution()

    # add absorption by keeping velocity constant after wave arrival time
    vp = np.sqrt(p_modulus / (density_solid * (1 - porosity)))
    wave_arrival_time = lenght / vp

    idx = np.argmax(p.time > wave_arrival_time)
    p.v[:, idx:] = p.v[:, idx][:, None]  # velocity remains constant after wave arrival time (to simulate absorption)
    p.u[:,
        idx:] = p.u[:,
                    idx][:,
                         None] + p.v[:, idx][:, None] * (p.time[idx:] - p.time[idx])  # displacement continues linearly
    p.write_results()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)
    ax.plot(p.time, p.v[10, :] * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax.plot(data_kratos["TIME"], np.array(data_kratos['NODE_9']['VELOCITY_Y']) * 1000, color="b", label="STEM")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [mm/s]")
    ax.grid()
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-4, 4)
    ax.legend(loc='upper right')
    plt.savefig(output_file)
    plt.close()


#     E = 30e6  # Pa
#     nu = 0.2  # dimensionless
#     rho = 2000  # kg/m³
#     force = -1e3 * 2 # N
#     speed = 10  # m/s

#     x, y, z = 0.0, 0.0, 1
#     time = np.linspace(-0.1, 0.1, num=100)

#     model = MovingLoadElasticHalfSpace(E, nu, rho, force, speed)

#     uz = []
#     for t in time:
#         model.compute_vertical_displacement(x, y, z, t, ky_max=10.0, n_ky=1000)
#         uz.append(np.real(model.vertical_displacement))

#     plt.plot(time, np.array(uz), color="b", label="Analytical")
#     plt.xlabel("Distance [m]")
#     plt.ylabel("Displacement [m]")
#     plt.grid()
#     plt.legend()
#     plt.show()
#     print(1)

#     # nodes = [k for k in data_kratos.keys() if k.startswith("NODE_")]
# # get point coordinated z = 5
# coordinates = []
# for n in nodes:
#     coordinates.append(data_kratos[n]["COORDINATES"][2])
#     if data_kratos[n]["COORDINATES"][2] == 5:
#         node_z5 = n

# idx_peak = np.argmax(np.abs(data_kratos[node_z5]['DISPLACEMENT_Y']))
# displacement_peak_time = [data_kratos[n]["DISPLACEMENT_Y"][idx_peak] for n in nodes]

# E = 30e6  # Pa
# nu = 0.25  # dimensionless
# rho = 2000  # kg/m³
# force = -1000  # N
# speed = 40  # m/s

# cp = np.sqrt(E / (rho * (1 - nu**2)))  # P-wave speed
# cs = cp * np.sqrt((1 - 2 * nu) / (2 * (1 - nu)))  # S-wave speed

# t_vals = np.linspace(-0.03, 0.03, 300)
# uz_vals = []

# for t in t_vals:
#     uz_vals.append(uz_moving_load(0, 5, 0, t, 1000, cp, cs, rho, speed))

# # x_list = np.linspace(-5, 5, 100)
# # Using z=0.01m to avoid the singularity exactly at the load point
# # disp = [model.vertical_displacement(x, z=0.0001) for x in x_list]

# plt.plot(coordinates, np.array(displacement_peak_time) * 1000, color="r", marker="x", label="STEM")
# plt.plot(speed * t_vals, np.array(uz_vals) * 1000, color="b", label="Analytical")
# plt.xlabel("Distance [m]")
# plt.ylabel("Displacement [mm]")
# plt.grid()
# # plt.xlim(0, 0.5)
# # plt.ylim(-4, 4)
# plt.legend()
# plt.savefig(output_file)
# plt.close()
