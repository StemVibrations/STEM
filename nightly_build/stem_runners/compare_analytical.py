import json

import numpy as np
import matplotlib.pyplot as plt

from benchmark_tests.analytical_solutions.strip_load import StripLoad
from benchmark_tests.analytical_solutions.pekeris import Pekeris, LoadType
from benchmark_tests.analytical_solutions.analytical_wave_prop import OneDimWavePropagation

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

    plt.plot(data_kratos["TIME"],
             np.array(data_kratos['NODE_9']['VELOCITY_Y']) * 1000,
             color="r",
             marker="x",
             label="STEM")
    plt.plot(p.time, p.v[10, :] * 1000, color="b", label="Analytical")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [mm/s]")
    plt.grid()
    plt.xlim(0, 0.5)
    plt.ylim(-4, 4)
    plt.legend()
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

    cs = np.sqrt(young_modulus / (2 * (1 + poisson_ratio)) / density_solid)
    shear_modulus = young_modulus / (2 * (1 + poisson_ratio))

    lmb = Pekeris(tau_max=8)
    lmb.material_properties(poisson_ratio, density_solid, young_modulus)
    lmb.loading(load_value * 4, LoadType.Heaviside)
    lmb.solution(coords)
    lmb.results(output_folder="./", file_name="Heaviside", plots=False)

    fig, ax = plt.subplots(nrows=len(coords), ncols=1, figsize=(6, 10), sharex=True, sharey=True)
    for j, c in enumerate(coords):
        # find index in STEM data
        node_key = [k for k in keys_nodes if pekeris_data_kratos[k]["COORDINATES"][0] == c][0]
        ax[j].plot(np.array(pekeris_data_kratos["TIME"]),
                   np.array(pekeris_data_kratos[node_key]["DISPLACEMENT_Y"]),
                   color="r",
                   marker="x",
                   label="STEM")
        ax[j].plot(np.array(lmb.time)[:, j], np.array(lmb.u)[:, j], color="b", label="Analytical")

        ax[j].set_ylabel(f'Displacement at x={c} m [m]')
        ax[j].grid()
        ax[j].legend(loc=1)
        ax[j].text(0.5, -0.15, f'({chr(97+j)})', transform=ax[j].transAxes, ha='center', va='top', fontsize=12)

    ax[1].set_xlabel('Time [s]')
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
        ax[j].plot([i for i in x], [stress_zz_kratos[int(t / time_step)][i] / 1e3 for i in range(len(x))],
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
        ax[j].plot([i for i in x], [stress_zz_kratos[int(t / time_step)][i] / 1e3 for i in range(len(x))],
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
