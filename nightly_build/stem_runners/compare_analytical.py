import json
from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from benchmark_tests.analytical_solutions.strip_load import StripLoad
from benchmark_tests.analytical_solutions.pekeris import Pekeris, LoadType
from benchmark_tests.analytical_solutions.analytical_wave_prop import OneDimWavePropagation
from benchmark_tests.analytical_solutions.linear_spring_damper_mass import LinearSpringDamperMass
from benchmark_tests.analytical_solutions.boussinesq import Boussinesq
from benchmark_tests.analytical_solutions.wave_in_infinite_pile import InfinitePileWaveSolution
from benchmark_tests.analytical_solutions.moving_load_on_beam import BeamMovingLoadAnalytical
from benchmark_tests.analytical_solutions.point_load_moving import MovingLoadElasticHalfSpace


def compare_wave_propagation(path_model: Path, output_file: Path):
    """
    Compare the analytical solution of wave propagation in a 1D column with the results from STEM.
    The STEM results are performed for 2D and 3D.

    Args:
        - path_model (Path): Path to the folder containing the JSON files with the results from STEM.
        - output_file (Path): Path to save the comparison plot.
    """

    # load data from STEM
    path_2d_results = path_model / "calculated_output_2.json"
    path_3d_results = path_model / "calculated_output_3.json"

    with open(path_2d_results, "r") as f:
        data_kratos_2d = json.load(f)

    with open(path_3d_results, "r") as f:
        data_kratos_3d = json.load(f)

    young_modulus = 50e6  # Pa
    poisson_ratio = 0.3
    density_solid = 2000  # kg/m3
    porosity = 0
    load_value = -1e3  # Pa
    lenght = 10  # m
    nb_elements = 20

    p_modulus = (young_modulus * (1 - poisson_ratio)) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    p = OneDimWavePropagation(nb_terms=100)
    p.properties(density_solid * (1 - porosity), p_modulus, load_value, lenght, nb_elements)
    p.solution()
    p.write_results()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True, sharey=True)
    ax[0].plot(p.time, p.v[5, :] * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax[0].plot(data_kratos_2d["TIME"],
               np.array(data_kratos_2d["NODE_5"]['VELOCITY_Y']) * 1000,
               color="b",
               marker="o",
               markersize=3,
               markevery=5,
               label="STEM 2D")
    ax[0].plot(data_kratos_3d["TIME"],
               np.array(data_kratos_3d["NODE_9"]['VELOCITY_Y']) * 1000,
               color="orange",
               linestyle="-.",
               label="STEM 3D")
    ax[0].text(0.5, -0.05, '(a)', transform=ax[0].transAxes, ha='center', va='top', fontsize=12)

    ax[1].plot(p.time, p.v[10, :] * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax[1].plot(data_kratos_2d["TIME"],
               np.array(data_kratos_2d["NODE_6"]['VELOCITY_Y']) * 1000,
               color="b",
               marker="o",
               markersize=3,
               markevery=5,
               label="STEM 2D")
    ax[1].plot(data_kratos_3d["TIME"],
               np.array(data_kratos_3d["NODE_10"]['VELOCITY_Y']) * 1000,
               color="orange",
               linestyle="-.",
               label="STEM 3D")
    ax[1].text(0.5, -0.075, '(b)', transform=ax[1].transAxes, ha='center', va='top', fontsize=12)

    ax[2].plot(p.time, p.v[15, :] * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax[2].plot(data_kratos_2d["TIME"],
               np.array(data_kratos_2d["NODE_7"]['VELOCITY_Y']) * 1000,
               color="b",
               marker="o",
               markersize=3,
               markevery=5,
               label="STEM 2D")
    ax[2].plot(data_kratos_3d["TIME"],
               np.array(data_kratos_3d["NODE_11"]['VELOCITY_Y']) * 1000,
               color="orange",
               linestyle="-.",
               label="STEM 3D")
    ax[2].text(0.5, -0.20, '(c)', transform=ax[2].transAxes, ha='center', va='top', fontsize=12)

    ax[0].set_ylabel("Velocity at y=2.5m [mm/s]")
    ax[1].set_ylabel("Velocity at y=5m [mm/s]")
    ax[2].set_ylabel("Velocity at y=7.5m [mm/s]")
    ax[2].set_xlabel("Time [s]")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_xlim(0, 0.5)
    ax[0].set_ylim(-4, 4)
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
    plt.savefig(output_file)
    plt.close()


def compare_pekeris(path_model: Path, output_file: Path):
    """
    Compare the analytical solution of the Lamb problem (Pekeris solution) with the results from STEM.

    Args:
        - path_model (Path): Path to the JSON file containing the results from STEM.
        - output_file (Path): Path to save the comparison plot.
    """
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


def compare_strip_load(path_model: List[Path], output_file: Path):
    """
    Compare the analytical solution of the strip load problem with the results from STEM.

    Args:
        - path_model (List[Path]): List of paths to the JSON files containing the results from STEM.
        - output_file (Path): Path to save the comparison plot.
    """

    # load data from STEM
    path_2d_results = path_model[0] / "json_output.json"
    path_3d_results = path_model[1] / "json_output.json"

    with open(path_2d_results, "r") as f:
        data_kratos_2d = json.load(f)

    with open(path_3d_results, "r") as f:
        data_kratos_3d = json.load(f)

    # get data 2D
    coords_2d = []
    for n in data_kratos_2d.keys():
        if n.startswith("NODE_"):
            coords_2d.append((data_kratos_2d[n]["COORDINATES"][0], n))

    stress_2d = []
    for t in [1, 2, 3]:
        aux = []
        dist_2D = []
        for c, n in coords_2d:
            dist_2D.append(c)
            aux.append(data_kratos_2d[n]['CAUCHY_STRESS_VECTOR'][t][1])
        stress_2d.append(aux)

    # index of sorted dist_2D
    sorted_indices = np.argsort(dist_2D)
    dist_2D = [dist_2D[i] for i in sorted_indices]
    for i in range(len(stress_2d)):
        stress_2d[i] = [stress_2d[i][j] for j in sorted_indices]

    # get data 3D
    coords_3d = []
    for n in data_kratos_3d.keys():
        if n.startswith("NODE_"):
            coords_3d.append((data_kratos_3d[n]["COORDINATES"][0], n))

    stress_3d = []
    for t in [1, 2, 3]:
        aux = []
        dist_3D = []
        for c, n in coords_3d:
            dist_3D.append(c)
            aux.append(data_kratos_3d[n]['CAUCHY_STRESS_VECTOR'][t][1])
        stress_3d.append(aux)

    # index of sorted dist_3D
    sorted_indices = np.argsort(dist_3D)
    dist_3D = [dist_3D[i] for i in sorted_indices]
    for i in range(len(stress_3d)):
        stress_3d[i] = [stress_3d[i][j] for j in sorted_indices]

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

        ax[j].plot([i for i in x_coordinates], [i / 1e3 for i in all_sigma_zz],
                   color="r",
                   marker="x",
                   markevery=10,
                   label="Analytical")

        # plot vertical stress: Figure 12.14 in Verruijt
        ax[j].plot(dist_2D, [i / 1e3 for i in stress_2d[j]],
                   color="b",
                   marker="o",
                   markersize=3,
                   markevery=5,
                   label="STEM 2D")

        ax[j].plot(dist_3D, [i / 1e3 for i in stress_3d[j]], color="orange", linestyle="-.", label="STEM 3D")

        ax[j].set_ylabel(f'Vertical stress at t={t} s [kPa]')
        ax[j].grid()
        ax[j].legend()

    ax[2].set_xlabel('Distance [m]')
    ax[2].set_xlim(0, 20)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def compare_sdof(path_model: Path, output_file: Path):
    """
    Compare the analytical solution of a SDOF system with the results from STEM.

    Args:
        - path_model (Path): Path to the JSON file containing the results from STEM.
        - output_file (Path): Path to save the comparison plot.
    """
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


def compare_moving_load(path_model: Path, output_file: Path):
    """
    Compare the analytical solution of a moving load on an elastic half-space with the results from STEM.

    Args:
        - path_model (Path): Path to the JSON file containing the results from STEM.
        - output_file (Path): Path to save the comparison plot.
    """

    # load data from STEM
    with open(path_model, "r") as f:
        data_kratos = json.load(f)

    plt.plot(np.array(data_kratos["TIME"]) - data_kratos["TIME"][0],
             np.array(data_kratos['NODE_16']['DISPLACEMENT_Y']) * 1000,
             color="r",
             marker="x",
             markevery=5,
             label="STEM")

    # Example usage
    E = 30e6  # Pa
    nu = 0.2  # dimensionless
    rho = 2000  # kg/m³
    force = -2e3  # N  (factor 2 due to symmetry BC in STEM)
    speed = 10  # m/s

    x, y, z = 0.0, 0.0, 1
    time = np.linspace(-1, 1, num=100)

    model = MovingLoadElasticHalfSpace(E, nu, rho, force, speed)
    uz = []
    for t in time:
        print(f"Calculating vertical displacement at time {t:.3f} s of {time[-1]:.3f} s", end="\r")
        model.compute_vertical_displacement(x, y, z, t, ky_max=200.0, n_ky=2000)
        uz.append(np.real(model.vertical_displacement))

    idx_kratos = np.argmin(data_kratos['NODE_16']['DISPLACEMENT_Y'])
    plt.plot(time + np.array(data_kratos["TIME"])[idx_kratos] - data_kratos["TIME"][0], (np.array(uz) - uz[0]) * 1000,
             label="Analytical")

    plt.xlabel("Time [s]")
    plt.ylabel("Vertical displacement [mm]")
    plt.legend()
    plt.grid()
    plt.xlim(0, 1.4)
    plt.savefig(output_file)
    plt.close()


def compare_boussinesq(path_model: Path, output_file: Path):
    """
    Compare the analytical solution of the Boussinesq problem with the results from STEM.

    Args:
        - path_model (Path): Path to the folder containing the JSON files with the results from STEM.
        - output_file (Path): Path to save the comparison plot.
    """

    young_modulus = 20e6  # Pa
    poisson_ratio = 0.3
    load_radius = 0.1  # m
    load_value = -10e3  # Pa

    y_max = 30  # m

    output_vert_displacements_surface_file_name = Path(path_model) / "json_output_surface.json"
    output_vert_stresses_depth_file_name = Path(path_model) / "json_output_depth.json"

    with open(output_vert_displacements_surface_file_name, 'r') as f:
        output_vert_displacements_surface = json.load(f)

    with open(output_vert_stresses_depth_file_name, 'r') as f:
        output_vert_stresses_depth = json.load(f)

    analytical_sol = Boussinesq(young_modulus, poisson_ratio, load_radius, load_value)
    calculated_x_coordinates_surface = []
    calculated_disp_results = []
    analytical_disp_results = []
    for key, value in output_vert_displacements_surface.items():
        if key != "TIME":
            calculated_x_coordinates_surface.append(value["COORDINATES"][0])
            calculated_disp_results.append(value["DISPLACEMENT_Y"][0])
            analytical_disp_results.append(
                analytical_sol.calculate_vertical_displacement_on_surface(value["COORDINATES"][0]))

    sort_idx = np.argsort(calculated_x_coordinates_surface)
    calculated_x_coordinates_surface_sorted = np.array(calculated_x_coordinates_surface)[sort_idx]
    calculated_disp_results_sorted = np.array(calculated_disp_results)[sort_idx]
    analytical_disp_results_sorted = np.array(analytical_disp_results)[sort_idx]

    calculated_y_coordinates = []
    calculated_stress_results = []
    analytical_stress_results = []
    for key, value in output_vert_stresses_depth.items():
        if key != "TIME":
            calculated_y_coordinates.append(value["COORDINATES"][1])
            calculated_stress = value["CAUCHY_STRESS_VECTOR"][0][1]
            calculated_stress_results.append(calculated_stress)
            depth = y_max - value["COORDINATES"][1]
            analytical_stress_results.append(analytical_sol.calculate_vertical_stress_below_load_centre(depth))

    sort_idx = np.argsort(calculated_y_coordinates)
    calculated_y_coordinates_sorted = np.array(calculated_y_coordinates)[sort_idx]
    calculated_stress_results_sorted = np.array(calculated_stress_results)[sort_idx]
    analytical_stress_results_sorted = np.array(analytical_stress_results)[sort_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Plot displacement results
    m_to_mm = 1e3

    ax1.plot(calculated_x_coordinates_surface_sorted,
             analytical_disp_results_sorted * m_to_mm,
             'x-',
             color='r',
             label='Analytical')

    ax1.plot(calculated_x_coordinates_surface_sorted,
             calculated_disp_results_sorted * m_to_mm,
             '--',
             color='b',
             label='STEM')
    ax1.set_xlabel('X-Coordinate [m]')
    ax1.set_ylabel('Vertical displacement along surface [mm]')
    # ax1.set_title('Vertical Displacement at Surface')
    ax1.grid()
    ax1.legend()

    pa_to_kpa = 1e-3
    ax2.plot(analytical_stress_results_sorted * pa_to_kpa,
             calculated_y_coordinates_sorted,
             'x-',
             color='r',
             label='Analytical')

    ax2.plot(calculated_stress_results_sorted * pa_to_kpa,
             calculated_y_coordinates_sorted,
             '--',
             color='b',
             label='STEM')

    ax2.set_xlabel('Vertical stress below load centre [kPa]')
    ax2.set_ylabel('Y-Coordinate [m]')
    # ax2.set_title('Vertical Stress Below Load')
    ax2.grid()
    ax2.legend()

    fig.tight_layout()

    plt.savefig(output_file)
    plt.close()


def compare_vibrating_dam(path_model: List[Path], output_file: Path):
    """
    Compare the results of the vibrating dam simulation between 2D and 3D models.

    Args:
        path_model (List[Path]): List of paths to the JSON files with the results from STEM.
        output_file (Path): Path to save the comparison plot.
    """
    # load data from STEM
    with open(path_model[0], "r") as f:
        data_kratos_2D = json.load(f)

    with open(path_model[1], "r") as f:
        data_kratos_3D = json.load(f)

    feet_to_m = 0.3048
    shear_wave_velocity = 1200 * feet_to_m  # ft/s to m/s
    y_max = 150 * feet_to_m

    time_step = data_kratos_2D["TIME"][1] - data_kratos_2D["TIME"][0]
    calculated_horizontal_displacement = np.array(data_kratos_2D["NODE_2"]["DISPLACEMENT_X"])
    f, Pxx_2D = welch(calculated_horizontal_displacement,
                      fs=1 / time_step,
                      nfft=20000,
                      nperseg=len(calculated_horizontal_displacement))

    time_step = data_kratos_3D["TIME"][1] - data_kratos_3D["TIME"][0]
    calculated_horizontal_displacement = np.array(data_kratos_3D["NODE_2"]["DISPLACEMENT_X"])
    f, Pxx_3D = welch(calculated_horizontal_displacement,
                      fs=1 / time_step,
                      nfft=20000,
                      nperseg=len(calculated_horizontal_displacement))

    # the beta values follow from literature for the shear beam natural frequencies
    betas = [2.404, 5.520, 8.654, 11.792, 14.931]
    expected_natural_frequencies = [shear_wave_velocity / y_max * beta / (2 * np.pi) for beta in betas]

    # plot PSD
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)
    # plot expected natural frequencies
    for i, expected_natural_frequency in enumerate(expected_natural_frequencies):
        ax.axvline(expected_natural_frequency,
                   color='r',
                   linestyle='-',
                   marker="x",
                   label=f'Analytical' if i == 0 else None)

    ax.plot(f, Pxx_2D, color="b", marker="o", markersize=3, markevery=5, label="STEM 2D")
    ax.plot(f, Pxx_3D, color="orange", linestyle="-.", label="STEM 3D")

    ax.set_xlim(0, 20)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [m^2/Hz]')
    ax.legend(loc='upper right')
    ax.grid()

    plt.savefig(output_file)
    plt.close()


def compare_abs_boundary(path_model: Path, output_file: Path):
    """
    Compare the results of the absorbing boundary simulation between 2D and 3D models.

    Args:
        path_model (Path): Path to the directory containing the JSON files with the results from STEM.
        output_file (Path): Path to save the comparison plot.
    """

    # load data from STEM
    path_2d_results = Path(path_model) / "calculated_output_2D.json"
    path_3d_results = Path(path_model) / "calculated_output_3D.json"
    with open(path_2d_results, "r") as f:
        data_kratos_2d = json.load(f)

    with open(path_3d_results, "r") as f:
        data_kratos_3d = json.load(f)

    young_modulus = 50e6  # Pa
    poisson_ratio = 0.3
    density_solid = 2000  # kg/m3
    porosity = 0
    load_value = -1e3  # Pa

    p_modulus = (young_modulus * (1 - poisson_ratio)) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    bulk_density = density_solid * (1 - porosity)
    analytical_sol = InfinitePileWaveSolution(p_modulus, bulk_density, load_value)
    t = np.linspace(0, 0.5, 100)
    _, analytical_v_0 = analytical_sol.calculate(7.5, t)
    _, analytical_v_1 = analytical_sol.calculate(5, t)
    _, analytical_v_2 = analytical_sol.calculate(2.5, t)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True, sharey=True)
    ax[0].plot(t, analytical_v_0 * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax[0].plot(data_kratos_2d["TIME"],
               np.array(data_kratos_2d["NODE_5"]['VELOCITY_Y']) * 1000,
               color="b",
               marker="o",
               markersize=3,
               markevery=5,
               label="STEM 2D")
    ax[0].plot(data_kratos_3d["TIME"],
               np.array(data_kratos_3d["NODE_9"]['VELOCITY_Y']) * 1000,
               color="orange",
               linestyle="-.",
               label="STEM 3D")
    ax[0].text(0.5, -0.05, '(a)', transform=ax[0].transAxes, ha='center', va='top', fontsize=12)

    ax[1].plot(t, analytical_v_1 * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax[1].plot(data_kratos_2d["TIME"],
               np.array(data_kratos_2d["NODE_6"]['VELOCITY_Y']) * 1000,
               color="b",
               marker="o",
               markersize=3,
               markevery=5,
               label="STEM 2D")
    ax[1].plot(data_kratos_3d["TIME"],
               np.array(data_kratos_3d["NODE_10"]['VELOCITY_Y']) * 1000,
               color="orange",
               linestyle="-.",
               label="STEM 3D")
    ax[1].text(0.5, -0.075, '(b)', transform=ax[1].transAxes, ha='center', va='top', fontsize=12)

    ax[2].plot(t, analytical_v_2 * 1000, label="Analytical", marker="x", color='r', markevery=5)
    ax[2].plot(data_kratos_2d["TIME"],
               np.array(data_kratos_2d["NODE_7"]['VELOCITY_Y']) * 1000,
               color="b",
               marker="o",
               markersize=3,
               markevery=5,
               label="STEM 2D")
    ax[2].plot(data_kratos_3d["TIME"],
               np.array(data_kratos_3d["NODE_11"]['VELOCITY_Y']) * 1000,
               color="orange",
               linestyle="-.",
               label="STEM 3D")
    ax[2].text(0.5, -0.20, '(c)', transform=ax[2].transAxes, ha='center', va='top', fontsize=12)

    ax[0].set_ylabel("Velocity at y=2.5m [mm/s]")
    ax[1].set_ylabel("Velocity at y=5m [mm/s]")
    ax[2].set_ylabel("Velocity at y=7.5m [mm/s]")
    ax[2].set_xlabel("Time [s]")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_xlim(0, 0.5)
    ax[0].set_ylim(-4, 4)
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
    plt.savefig(output_file)
    plt.close()


def compare_simply_supported_beam(path_model: Path, output_file: Path):
    """
    Compare the results of the simply supported beam simulation between 2D and 3D models.

    Args:
        - path_model (Path): Path to the directory containing the JSON files with the results from STEM
        - output_file (Path): Path to save the comparison plot.
    """

    path_model = Path(path_model)

    # Specify beam material model

    DENSITY = 1
    CROSS_AREA = 0.5
    I33 = 1
    total_length = 25
    q = 1  # uniform load in N/m

    YOUNG_MODULUS = 16 * DENSITY * CROSS_AREA * total_length**4 / (np.pi**2 * I33)

    # expected frequency and max displacement
    expected_f = 1 / (2 * np.pi) * (np.pi / total_length)**2 * np.sqrt(YOUNG_MODULUS * I33 / (DENSITY * CROSS_AREA))
    expected_max_disp = 5 * q * total_length**4 / (384 * YOUNG_MODULUS * I33)

    period = 1 / expected_f

    # load data from STEM
    with open(path_model / "json_output_2D_stage_2.json", "r") as f:
        data_kratos_2D = json.load(f)

    with open(path_model / "json_output_3D_stage_2.json", "r") as f:
        data_kratos_3D = json.load(f)

    time = np.array(data_kratos_2D["TIME"])
    displacement_2D = data_kratos_2D["NODE_3"]["DISPLACEMENT_Y"]
    displacement_3D = data_kratos_3D["NODE_3"]["DISPLACEMENT_Y"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)

    # set vertical line at 1/f
    start_time = 0.5 + time[1] - time[0]  # stage 1 duration + delta_time
    ax.axvline(x=period, label='Analytical period', color='r', markevery=5, linestyle='--')
    ax.axvline(x=period * 2, color='r', linestyle='--')
    ax.axvline(x=period * 3, color='r', linestyle='--')
    ax.axvline(x=period * 4, color='r', linestyle='--')
    ax.axhline(y=expected_max_disp, color='r', linestyle=':', label='Analytical displacement limit')
    ax.axhline(y=-expected_max_disp, color='r', linestyle=':')
    ax.plot(time - start_time, displacement_2D, color="b", marker="o", markersize=3, markevery=5, label='STEM 2D')
    ax.plot(time - start_time, displacement_3D, color='orange', linestyle='-.', label='STEM 3D')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mid-span vertical displacement (m)")
    ax.legend(loc='upper right')

    ax.set_xlim(0, 2)
    ax.set_ylim(-0.02, 0.02)
    ax.grid()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def compare_moving_load_on_beam(path_model: Path, output_file: Path):
    """
    Compare the results of the moving load on beam simulation between 2D and 3D models.

    Args:
        - path_model (List[Path]): List of paths to the directories containing the JSON files with the results from STEM
        - output_file (Path): Path to save the comparison plot.
    """

    length = 25
    velocity = 10  # m/s
    time_array = np.linspace(0, length / velocity, 500)
    analytical_solution = BeamMovingLoadAnalytical(length, 210e9, 1e-4, 0.01, 7850, 1000, velocity)
    analytical_deflection = analytical_solution.calculate_dynamic_deflection(length / 2, time_array)

    path_model = Path(path_model)

    # load data from STEM
    with open(path_model / "json_output_2D.json", "r") as f:
        data_kratos_2D = json.load(f)

    with open(path_model / "json_output_3D.json", "r") as f:
        data_kratos_3D = json.load(f)

    time_kratos = np.array(data_kratos_2D["TIME"])
    displacement_2D = data_kratos_2D["NODE_3"]["DISPLACEMENT_Y"]
    displacement_3D = data_kratos_3D["NODE_3"]["DISPLACEMENT_Y"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)

    ax.plot(time_array, analytical_deflection, color='red', marker='x', label='Analytical Solution', markevery=25)
    ax.plot(time_kratos, displacement_2D, color='blue', linestyle='-', label='STEM 2D')
    ax.plot(time_kratos, displacement_3D, color='orange', linestyle='-.', label='STEM 3D')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mid-span vertical displacement (m)")
    ax.legend(loc='upper right')

    ax.grid()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
