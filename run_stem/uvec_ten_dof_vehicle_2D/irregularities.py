import numpy as np
import numpy.typing as npt


def calculate_rail_irregularity(x: float, f_min: float = 2, f_max: float = 500, N: int = 2000, Av: float = 0.00002095,
                                omega_c: float = 0.8242, seed=14) -> float:
    """
     Creates rail unevenness following :cite: `zhang_2001`.

    A summary of default values can be found in : cite: `Podworna_2015`.

    Args:
        - x (float) : position of the node [m]
        - f_min (float): minimum spatial frequency for the PSD of the unevenness [1/m] (default 2 1/m)
        - f_max: (float) maximum spatial frequency for the PSD of the unevenness [1/m] (default 500 1/m)
        - N (int): number of frequency increments [-] (default 2000)
        - Av (float): vertical track irregularity parameters [m2 rad/m]  (default 0.00002095 m2 rad/m)
        - omega_c (float): critical wave number [rad/m] (default 0.8242 rad/m)
        - seed: (int) seed for random generator [-] (default 14)

    Returns:
        - irregularity (float): irregularity at the node [m]
    """

    # random generator
    random_generator = np.random.default_rng(seed)

    # define omega range
    omega_max = 2 * np.pi * f_max
    omega_min = 2 * np.pi * f_min
    delta_omega = (omega_max - omega_min) / N

    # for each frequency increment
    omega_n = omega_min + delta_omega * np.arange(N)
    phi = random_generator.uniform(0, 2 * np.pi, N)
    irregularity = np.sum(np.sqrt(4 * spectral(omega_n, Av, omega_c) * delta_omega) * np.cos(omega_n * x - phi))

    return irregularity


def spectral(omega: npt.NDArray[np.float64], Av: float, omega_c: float) -> npt.NDArray[np.float64]:
    """
    Computes spectral unevenness

    Args:
        - omega (npt.NDArray[np.float64]): wave number [rad/m]
        - Av (float): vertical track irregularity parameters [m2 rad/m]
        - omega_c (float): critical wave number [rad/m]

    Returns:
        - spectral_unevenness (npt.NDArray[np.float64]): spectral unevenness [m3 / rad]
    """

    spectral_unevenness = 2 * np.pi * Av * omega_c ** 2 / ((omega ** 2 + omega_c ** 2) * omega ** 2)
    return spectral_unevenness
