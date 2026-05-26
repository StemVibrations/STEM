from typing import Tuple, Union

import numpy as np


class InfinitePileWaveSolution:
    """
    Analytical solution for wave propagation in an infinite pile under a sudden load :cite:`Verruijt_2010`.

    Attributes:
        - K (float): Bulk modulus of the pile material.
        - rho (float): Density of the pile material.
        - q (float): Load applied at the top of the pile (force per unit area).
        - c (float): Wave speed in the pile material.
    """

    def __init__(self, K: float, rho: float, load: float):
        """
        Initialize the infinite pile wave solution.

        Args:
            - K (float): Bulk modulus of the pile material.
            - rho (float): Density of the pile material.
            - load (float): load applied at the top of the pile (force per unit area).
        """
        self.K = K
        self.rho = rho
        self.q = load

        self.c = (K / rho)**0.5  # Wave speed

    def calculate(self, x: float, t: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the analytical solution at position x and time t.

        Args:
            - x (float): Position along the pile (m).
            - t (Union[float, np.ndarray]): Time(s) at which to evaluate the solution (s).

        Returns:
            - Tuple[np.ndarray, np.ndarray]: Displacement and velocity at position x and time t.

        """

        t = np.asarray(t, dtype=float)
        mask = t >= x / self.c

        u = np.zeros_like(t)
        v = np.zeros_like(t)

        v[mask] = self.q * self.c / self.K
        u[mask] = v[mask] * (t[mask] - x / self.c)

        return u, v
