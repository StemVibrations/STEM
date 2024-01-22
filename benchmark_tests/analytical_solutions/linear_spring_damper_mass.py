import sys
import numpy as np


class LinearSpringDamperMass:
    """
    Analytical solution of a mass on a spring-damper system.

    Attributes:
        - k [float]: spring stiffness [N/m]
        - c [float]: damping coefficient [Ns/m]
        - m [float]: mass [kg]
        - g [float]: gravity acceleration [m/s^2]
        - time [npt.NDArray[np.float64]]: time vector [s]
        - displacement [npt.NDArray[np.float64]]: displacement vector [m]
    """

    def __init__(self, k: float, c: float, m: float, g: float, end_time: float, n_steps: int):
        """
        Initialise the system.

        Args:
            - k [float]: spring stiffness [N/m]
            - c [float]: damping coefficient [Ns/m]
            - m [float]: mass [kg]
            - g [float]: gravity acceleration [m/s^2]
            - end_time [float]: end time of the simulation [s]
            - n_steps [int]: number of time steps
        """
        self.k = k
        self.c = c
        self.m = m
        self.g = g

        self.time = np.linspace(0, end_time, n_steps)
        self.displacement = np.zeros(n_steps)

    def solve(self):
        """
        Solve the system. That is, calculate the displacement of the mass at each time step.

        """

        # static displacement
        u_0 = self.m * self.g / self.k

        # resonance frequency of undamped system
        omega_0 = np.sqrt(self.k / self.m)

        # damping ratio of the system
        qsi = self.c / (2 * np.sqrt(self.k * self.m))

        # if the system overdamped: solution not valid
        if qsi >= 1:
            sys.exit("The system is overdamped.\nAnalytical solution not valid.")

        # resonance frequency of damped system
        omega_1 = omega_0 * np.sqrt(1 - qsi ** 2)

        # phase angle
        psi = np.arctan(qsi / np.sqrt(1 - qsi ** 2))

        # calculate displacement
        self.displacement = u_0 * np.cos(omega_1 * self.time - psi) / np.cos(psi) * np.exp(-qsi * omega_0 * self.time)