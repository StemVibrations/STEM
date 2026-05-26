import numpy as np


class BeamMovingLoadAnalytical:
    """
    Class to compute the analytical solution for an undamped simply supported beam under a moving point load.
    Based on the solution presented in :cite:`Fryba_1972` Chapter 1.3.2.2.

    Attributes:
        - L (float): Length of the beam.
        - A (float): Cross-sectional area of the beam.
        - rho (float): Density of the beam material.
        - P (float): Magnitude of the moving point load.
        - v (float): Velocity of the moving load.
        - rho_A (float): Mass per unit length of the beam.
        - EI (float): Flexural rigidity of the beam.
    """

    def __init__(self, length: float, young_modulus: float, inertia: float, cross_area: float, density: float,
                 load: float, velocity: float):
        """
        Initialize the BeamMovingLoadAnalytical class with beam properties and load parameters.

        Args:
            - length (float): Length of the beam.
            - young_modulus (float): Young's modulus of the beam material.
            - inertia (float): Second moment of area (moment of inertia) of the beam cross-section.
            - cross_area (float): Cross-sectional area of the beam.
            - density (float): Density of the beam material.
            - load (float): Magnitude of the moving point load.
            - velocity (float): Velocity of the moving load.
        """
        self.L: float = length
        self.A: float = cross_area
        self.rho: float = density

        self.P: float = load
        self.v: float = velocity

        self.rho_A: float = self.rho * self.A
        self.EI: float = young_modulus * inertia

    def critical_velocity(self) -> float:
        """
        Calculate the critical velocity of the beam.

        Returns:
            - float: Critical velocity.
        """
        return (np.pi / self.L) * np.sqrt(self.EI / self.rho_A)

    def static_deflection_at_midspan_approximation(self) -> float:
        """
        Calculate approximation of the static deflection at mid-span of a simply supported beam under a central point
        load, using only the first mode shape.

        Returns:
            - float: Static deflection at mid-span first mode approximation.
        """

        return (2 * self.P * self.L**3) / (np.pi**4 * self.EI)

    def calculate_dynamic_deflection(self, x: float, time: np.array, n_modes: int = 100):
        """
        Calculate the dynamic deflection of the beam at position x and time array time, using n_modes modes.

        Args:
            - x (float): Position along the beam length (0 <= x <= L).
            - time (np.array): Array of time points at which to calculate the deflection.
            - n_modes (int): Number of modes to consider in the calculation.

        Returns:
            - np.array: Dynamic deflection at position x and times in time array.
        """

        # Check if x is within the beam length and return zero deflection if outside
        if x < 0 or x > self.L:
            return np.zeros_like(time)

        # Dimensionless speed ratio
        alpha = self.v / self.critical_velocity()

        deflection = np.zeros_like(time)

        # Angular frequency of the moving load
        omega = np.pi * self.v / self.L

        for n in range(1, n_modes + 1):

            # omega_n = n^2 * omega_1
            # In this form, we use the dimensionless alpha/n ratio

            # Modal natural frequency for the sine term
            omega_1 = (np.pi / self.L)**2 * np.sqrt(self.EI / self.rho_A)
            omega_n = n**2 * omega_1

            # Mode calculation
            forced = np.sin(n * omega * time)

            mode_shape = np.sin(n * np.pi * x / self.L)

            # To avoid division by zero when n == alpha
            if np.isclose(n, alpha):
                deflection += 1 / (2 * n**4) * (forced - n * omega * time * np.cos(n * omega * time)) * mode_shape
            else:
                free = (alpha / n) * np.sin(omega_n * time)
                denom = (n**2) * (n**2 - alpha**2)

                deflection += (1.0 / denom) * (forced - free) * mode_shape

        return self.static_deflection_at_midspan_approximation() * deflection
