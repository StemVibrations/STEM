import sys
from typing import Dict

import numpy as np
import numpy.typing as Npt
from scipy.optimize import brentq
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


class MovingLoadElasticHalfSpace:
    """
    Implementation of the analytical solution for a moving load on an elastic half-space.
    Based on :cite:`Fryba_2013` (Chapter 18).
    The solution is only valid for subsonic load speeds (c < c2), and only computes vertical displacements.

    """

    def __init__(self, E: float, nu: float, rho: float, force: float, speed: float):
        """
        Initialize the elastic half-space model.

        Args:
            - E (float): Young's modulus (Pa)
            - nu (float): Poisson's ratio (dimensionless)
            - rho (float): Density (kg/m³)
            - force (float): Magnitude of the concentrated force (N)
            - speed (float): Speed of the moving load (m/s)
        """
        self.E = E
        self.nu = nu
        self.rho = rho
        self.force = force
        self.speed = speed

        # Lamé parameters
        self.lame_lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
        self.G = E / (2 * (1 + nu))  # Shear modulus
        self.M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))  # P-wave modulus

        # Wave speeds
        self.cp = np.sqrt(self.M / self.rho)  # P-wave speed
        self.cs = np.sqrt(self.G / self.rho)  # S-wave speed
        self.cr = None

        # Parameters
        self.Gp = None  # Green's function for P-waves
        self.Gs = None  # Green's function for S-waves

        # Rayleigh wave speed
        self.Rayleigh_wave_speed()

        # wave Mach number
        self.Mp = self.speed / self.cp
        self.Ms = self.speed / self.cs
        # wave decay parameters
        self.beta_p = np.sqrt(1 - self.Mp**2)
        self.beta_s = np.sqrt(1 - self.Ms**2)
        self.beta_r = np.sqrt(1 - (self.speed / self.cr)**2)

        # check speed regime
        self.__check_speed_regime()

    @staticmethod
    def __rayleigh_func(xi: float, eta: float) -> float:
        """
        Rayleigh function for computing the Rayleigh wave speed.

        Args:
            - xi (float): Dimensionless Rayleigh speed (c_r / c_s)
            - eta (float): Ratio of shear wave speed to compressional wave speed (c_s / c_p)

        Returns:
            - (float): Value of the Rayleigh function
        """
        return xi**3 - 8 * xi**2 + 8 * xi * (3 - 2 * eta) * xi**2 - 16 * (1 - eta)

    def Rayleigh_wave_speed(self):
        """
        Compute the Rayleigh wave speed c_r based on the material properties.

        It solves Rayleigh equation: ξ³ - 8ξ² + 8ξ(3 - 2n) - 16(1 - η) = 0
        See: https://en.wikipedia.org/wiki/Rayleigh_wave
        """

        eta = self.cs / self.cp
        # xi = c_r / self.cs

        # solve cubic equation for qsi
        xi = brentq(self.__rayleigh_func, 0, 1, args=(eta))
        self.cr = xi * self.cs

    def __check_speed_regime(self):
        """
        Check the speed regime.
        """
        if self.speed > self.cr:
            sys.exit("Error: The current implementation only supports subsonic speeds (c < cr).")

    def compute(self, x: float, y: float, z: float, t: float, ky_max: float = 8.) -> float:
        """
        Compute the vertical displacement at a given point and time.

        It assumes that the poin load moves along the y-axis.

        Args:
            - x (float): x-coordinate of the observation point (m)
            - y (float): y-coordinate of the observation point (m)
            - z (float): z-coordinate of the observation point (m)
            - t (float): time (s)
        Returns:
            - uz (float): vertical displacement at the observation point (m)
        """

        # Cylindrical coordinates
        r = np.sqrt(x**2 + z**2)
        theta = np.arctan2(z, x)

        # Moving coordinate
        y_prime = y - self.speed * t

        # ky limits for integration
        ky = np.linspace(-ky_max, ky_max, 100)
        # tau limits for integration
        # np.exp(r * ky * tau **2) < epsilon (e.g. 1e-10)
        # tau_max > np.sqrt(-lm(epsilon) / r * np.abs(ky))
        # using epsilon = 1e-10 yields: tau_max > np.sqrt(23 / (r * np.abs(ky))) ~ 4.8
        tau_max = 5.0 / np.sqrt(max(ky_max * r, 1e-6))
        tau = np.linspace(-tau_max, tau_max, 100)
        self.__Gp(theta, tau, ky, r)
        # self.__Gs(theta, tau, ky, r)

        # Synthesis of Fourier components
        # integrand = (self.Gp + self.Gs) * np.exp(1j * ky * y_prime)
        integrand = (self.Gp) * np.exp(1j * ky * y_prime)

        # scale factor Equation 10
        u = -self.force / (4 * np.pi**2 * self.G) * integrand

        return

    def __Gp(self, theta: float, tau: Npt.NDArray, ky: Npt.NDArray, r: float):

        # compute angles for region classification (Fig 3)
        theta_r = np.arccos(np.real(self.beta_r / self.beta_p))
        theta_s = np.arccos(np.real(self.beta_s / self.beta_p))

        if theta_r < theta < np.pi - theta_r:
            region = 'I'
        elif (0 <= theta < theta_s) or (np.pi - theta_s < theta <= np.pi):
            region = 'III'
        else:
            region = 'II'

        # Equation 24
        term1 = -1j * np.cos(theta) * (tau**2 + self.beta_p)
        term2 = tau * np.sin(theta) * np.sqrt(tau**2 + 2 * self.beta_p + 0j)
        kx_bar = term1 + term2

        # variables between Equation 5 and 6
        kx = ky * kx_bar
        k_sq = kx**2 + ky**2
        kp = self.Mp * ky
        ks = self.Ms * ky
        v = np.sqrt(k_sq - kp**2)
        v_prime = np.sqrt(k_sq - ks**2)

        # Equation 24 & 25
        diff_kx_bar = -1j * np.cos(theta) * 2 * tau + \
            np.sin(theta) * (np.sqrt(tau**2 + 2*self.beta_p + 0j) +
                             tau**2 * np.sin(theta) / np.sqrt(tau**2 + 2*self.beta_p + 0j))

        # compute Rayleigh denominator
        F, F_prime = self.__rayleigh_denominator(k_sq, ky, kx_bar, ks, v, v_prime)

        # Equation 8
        A = (2 * k_sq - ks**2) / F
        E = -v * A * np.exp(-v * z)

        # Equation 14
        integral = (E / F) * diff_kx_bar * np.exp(-ky * r * tau**2)

        # Integrate using trapezoidal rule
        dtau = tau[1] - tau[0]
        integral_sdp = trapezoid(integral, dx=dtau)

        # Equation 25
        self.Gp = ky * np.exp(-ky * r * self.beta_p) * integral_sdp

        # perform pole correction
        correction = self.__pole_correction_p(region, theta, ky, E, F_prime)

        self.Gp += correction

    def __pole_correction_p(self, region: str, theta: float, ky: Npt.NDArray, E: Npt.NDArray,
                            F_prime: Npt.NDArray) -> Npt.NDArray:
        """
        Perform pole correction for the Green's function.

        Args:
            - region (str): Region classification ('I', 'II', 'III')
            - theta (float): Angle in cylindrical coordinates
            - ky (Npt.NDArray): Array of ky values
            - E (Npt.NDArray): Array of E values from Equation 8
            - F_prime (Npt.NDArray): Array of derivatives of the Rayleigh function

        Returns:
            - correction (Npt.NDArray): Pole correction values
        """
        if region == 'I':
            return 0.0

        v_r = self.beta_p / self.beta_r

        exponent = -ky * (v_r * z + self.beta_r * x * np.sign(np.cos(theta)))
        correction = (-np.sign(np.cos(theta)) * 2j * np.pi * (E / F_prime) * np.exp(exponent))

        return correction

    def __rayleigh_denominator(self, k_sq: Npt.NDArray, ky: Npt.NDArray, kx_bar: Npt.NDArray, ks: Npt.NDArray,
                               v: Npt.NDArray, v_prime: Npt.NDArray) -> Npt.NDArray:
        """
        Compute Rayleigh function F(k) and their derivative F'(k).

        Args:
            - k_sq (Npt.NDArray): Array of k squared values
            - ky (Npt.NDArray): Array of ky values
            - ks (Npt.NDArray): Array of ks values
            - kx_bar (Npt.NDArray): Array of kx_bar values
            - v (Npt.NDArray): Array of v values
            - v_prime (Npt.NDArray): Array of v_prime values

        Returns:
            - F (Npt.NDArray): Array of Rayleigh function values
        """
        # Equation 9
        F = (2 * k_sq - ks**2)**2 - 4 * k_sq * v * v_prime

        # Derivative of F with respect to kx_bar (Equation 29) - NOTE: this equation is wrong the the paper
        # i fixed it with maxima
        F_prime = 8 * kx_bar * ky**2 * (2 * (kx_bar**2 * ky**2 + ky**2) - self.Ms**2 *
                                        ky**2) - 8 * kx_bar * ky**2 * np.sqrt(self.Mp * ky) * np.sqrt(self.Ms * ky)

        return F, F_prime

    def __Gs(self):

        pass

    # def _compute_beta_parameters(self, c: float) -> Dict:
    #     """Compute β parameters for given velocity c."""
    #     params = {}
    #     params['M_p'] = c / self.cp
    #     params['M_s'] = c / self.cs
    #     params['beta_p'] = np.sqrt(1 - params['M_p']**2 + 0j)
    #     params['beta_s'] = np.sqrt(1 - params['M_s']**2 + 0j)
    #     params['beta_R'] = np.sqrt(1 - (c/self.c_R)**2 + 0j)
    #     params['c'] = c
    #     return params

    # @staticmethod
    # def function_to_integrate(q1, q2, x, y, z, alpha1, alpha2):

    #     # Equation 18.14
    #     n1 = np.sqrt((1 - alpha1**2) * q1**2 + q2**2)
    #     n2 = np.sqrt((1 - alpha2**2) * q1**2 + q2**2)

    #     # Equation (18.19)
    #     B = (q1**2 + q2**2 + n2**2)**2 - 4 * n1 * n2 * (q1**2+q2**2)
    #     B[B < 1e-12] = 1e-12

    #     first_part = n1 * (q1**2 + q2**2 + n2**2) * np.exp(-n1 * z)
    #     second_part = -2 * n1 * (q1**2 + q2**2) * np.exp(-n2 * z)
    #     integrals = 1 / B * (first_part + second_part) * np.exp(1j * (x * q1 + y * q2))
    #     return integrals

    # def vertical_displacement(self, x: float, y: float, z: float, limits=(-15, 15), num_points=500) -> float:

    #     q1_list = np.linspace(limits[0], limits[1], num_points)
    #     q2_list = np.linspace(limits[0], limits[1], num_points)

    #     q1, q2 = np.meshgrid(q1_list, q2_list, indexing="ij")
    #     # Evaluate function on the grid
    #     fct = self.function_to_integrate(q1, q2, x, y, z, self.alpha1, self.alpha2)

    #     first_int = integrate.simpson(fct, q1_list, axis=1)
    #     second_int = integrate.simpson(first_int, q2_list)

    #     vert_disp = (self.force / (4 * np.pi**2 * self.G)) * second_int

    #     return np.real(vert_disp)


if __name__ == "__main__":
    # Example usage
    E = 30e6  # Pa
    nu = 0.4  # dimensionless
    rho = 2000  # kg/m³
    force = 1e4  # N
    speed = 10  # m/s

    x, y, z = 0.0, 0.0, 10
    time = np.linspace(-5, 5, num=100)

    model = MovingLoadElasticHalfSpace(E, nu, rho, force, speed)

    for t in time:
        model.compute(x, y, z, t)
