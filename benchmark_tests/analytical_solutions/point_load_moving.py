import sys
from typing import Dict

import numpy as np
import numpy.typing as Npt
from scipy.optimize import brentq
from scipy.integrate import trapezoid
from numpy.polynomial.hermite import hermgauss
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
        self.cr = self.Rayleigh_wave_speed()  # Rayleigh wave speed

        # Parameters
        self.Gp = None  # Green's function for P-waves
        self.Gs = None  # Green's function for S-waves
        self.vertical_displacement = None  # Vertical displacement uz

        # wave Mach number
        self.Mp = self.speed / self.cp
        self.Ms = self.speed / self.cs
        # wave decay parameters
        self.beta_p = np.sqrt(1 - self.Mp**2)
        self.beta_s = np.sqrt(1 - self.Ms**2)
        self.beta_r = np.sqrt(1 - (self.speed / self.cr)**2)

        # Gaussian quadrature points and weights
        self.n_gh = 128
        self.gh_x, self.gh_w = hermgauss(self.n_gh)

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
        return xi**3 - 8 * xi**2 + 8 * xi * (3 - 2 * eta) - 16 * (1 - eta)

    def Rayleigh_wave_speed(self):
        """
        Compute the Rayleigh wave speed c_r based on the material properties.

        It solves Rayleigh equation: ξ³ - 8ξ² + 8ξ(3 - 2n) - 16(1 - η) = 0
        See: https://en.wikipedia.org/wiki/Rayleigh_wave
        """

        eta = self.cs**2 / self.cp**2

        # solve cubic equation for qsi
        xi = brentq(self.__rayleigh_func, 0, 1, args=(eta, ))
        return np.sqrt(xi) * self.cs

    def __check_speed_regime(self):
        """
        Check the speed regime.
        """
        if self.speed > self.cr:
            sys.exit("Error: The current implementation only supports subsonic speeds (c < cr).")

    def compute(self, x: float, y: float, z: float, t: float, ky_max: float = 8., n_tau=100, n_ky=400) -> float:
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
        ky = np.linspace(ky_max, ky_max, n_ky)
        # tau limits for integration
        # np.exp(r * ky * tau **2) < epsilon (e.g. 1e-10)
        # tau_max > np.sqrt(-lm(epsilon) / r * np.abs(ky))
        # using epsilon = 1e-10 yields: tau_max > np.sqrt(23 / (r * np.abs(ky))) ~ 4.8
        tau_max = 5.0 / np.sqrt(max(ky_max * r, 1e-6))
        tau = np.linspace(-tau_max, tau_max, n_tau)

        self.Gp = np.zeros_like(ky, dtype=complex)
        self.Gs = np.zeros_like(ky, dtype=complex)

        for i, ky_i in enumerate(ky):
            self.Gp[i] = self.__Gp(theta, tau, ky_i, r, z)
            self.Gs[i] = self.__Gs(theta, tau, ky_i, r, z)

        # Synthesis of Fourier components Equation 13
        integrand = (self.Gp + self.Gs) * np.exp(-1j * ky * y_prime)
        I_ky = trapezoid(integrand, x=ky)

        # scale factor Equation 10c
        self.vertical_displacement = -self.force / (4 * np.pi**2 * self.G) * I_ky

        return np.real(self.vertical_displacement)

    def __Gp(self, theta: float, tau: Npt.NDArray, ky: float, r: float, z: float):

        # compute angles for region classification (Fig 3)
        theta_r = np.arccos(np.real(self.beta_r / self.beta_p))
        theta_s = np.arccos(np.real(self.beta_s / self.beta_p))

        if theta_r < theta < np.pi - theta_r:
            region = 'I'
        elif (0 <= theta < theta_s) or (np.pi - theta_s < theta <= np.pi):
            region = 'III'
        else:
            region = 'II'

        a = r * ky
        t = self.gh_x / np.sqrt(a)
        weight = self.gh_w / np.sqrt(a)

        # Equation 24
        kx_bar = -1j * np.cos(theta) * (t**2 + self.beta_p) + t * np.sin(theta) * np.sqrt(t**2 + 2 * self.beta_p)

        # Equation 25
        diff_kx_bar = -1j * np.cos(theta) * 2 * t + \
            np.sin(theta) * (np.sqrt(t**2 + 2*self.beta_p + 0j) +
                             t**2 * np.sin(theta) / np.sqrt(t**2 + 2*self.beta_p))

        # variables between Equation 5 and 6
        kx = ky * kx_bar
        k_sq = kx**2 + ky**2
        kp = self.Mp * ky
        ks = self.Ms * ky
        v = np.sqrt(k_sq - kp**2)
        v_prime = np.sqrt(k_sq - ks**2)

        # compute Rayleigh denominator
        F, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar, ks, v, v_prime)

        # Equation 8
        A = (2 * k_sq - ks**2) / F
        # Equation 10c
        E = -v * A * np.exp(-v * z)

        # Equation 14
        # integral = (E / F) * diff_kx_bar * np.exp(-ky * r * tau**2)
        # Integrate using trapezoidal rule
        # integral_sdp = trapezoid(integral, x=tau)
        # Using Gaussian quadrature
        integral_sdp = np.sum(weight * (E / F) * diff_kx_bar)

        # Equation 25
        Gp = ky * np.exp(-ky * r * self.beta_p) * integral_sdp

        # perform pole correction
        correction = self.__pole_correction_p(region, theta, ky, E, F, F_prime, kx_bar, diff_kx_bar, v, z)

        Gp += correction

        return Gp

    def __Gs(self, theta: float, tau: Npt.NDArray, ky: float, r: float, z: float):

        # compute angles for region classification (Fig 5)
        theta_r_star = np.arccos(np.real(self.beta_r / self.beta_s))

        if theta_r_star < theta < np.pi - theta_r_star:
            region = 'I'
        else:
            region = 'II'

        if ky <= 0:
            return 0

        a = r * ky
        t = self.gh_x / np.sqrt(a)
        weight = self.gh_w / np.sqrt(a)

        # Equation 34
        kx_bar = -1j * np.cos(theta) * (t**2 + self.beta_s) + t * np.sin(theta) * np.sqrt(t**2 + 2 * self.beta_s)

        # Equation 35
        diff_kx_bar = -1j * np.cos(theta) * 2 * t + \
            np.sin(theta) * (np.sqrt(t**2 + 2*self.beta_s) +
                             t**2 * np.sin(theta) / np.sqrt(t**2 + 2*self.beta_s))

        # variables between Equation 5 and 6
        kx = ky * kx_bar
        k_sq = kx**2 + ky**2
        kp = self.Mp * ky
        ks = self.Ms * ky
        v = np.sqrt(k_sq - kp**2)
        v_prime = np.sqrt(k_sq - ks**2)

        # compute Rayleigh denominator
        F, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar, ks, v, v_prime)

        # Equation 7
        C = 2 * v / F
        # Equation 10c
        E = k_sq * C * np.exp(-v_prime * z)

        # Equation 14
        # integral = (E / F) * diff_kx_bar * np.exp(-ky * r * tau**2)

        # Integrate using trapezoidal rule
        # integral_sdp = trapezoid(integral, x=tau)
        # Using Gaussian quadrature
        integral_sdp = np.sum(weight * (E / F) * diff_kx_bar)

        # Equation 25
        Gs = ky * np.exp(-ky * r * self.beta_s) * integral_sdp

        # perform pole correction
        correction = self.__pole_correction_s(region, theta, ky, E, F_prime)

        Gs += correction
        return Gs

    def __pole_correction_p(self, region: str, theta: float, ky: float, E: Npt.NDArray, F: Npt.NDArray,
                            F_prime: Npt.NDArray, kx_bar: Npt.NDArray, diff_kx_bar: Npt.NDArray, v: Npt.NDArray,
                            z: float) -> Npt.NDArray:
        """
        Perform pole correction for the Green's function.

        Args:
            - region (str): Region classification ('I', 'II', 'III')
            - theta (float): Angle in cylindrical coordinates
            - ky (Npt.NDArray): Array of ky values
            - E (Npt.NDArray): Array of E values from Equation 8
            - F (Npt.NDArray): Array of Rayleigh function values
            - F_prime (Npt.NDArray): Array of derivatives of the Rayleigh function
            - kx_bar (Npt.NDArray): Array of kx_bar values
            - diff_kx_bar (Npt.NDArray): Array of derivatives of kx_bar
            - v (Npt.NDArray): Array of v values
            - x (float): x-coordinate of the observation point (m)
            - z (float): z-coordinate of the observation point (m)

        Returns:
            - correction (Npt.NDArray): Pole correction values
        """
        if region == 'I':
            return 0.0

        # Pole location in normalized plane
        kx_bar_pole = -1j * self.beta_p * np.cos(theta)

        kx = ky * kx_bar_pole
        k_sq = kx**2 + ky**2
        kp = self.Mp * ky
        ks = self.Ms * ky
        v = np.sqrt(k_sq - kp**2)
        v_prime = np.sqrt(k_sq - ks**2)

        _, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar_pole, ks, v, v_prime)

        # Equation 28
        v_r_bar = np.sqrt(self.beta_p**2 - self.beta_r**2)
        x = r * np.cos(theta)
        exponent = -ky * (v_r_bar * z + self.beta_r * x * np.sign(np.cos(theta)))

        correction = (-np.sign(np.cos(theta)) * 2 * np.pi * 1j * (E / F_prime) * np.exp(exponent))

        if region == 'II':
            return correction

        if region == 'III':
            # Equation 31
            integral = E / F * np.exp(-ky * v * z + 1j * diff_kx_bar * x)
            correction += trapezoid(integral, dx=kx_bar)
            return correction

    def __pole_correction_s(self, region: str, theta: float, ky: float, E: Npt.NDArray,
                            F_prime: Npt.NDArray) -> Npt.NDArray:
        """
        Perform pole correction for the Green's function.

        Args:
            - region (str): Region classification ('I', 'II')
            - theta (float): Angle in cylindrical coordinates
            - ky (Npt.NDArray): Array of ky values
            - E (Npt.NDArray): Array of E values from Equation 8
            - F_prime (Npt.NDArray): Array of derivatives of the Rayleigh function

        Returns:
            - correction (Npt.NDArray): Pole correction values
        """
        if region == 'I':
            return 0.0

        # Pole location in normalized plane
        kx_bar_pole = -1j * self.beta_s * np.cos(theta)

        kx = ky * kx_bar_pole
        k_sq = kx**2 + ky**2
        kp = self.Mp * ky
        ks = self.Ms * ky
        v = np.sqrt(k_sq - kp**2)
        v_prime = np.sqrt(k_sq - ks**2)

        _, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar_pole, ks, v, v_prime)

        v_r_bar = np.sqrt(self.beta_s**2 - self.beta_r**2)
        exponent = -ky * (v_r_bar * z + self.beta_r * x * np.sign(np.cos(theta)))
        correction = (-np.sign(np.cos(theta)) * 2 * np.pi * 1j * (E / F_prime) * np.exp(exponent))
        return correction

    def __rayleigh_denominator(self, k_sq: Npt.NDArray, kx: Npt.NDArray, kx_bar: Npt.NDArray, ks: Npt.NDArray,
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
        # I fixed it with Maxima
        F_prime = 2*(4*kx_bar+(2*self.Ms**2*kx**2)/kx_bar**3)*(2*kx_bar**2-(self.Ms**2*kx**2)/kx_bar**2) - \
            8 * kx_bar * v * v_prime + 2 * self.Mp * kx * v_prime / v + 2 * self.Ms * kx * v / v_prime

        return F, F_prime


def main():
    # Example usage
    E = 30e6  # Pa
    nu = 0.1  # dimensionless
    rho = 2000  # kg/m³
    force = 1e3  # N
    speed = 10  # m/s

    x, y, z = 0.0, 0.0, 10
    time = np.linspace(-20, 20, num=100)

    model = MovingLoadElasticHalfSpace(E, nu, rho, force, speed)

    uz = []
    for t in time:
        model.compute(x, y, z, t)
        uz.append(np.real(model.vertical_displacement))

    plt.plot(time, uz)
    plt.xlabel("Time step")
    plt.ylabel("Vertical displacement uz (m)")
    plt.grid()
    plt.savefig("moving.png")
    plt.show()


if __name__ == "__main__":
    main()
