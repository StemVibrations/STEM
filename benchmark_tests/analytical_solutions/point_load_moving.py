import numpy as np
from numpy.lib.scimath import sqrt
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy import integrate
import sys


class MovingLoadElasticHalfSpace:
    """
    Implementation of the steady-state response of an elastic half-space
    to a moving point load (Vertical Displacement uz).
    Based on Liao et al. (2005).
    """

    def __init__(self, E: float, nu: float, rho: float, force: float, speed: float):
        """
        Initialize the model with material properties and load parameters.

        Args:
            - E (float): Young's modulus of the half-space
            - nu (float): Poisson's ratio of the half-space
            - rho (float): Density of the half-space
            - force (float): Magnitude of the moving point load
            - speed (float): Speed of the moving load (must be sub-Rayleigh)
        """
        self.E = E
        self.nu = nu
        self.rho = rho
        self.force = force
        self.speed = speed

        # Lamé parameters
        self.G = E / (2 * (1 + nu))  # Shear modulus (mu)
        self.M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

        # Wave speeds [cite: 71]
        self.cp = sqrt(self.M / self.rho)
        self.cs = sqrt(self.G / self.rho)
        self.cr = self.Rayleigh_wave_speed()  # Rayleigh wave speed

        # Check subsonic condition
        if self.speed >= self.cr:
            raise ValueError(f"Speed {self.speed:.2f} m/s must be sub-Rayleigh (< {self.cr:.2f} m/s)")

    def compute_vertical_displacement(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        ky_max: float = 50.0,
        n_ky: int = 2000,
    ):
        """
        Compute the vertical displacement uz at a point (x, y, z) and time t due to a moving point load.

        Args:
            - x (float): Horizontal distance from the load path (m)
            - y (float): Along-track distance from the load (m)
            - z (float): Depth below the surface (m)
            - t (float): Time at which to compute the displacement (s)
        Returns:
            - uz (float): Vertical displacement at the specified point and time
        """

        # wavenumber range for integration
        ky = np.geomspace(1e-4, ky_max, n_ky)

        # Equation 11
        self.Mp = self.speed / self.cp
        self.Ms = self.speed / self.cs
        self.beta_p = sqrt(1 - self.Mp**2)
        self.beta_s = sqrt(1 - self.Ms**2)
        self.beta_r = sqrt(1 - (self.speed / self.cr)**2)

        # Moving coordinate
        y_prime = y - self.speed * t

        integrand_list = []
        ti, wi = np.polynomial.hermite.hermgauss(128)

        # constants for integrand evaluation
        r = sqrt(x**2 + z**2)
        theta = np.arctan2(z, x)
        region_p, region_s = self.define_regions(theta)

        for _, k in enumerate(ky):
            Gp = self.__integrand_Gp(x, z, k, r, theta, region_p, ti, wi)
            Gs = self.__integrand_Gs(x, z, k, r, theta, region_s, ti, wi)

            # add residue/pole terms if applicable (use complex-step derivative at pole)
            # include +ky and -ky contributions (symmetry)
            I_plus = (Gp + Gs) * np.exp(-1j * k * y_prime)
            I_minus = (np.conj(Gp) + np.conj(Gs)) * np.exp(1j * k * y_prime)
            integrand_list.append(I_plus + I_minus)
        I_ky = integrate.trapezoid(integrand_list, x=ky)

        # # Synthesis of Fourier components Equation 13
        # integrand = (Gp + Gs) * np.exp(-1j * ky * y_prime)
        # I_ky = integrate.trapezoid(integrand, x=ky)

        # scale factor Equation 10c
        self.vertical_displacement = -self.force / (4 * np.pi**2 * self.G) * I_ky

    def define_regions(self, theta: float) -> tuple[str, str]:
        """
        Define the regions in the (x, z) plane based on the angle theta and the wave speeds.

        Args:
            - theta (float): Angle between the point of interest and the load path (radians)
        Returns:
            - region_p (str): Region classification for P-wave contribution (I, II, or III)
            - region_s (str): Region classification for S-wave contribution (I or II)
        """
        # compute angles for region classification (Fig 3)
        theta_r = np.arccos(np.real(self.beta_r / self.beta_p))
        theta_s = np.arccos(np.real(self.beta_s / self.beta_p))

        if theta_r < theta < np.pi - theta_r:
            region_p = 'I'
        elif (0 <= theta < theta_s) or (np.pi - theta_s < theta <= np.pi):
            region_p = 'III'
            sys.exit("Region III correction not implemented yet")
        else:
            region_p = 'II'

        # compute angles for region classification (Fig 5)
        theta_r_star = np.arccos(np.real(self.beta_r / self.beta_s))

        if theta_r_star < theta < np.pi - theta_r_star:
            region_s = 'I'
        else:
            region_s = 'II'

        return region_p, region_s

    def __integrand_Gp(self, x: float, z: float, ky: float, r: float, theta: float, region: str, ti: np.ndarray,
                       wi: np.ndarray) -> complex:
        """
        Compute the integrand Gp for the P-wave contribution to the vertical displacement.

        Args:
            - x (float): Horizontal distance from the load path (m)
            - z (float): Depth below the surface (m)
            - ky (float): Wavenumber for integration (1/m)
            - r (float): Distance from the load path to the point of interest (m)
            - theta (float): Angle between the point of interest and the load path (radians)
            - region (str): Region classification for P-wave contribution (I, II, or III)
            - ti (np.ndarray): Hermite quadrature nodes
            - wi (np.ndarray): Hermite quadrature weights
        Returns:
            - Gp (complex): Value of the integrand Gp for the given parameters
        """

        # map Hermite nodes
        # Decay factor is exp(-r * ky * tau**2)
        tau = ti / np.sqrt(r * ky)

        # Equation 24
        kx_bar = -1j * np.cos(theta) * (tau**2 + self.beta_p) +\
                 tau * np.sin(theta) * sqrt(tau**2 + 2 * self.beta_p)
        # Equation 19
        kx = ky * kx_bar
        # Equation 5c
        k = sqrt(kx**2 + ky**2)

        # Equation 12
        v = sqrt(kx**2 + ky**2 * self.beta_p**2)
        v_prime = sqrt(kx**2 + ky**2 * self.beta_s**2)
        # Equation 17
        v_bar = sqrt(kx_bar**2 + self.beta_p**2)
        v_bar_prime = sqrt(kx_bar**2 + self.beta_s**2)

        ks = self.Ms * ky

        # Equation 9
        F, _ = self.__F_prime(k, ky, kx_bar, v, v_prime, v_bar, v_bar_prime)
        # Equation 8
        A = (2 * k**2 - ks**2)
        # Equation 14
        Ep = -v * A

        # diff kx_bar / tau
        root = sqrt(tau**2 + 2 * self.beta_p)
        dkx_bar_dtau = -2 * 1j * tau * np.cos(theta) + np.sin(theta) * (root + tau**2 / root)

        # Equation 27
        # np.exp(-r * ky * tau**2) is not needed because we are using Hermite quadrature which
        # already includes the exp(-tau^2) weight in the weights wi.
        integrand = (Ep / F) * dkx_bar_dtau  #* np.exp(-r * ky * tau**2)
        Gp = ky * np.exp(-r * ky * self.beta_p) * np.sum(integrand * wi) / sqrt(r * ky)

        if region != 'I':
            k_ = -np.sign(np.cos(theta)) * 1j * self.beta_r  # Equation 28
            vr_bar = sqrt(self.beta_p**2 - self.beta_r**2)  # Equation 30
            exp = np.exp(-ky * (vr_bar * z + self.beta_r * x * np.sign(np.cos(theta))))
            _, fct = self.__F_prime(k_, ky, k_, v, v_prime, v_bar, v_bar_prime)
            correction = -np.sign(np.cos(theta)) * 2 * np.pi * 1j * Ep / fct * exp  # Equation 30
            Gp += correction
        if region == 'III':
            # int = integrate.trapezoid(Ep / F * np.exp(-ky * (v_prime * z + 1j * kx_bar * x)), kx_bar)
            # correction = ky * int
            # Gp += correction
            sys.exit("Region III correction not implemented yet")
        return Gp

    def __integrand_Gs(self, x: float, z: float, ky: float, r: float, theta: float, region: str, ti: np.ndarray,
                       wi: np.ndarray) -> complex:
        """
        Compute the integrand Gs for the S-wave contribution to the vertical displacement.

        Args:
            - x (float): Horizontal distance from the load path (m)
            - z (float): Depth below the surface (m)
            - ky (float): Wavenumber for integration (1/m)
            - r (float): Distance from the load path to the point of interest (m)
            - theta (float): Angle between the point of interest and the load path (radians)
            - region (str): Region classification for S-wave contribution (I or II)
            - ti (np.ndarray): Hermite quadrature nodes
            - wi (np.ndarray): Hermite quadrature weights
        Returns:
            - Gs (complex): Value of the integrand Gs for the given parameters
        """

        # map Hermite nodes
        # Decay factor is exp(-r * ky * tau**2)
        tau = ti / np.sqrt(r * ky)

        # Equation 33
        kx_bar = -1j * np.cos(theta) * (tau**2 + self.beta_s) +\
                 tau * np.sin(theta) * sqrt(tau**2 + 2 * self.beta_s)
        # Equation 19
        kx = ky * kx_bar
        # Equation 5c
        k = sqrt(kx**2 + ky**2)

        # Equation 12
        v = sqrt(kx**2 + ky**2 * self.beta_p**2)
        v_prime = sqrt(kx**2 + ky**2 * self.beta_s**2)
        # Equation 17
        v_bar = sqrt(kx_bar**2 + self.beta_p**2)
        v_bar_prime = sqrt(kx_bar**2 + self.beta_s**2)

        # Equation 9
        F, _ = self.__F_prime(k, ky, kx_bar, v, v_prime, v_bar, v_bar_prime)
        # Equation 8
        C = 2 * v
        # Equation 14
        Es = k**2 * C

        # diff kx_bar / tau
        dkx_bar_dtau = -2 * 1j * tau * np.cos(theta) +\
                       sqrt(tau**2 + 2 * self.beta_s) * np.sin(theta) + \
                        tau**2 * np.sin(theta) / (sqrt(tau**2 + 2 * self.beta_s))

        # Equation 35
        # np.exp(-r * ky * tau**2) is not needed because we are using Hermite quadrature which
        # already includes the exp(-tau^2) weight in the weights wi.
        integrand = (Es / F) * dkx_bar_dtau  #* np.exp(-r * ky * tau**2)
        Gs = ky * np.exp(-r * ky * self.beta_s) * np.sum(integrand * wi) / sqrt(r * ky)

        if region == 'II':
            k_ = -np.sign(np.cos(theta)) * 1j * self.beta_r  # Equation 36
            vr_bar = sqrt(self.beta_s**2 - self.beta_r**2)  # Equation 36
            exp = np.exp(-ky * (vr_bar * z + self.beta_r * x * np.sign(np.cos(theta))))
            _, fct = self.__F_prime(k_, ky, k_, v, v_prime, v_bar, v_bar_prime)
            correction = -np.sign(np.cos(theta)) * 2 * np.pi * 1j * Es / fct * exp  # Equation 36
            Gs += correction

        return Gs

    def __F_prime(self, k, ky, kx_bar, v, v_prime, v_bar, v_bar_prime):

        # Equation 9
        F_k = (2 * k**2 - (self.Ms * ky)**2)**2 - 4 * k**2 * v * v_prime
        # Derivative of F with respect to kr (Equation 29)
        # F_k_prime = 4 * kx_bar * ky**2 * ((2 * (2 * k**2 - self.Ms**2) - 2 * v_bar * v_bar_prime) - k**2 *
        #                                   (v_bar / v_bar_prime + v_bar_prime / v_bar))
        term1 = 2 * k**2 - ky**2 * (self.Ms**2 + v_bar * v_bar_prime)
        term2 = 0.5 * k**2 * (v_bar / v_bar_prime + v_bar_prime / v_bar)
        F_k_prime = 8 * kx_bar * ky**2 * (term1 - term2)
        return F_k, F_k_prime

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
        xi = brentq(self.__rayleigh_func, 0, 0.99, args=(eta, ))
        return sqrt(xi) * self.cs


def main():
    """
    Reproduce Figure 7a from Liao et al. (2005) for the vertical displacement uz at a point (x, z) = (0, 10 m)
    """

    # Defined Constants from Paper
    c_s_target = 1000.0  # m/s
    rho = 2000.0  # kg/m^3
    nu = 0.25  # Poisson's ratio

    E = rho * (c_s_target**2) * 2 * (1 + nu)
    force = 1e6
    speed = 700

    x, y, z = 0.0, 0.0, 10
    time = np.linspace(-0.2, 0.2, num=100)

    model = MovingLoadElasticHalfSpace(E, nu, rho, force, speed)

    uz = []
    for t in time:
        print(f"t {t} / {time[-1]}")
        model.compute_vertical_displacement(x, y, z, t, ky_max=200.0, n_ky=2000)
        uz.append(np.real(model.vertical_displacement))

    radius = sqrt(x**2 + z**2)
    shear_modulus = model.G

    tau = model.cs * time / radius
    dimensionless_displacement = np.array(uz) * shear_modulus * radius / model.force

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))

    ax[0].plot(time, uz)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Vertical displacement uz (m)")
    ax[0].grid()

    ax[1].plot(tau, dimensionless_displacement)
    ax[1].set_xlabel("Dimensionless time τ = c_s t / r")
    ax[1].set_ylabel("Dimensionless displacement U_z = uz G r / F")
    ax[1].grid()
    plt.show()


if __name__ == "__main__":
    main()
