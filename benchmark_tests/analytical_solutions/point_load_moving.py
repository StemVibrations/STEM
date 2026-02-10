import numpy as np
from numpy.lib.scimath import sqrt
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy import integrate


class MovingLoadElasticHalfSpace:
    """
    Implementation of the steady-state response of an elastic half-space
    to a moving point load (Vertical Displacement uz).
    Based on Liao et al. (2005).
    """

    def __init__(self, E: float, nu: float, rho: float, force: float, speed: float):
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

    def compute_vertical_displacement(self,
                                      x: float,
                                      y: float,
                                      z: float,
                                      t: float,
                                      ky_max: float = 50.0,
                                      n_ky: int = 2000,
                                      n_tau: int = 100):
        """
        Compute the vertical displacement uz at a point (x, y, z) and time t due to a moving point load.

        Args:
            - x, y, z (float): Spatial coordinates of the observation point
            - t (float): Time at which to compute the displacement
            - ky_max (float): Maximum value of the wavenumber in the y-direction for integration
            - n_ky (int): Number of points for numerical integration in the y-direction
            - n_tau (int): Number of points for numerical integration in the tau-direction
        Returns:
            - uz (float): Vertical displacement at the specified point and time
        """

        #
        # ky = np.linspace(0, ky_max, n_ky)
        ky = np.geomspace(1e-4, ky_max, n_ky)
        delta_ky = ky[1] - ky[0]

        # Equation 11
        self.Mp = self.speed / self.cp
        self.Ms = self.speed / self.cs
        self.beta_p = sqrt(1 - self.Mp**2)
        self.beta_s = sqrt(1 - self.Ms**2)
        self.beta_r = sqrt(1 - (self.speed / self.cr)**2)

        # Moving coordinate
        y_prime = y - self.speed * t

        # tau limits for integration
        # Decay factor is exp(-r * ky * tau**2)
        # define convergence criterion
        # exp(r * ky * tau**2) < tol
        # r = sqrt(x**2 + z**2)
        # tol = 1e-12

        # Gp = np.zeros_like(ky, dtype=complex)
        # Gs = np.zeros_like(ky, dtype=complex)
        integrand_list = []
        ti, wi = np.polynomial.hermite.hermgauss(128)

        for i, k in enumerate(ky):
            if k <= 1e-6:  # skip ky=0 to avoid singularity
                continue
            # tau_max = np.sqrt(np.log(1 / tol) / (r * k))  # see line above for derivation
            # tau = np.linspace(-tau_max, tau_max, n_tau)
            Gp = self.__intregand_Gp(x, z, k, ti, wi)
            Gs = self.__intregand_Gs(x, z, k, ti, wi)

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

    def __intregand_Gp(self, x, z, ky, ti, wi):

        r = sqrt(x**2 + z**2)
        theta = np.arctan2(z, x)

        # map Hermite nodes
        tau = ti / np.sqrt(r * ky)

        # compute angles for region classification (Fig 3)
        theta_r = np.arccos(np.real(self.beta_r / self.beta_p))
        theta_s = np.arccos(np.real(self.beta_s / self.beta_p))

        if theta_r < theta < np.pi - theta_r:
            region = 'I'
        elif (0 <= theta < theta_s) or (np.pi - theta_s < theta <= np.pi):
            region = 'III'
            print(f"Region III: theta={theta:.2f} rad")
        else:
            region = 'II'

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

        # kp = self.Mp * ky
        ks = self.Ms * ky

        # Equation 9
        F, _ = self.__F_prime(k, kx, ky, kx_bar, v, v_prime, v_bar, v_bar_prime)
        # Equation 8
        A = (2 * k**2 - ks**2)
        # Equation 14
        Ep = -v * A

        # diff kx_bar / tau
        root = sqrt(tau**2 + 2 * self.beta_p)
        dkx_bar_dtau = -2 * 1j * tau * np.cos(theta) + np.sin(theta) * (root + tau**2 / root)

        # Equation 27
        # integrand = (Ep / F) * np.exp(-r * ky * tau**2) * dkx_bar_dtau
        integrand = (Ep / F) * dkx_bar_dtau  #* np.exp(-r * ky * tau**2)
        Gp = ky * np.exp(-r * ky * self.beta_p) * np.sum(integrand * wi) / sqrt(r * ky)
        # Gp = ky * np.exp(-r * ky * self.beta_p) * integrate.trapezoid(integrand, tau)

        if region == 'II' or region == 'III':
            k_ = -np.sign(np.cos(theta)) * 1j * self.beta_r  # Equation 28
            vr_bar = sqrt(self.beta_p**2 - self.beta_r**2)  # Equation 30
            exp = np.exp(-ky * (vr_bar * z + self.beta_r * x * np.sign(np.cos(theta))))
            _, fct = self.__F_prime(k_, kx, ky, k_, v, v_prime, v_bar, v_bar_prime)
            correction = -np.sign(np.cos(theta)) * 2 * np.pi * 1j * Ep / fct * exp  # Equation 30
            Gp += correction
        if region == 'III':
            int = integrate.trapezoid(Ep / F * np.exp(-ky * (v_prime * z + 1j * kx_bar * x)), kx_bar)
            correction = ky * int
            Gp += correction
        return Gp

    def __intregand_Gs(self, x, z, ky, ti, wi):

        r = sqrt(x**2 + z**2)
        theta = np.arctan2(z, x)

        # map Hermite nodes
        tau = ti / np.sqrt(r * ky)

        # compute angles for region classification (Fig 5)
        theta_r_star = np.arccos(np.real(self.beta_r / self.beta_s))

        if theta_r_star < theta < np.pi - theta_r_star:
            region = 'I'
        else:
            region = 'II'

        # Equation 33
        kx_bar = -1j * np.cos(theta) * (tau**2 + self.beta_s) + tau * np.sin(theta) * sqrt(tau**2 + 2 * self.beta_s)
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
        # kp = self.Mp * ky
        # ks = self.Ms * ky

        # Equation 9
        F, _ = self.__F_prime(k, kx, ky, kx_bar, v, v_prime, v_bar, v_bar_prime)
        # Equation 8
        C = 2 * v  #/ F
        # Equation 14
        Es = k**2 * C  #* np.exp(-v * z)

        # diff kx_bar / tau
        dkx_bar_dtau = -2 * 1j * tau * np.cos(theta) +\
                       sqrt(tau**2 + 2 * self.beta_s) * np.sin(theta) + \
                        tau**2 * np.sin(theta) / (sqrt(tau**2 + 2 * self.beta_s))

        # Equation 35
        integrand = (Es / F) * dkx_bar_dtau  #* np.exp(-r * ky * tau**2)
        Gs = ky * np.exp(-r * ky * self.beta_s) * np.sum(integrand * wi) / sqrt(r * ky)
        # integrand = (Es / F) * np.exp(-r * ky * tau**2) * dkx_bar_dtau
        # Gs = ky * np.exp(-r * ky * self.beta_s) * integrate.trapezoid(integrand, tau)

        if region == 'II':
            k_ = -np.sign(np.cos(theta)) * 1j * self.beta_r  # Equation 36
            vr_bar = sqrt(self.beta_s**2 - self.beta_r**2)  # Equation 36
            exp = np.exp(-ky * (vr_bar * z + self.beta_r * x * np.sign(np.cos(theta))))
            _, fct = self.__F_prime(k_, kx, ky, k_, v, v_prime, v_bar, v_bar_prime)
            correction = -np.sign(np.cos(theta)) * 2 * np.pi * 1j * Es / fct * exp  # Equation 36
            Gs += correction

        return Gs

    def __F_prime(self, k, kx, ky, kx_bar, v, v_prime, v_bar, v_bar_prime):

        # Equation 9
        F_k = (2 * k**2 - (self.Ms * ky)**2)**2 - 4 * k**2 * v * v_prime
        # Derivative of F with respect to kr (Equation 29)
        # NOTE: this equation is wrong the the paper.
        # F_k_prime = 8*k*sqrt(ky**2-(self.Mp*kx)/k+kx**2)*sqrt(ky**2-(self.Ms*kx)/k+kx**2)+ \
        #     (2*self.Mp*kx*sqrt(ky**2-(self.Ms*kx)/k+kx**2))/sqrt(ky**2-(self.Mp*kx)/k+kx**2)+\
        #         (2*self.Ms*kx*sqrt(ky**2-(self.Mp*kx)/k+kx**2))/sqrt(ky**2-(self.Ms*kx)/k+kx**2)
        #F_k_prime = ky**2 * (8 * kx_bar * (2 * k**2 - ky**2) * (self.Ms**2 + v_bar*v_bar_prime) + 4 * kx**2 * k**2 * (v_bar/v_bar_prime + v_bar_prime/v_bar))
        F_k_prime = 4 * kx_bar * ky**2 * ((2 * (2 * k**2 - self.Ms**2) - 2 * v_bar * v_bar_prime) - k**2 *
                                          (v_bar / v_bar_prime + v_bar_prime / v_bar))
        # F_k_prime = 4 * kx * (2 * (2 * k**2 - ks**2) - 2 * v * v_prime - k**2 * (v / v_prime + v_prime / v))
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
    # 1. Defined Constants from Paper
    c_s_target = 1000.0  # m/s
    rho = 2000.0  # kg/m^3
    nu = 0.25  # Poisson's ratio

    # 2. Back-calculate Young's Modulus E to match cs = 1000 m/s
    # Formula: cs = sqrt( G / rho ) and G = E / (2*(1+nu))
    # Therefore: E = rho * cs^2 * 2 * (1 + nu)
    E = rho * (c_s_target**2) * 2 * (1 + nu)  # Result: 5e9 Pa
    force = 1e6  # N
    speed = 700  # m/s

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
