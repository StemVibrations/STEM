import os
from typing import List, Optional
import numpy as np
import numpy.typing as npt
from scipy.optimize import root
from scipy.integrate import trapezoid
import matplotlib.pylab as plt


class LoadType:
    """
    Load types: Heaviside or Pulse
    """
    Heaviside = "heaviside"
    Pulse = "pulse"


class Pekeris:
    r"""
    Based on Verruijt: An Introduction to Soil Dynamics (Chapter 13: Section 13.2).

    The solution is found for normalised variables: tau = (c_s * t / r)  and u_bar = P / (2 * pi * G * r),
    where:
        - tau: normalised time
        - u_bar: normalised displacement
        - c_s: shear wave velocity
        - t: time
        - r: radius
        - P: load
        - G: shear modulus

    Only vertical displacement is computed.
    In this way the solution only depends on Poisson ratio.
    The results for different radius and E are found as post-processing.

    Attributes:
        - load (float): load amplitude
        - load_type (Optional[str]): type of load
        - radius (List[float]): list of radius
        - shear_modulus (float): shear modulus
        - young (float): Young modulus
        - poisson (float): Poisson ratio
        - rho (float): density
        - cs (float): shear wave velocity
        - cr (float): Rayleigh wave velocity
        - eta (float): ratio shear wave / compression wave velocity
        - nb_steps (int): number of steps for time discretisation
        - steps_int (int): number of steps for numerical integration
        - tol (float): small number for the integration
        - tau (npt.NDArray[np.float64]): normalised time
        - time (npt.NDArray[np.float64]): time
        - pulse_samples (int): number of samples for pulse load
        - u (npt.NDArray[np.float64]): displacement
        - u_bar (npt.NDArray[np.float64]): normalised displacement
    """

    def __init__(self,
                 nb_steps: int = 1000,
                 tol: float = 0.005,
                 tau_max: float = 4,
                 step_int: int = 1000,
                 pulse_samples: int = 2):
        """
        Pekeris wave solution for vertical displacement

        Args:
            - nb_steps (int): number of steps for time discretisation (default = 1000)
            - tol (float): tolerance number for the integration (default = 0.005)
            - tau_max (float): maximum value of tau (default = 4)
            - step_int (int): number of steps for numerical integration (default = 1000)
            - pulse_samples (int): number of samples for pulse load (default = 2)
        """
        self.load: float = np.nan
        self.load_type: Optional[str] = None
        self.radius: List[float] = []
        self.shear_modulus: float = np.nan  # shear modulus
        self.young: float = np.nan  # Young modulus
        self.poisson: float = np.nan  # Poisson ratio
        self.rho: float = np.nan  # density
        self.cs: float = np.nan  # shear wave velocity
        self.cr: float = np.nan  # Rayleigh wave velocity
        self.eta: float = np.nan  # ratio shear wave / compression wave velocity
        self.nb_steps: int = int(nb_steps)  # for the discretisation of tau
        self.steps_int: int = int(step_int)  # number of steps for numerical integration
        self.tol: float = tol  # small number for the integration
        self.tau: npt.NDArray[np.float64] = np.linspace(0, tau_max, self.nb_steps)  # normalised time
        self.time: npt.NDArray[np.float64] = np.empty(shape=(0, 0))  # time
        self.pulse_samples: int = pulse_samples  # number of samples for pulse load
        self.u: npt.NDArray[np.float64] = np.empty(shape=(0, 0))  # displacement
        self.u_bar: npt.NDArray[np.float64] = np.empty(self.tau.shape[0])  # normalised displacement

    def material_properties(self, nu: float, rho: float, young: float):
        r"""
        Material properties

        Args:
            - nu (float): Poisson ratio [-]
            - rho (float): density [kgm^-3]
            - young (float): Young modulus [Pa]
        """
        self.poisson = nu
        self.rho = rho
        self.young = young

    def loading(self, p: float, load_type: str):
        r"""
        Load properties

        Args:
            - p (float): load amplitude
            - load_type (str): type of load
        """
        self.load = p
        self.load_type = load_type

    def solution(self, radius: List[float]):
        r"""
        Compute the solution for the given radius

        Args:
            - radius (List[float]): list of radius
        """
        self.radius = radius
        self.u = np.zeros((len(self.tau), len(radius)))
        self.time = np.zeros((len(self.tau), len(radius)))

        # compute wave speed
        self.elastic_props()

        # determine arrival of compression wave
        # displacements are zero before it
        self.displacement_before_compression()

        # displacements between arrival of compression wave and shear wave
        self.displacement_before_shear()

        # displacement before arrival of Rayleigh wave
        self.displacement_before_rayleigh()

        # displacement after arrival of Rayleigh wave
        self.displacement_after_rayleigh()

        return

    def displacement_before_compression(self):
        r"""
        Compute displacement before arrival of compression wave
        """
        idx = np.where(self.tau <= self.eta)[0]
        self.u_bar[idx] = 0

    def displacement_before_shear(self):
        r"""
        Compute displacement before arrival of shear wave
        """
        # limits for integration
        theta_array = np.linspace(0, np.pi / 2, self.steps_int)

        # for each tau
        for idx, tau in enumerate(self.tau):
            # if tau within these limits: evaluate integral
            if (tau > self.eta) & (tau <= 1):

                # G1 integral
                integral = []
                for theta in theta_array:
                    y = np.sqrt(self.eta**2 + (tau**2 - self.eta**2) * np.sin(theta)**2)
                    integral.append((1 - 2 * y**2)**2 * np.sin(theta)**2 /
                                    (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))

                # perform integration
                integral = trapezoid(np.array(integral), theta_array)

                # compute G1 integral
                G1_integral = (tau**2 - self.eta**2) * integral

                # normalised displacement
                self.u_bar[idx] = -G1_integral / np.pi**2

    def displacement_before_rayleigh(self):
        r"""
        Compute displacement before arrival of Rayleigh wave
        """

        # limits for integration
        theta = np.linspace(0, np.pi / 2, self.steps_int)

        # for each tau
        for idx, tau in enumerate(self.tau):
            # if tau within these limits: evaluate integral
            if (tau > 1) & (tau <= self.cr):
                # G1 and G2 integrals
                integral_1 = []
                integral_2 = []
                for t in theta:
                    y = np.sqrt(self.eta**2 + (tau**2 - self.eta**2) * np.sin(t)**2)
                    integral_1.append((1 - 2 * y**2)**2 * np.sin(t)**2 /
                                      (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))
                    y = np.sqrt(1 + (tau**2 - 1) * np.sin(t)**2)
                    integral_2.append(y**2 * (y**2 - self.eta**2) * np.sin(t)**2 /
                                      (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))

                # perform integration
                integral_1 = trapezoid(np.array(integral_1), theta)
                integral_2 = trapezoid(np.array(integral_2), theta)

                # compute G1 & G2 integrals
                G1_integral = (tau**2 - self.eta**2) * integral_1
                G2_integral = 4 * (tau**2 - 1) * integral_2

                # normalised displacement
                self.u_bar[idx] = -(G1_integral + G2_integral) / np.pi**2

    def displacement_after_rayleigh(self):
        r"""
        Compute displacement after arrival of Rayleigh wave
        """

        # for each tau
        for idx, tau in enumerate(self.tau):
            # if tau within these limits: evaluate integral
            if tau > self.cr:
                # G1 and G2 integrals
                integral_1_1 = []
                integral_1_2 = []
                integral_2_1 = []
                integral_2_2 = []

                # limits for integration
                theta_r1 = np.arcsin(np.sqrt((self.cr**2 - self.eta**2) / (tau**2 - self.eta**2)))
                theta_r1_low = np.linspace(0, theta_r1 - self.tol, self.steps_int)
                theta_r1_high = np.linspace(theta_r1 + self.tol, np.pi / 2, self.steps_int)

                theta_r2 = np.arcsin(np.sqrt((self.cr**2 - 1) / (tau**2 - 1)))
                theta_r2_low = np.linspace(0, theta_r2 - self.tol, self.steps_int)
                theta_r2_high = np.linspace(theta_r2 + self.tol, np.pi / 2, self.steps_int)

                for t in range(self.steps_int):
                    y = np.sqrt(self.eta**2 + (tau**2 - self.eta**2) * np.sin(theta_r1_low[t])**2)
                    integral_1_1.append(
                        (1 - 2 * y**2)**2 * np.sin(theta_r1_low[t])**2 /
                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))
                    y = np.sqrt(self.eta**2 + (tau**2 - self.eta**2) * np.sin(theta_r1_high[t])**2)
                    integral_1_2.append(
                        (1 - 2 * y**2)**2 * np.sin(theta_r1_high[t])**2 /
                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))

                    y = np.sqrt(1 + (tau**2 - 1) * np.sin(theta_r2_low[t])**2)
                    integral_2_1.append(y**2 * (y**2 - self.eta**2) * np.sin(theta_r2_low[t])**2 /
                                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 *
                                         (1 - self.eta**2) * y**6))
                    y = np.sqrt(1 + (tau**2 - 1) * np.sin(theta_r2_high[t])**2)
                    integral_2_2.append(y**2 * (y**2 - self.eta**2) * np.sin(theta_r2_high[t])**2 /
                                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 *
                                         (1 - self.eta**2) * y**6))

                # perform integration
                integral_1_1 = trapezoid(np.array(integral_1_1), theta_r1_low)
                integral_1_2 = trapezoid(np.array(integral_1_2), theta_r1_high)
                integral_2_1 = trapezoid(np.array(integral_2_1), theta_r2_low)
                integral_2_2 = trapezoid(np.array(integral_2_2), theta_r2_high)

                # compute G1 & G2
                G1_integral = (tau**2 - self.eta**2) * (integral_1_1 + integral_1_2)
                G2_integral = 4 * (tau**2 - 1) * (integral_2_1 + integral_2_2)

                # normalised displacement
                self.u_bar[idx] = -(G1_integral + G2_integral) / np.pi**2

    def elastic_props(self):
        r"""
        Compute elastic properties of the solid material
        """

        # shear modulus
        self.shear_modulus = self.young / (2 * (1 + self.poisson))
        # shear wave
        self.cs = np.sqrt(self.shear_modulus / self.rho)
        # ratio of wave velocities
        self.eta = np.sqrt((1 - 2 * self.poisson) / (2 * (1 - self.poisson)))
        # determine Rayleigh wave speed: root finder of Rayleigh wave
        dd = root(self.d_function, (1 - self.poisson) / 8, tol=1e-12)
        d = dd.x[0]
        self.cr = np.sqrt(1 + d)

    def results(self, plots: bool = True, output_folder: str = "./", file_name: str = "results"):
        r"""
        Post-processing of the results

        Args:
            - plots (bool): checks if make plots (default = True)
            - output_folder (str): folder to save the results (default = "./")
            - file_name (str): name of the file (default = "results")
        """

        # if load type pulse:
        # subtract two solutions with self.pulse_samples
        if self.load_type == "pulse":
            aux = np.zeros(len(self.u_bar))
            aux[self.pulse_samples:] = self.u_bar[:-self.pulse_samples]
            self.u_bar -= aux

        # scale the results for each radius
        for i, r in enumerate(self.radius):
            self.u[:, i] = self.u_bar * self.load / (self.shear_modulus * r)
            self.time[:, i] = self.tau * r / self.cs

        # if plots
        if plots:
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)
            if not file_name:
                file_name = "results"
            # plot normalised
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.set_position([0.20, 0.15, 0.75, 0.8])
            plt.rcParams.update({'font.size': 10})
            ax.plot(self.tau, self.u_bar)
            ax.set_xlim((0, np.max(self.tau)))
            # ax.set_ylim((0.15, -0.25))
            ax.set_xlabel(r"$\tau$")
            ax.set_ylabel(r"$\bar{u}$")
            ax.grid()
            plt.savefig(os.path.join(output_folder, f"{file_name}_normalised.png"))
            plt.close()

            # plot magnitude of results for all radius
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.set_position([0.20, 0.15, 0.75, 0.8])
            plt.rcParams.update({'font.size': 10})

            for i, r in enumerate(self.radius):
                ax.plot(self.time[:, i], self.u[:, i], label=f"Radius: {r} m")

            ax.set_xlim((0, np.max(np.max(self.time))))
            # ax.set_ylim((0.5, -0.5))
            ax.grid()
            ax.set_xlabel(r"Time [s]")
            ax.set_ylabel(r"Vertical displacement [m]")
            ax.legend(loc="upper right")
            plt.savefig(os.path.join(output_folder, f"{file_name}_displacement.png"))
            plt.close()

    def d_function(self, d: float) -> float:
        r"""
        Function to compute the residual of the Rayleigh wave speed

        Args:
            - d (float): parameter to compute the residual of the Rayleigh speed

        Returns:
            - float: residual of the Rayleigh speed
        """
        return d - ((1 - self.poisson) / 8) / ((1 + d) * (d + self.poisson))


if __name__ == "__main__":
    lmb = Pekeris()
    lmb.material_properties(0.2, 2000, 100e3)
    lmb.loading(-1000, LoadType.Pulse)
    lmb.solution([3, 4, 5, 6])
    lmb.results(output_folder="Results", file_name="Pulse")

    lmb = Pekeris()
    lmb.material_properties(0.2, 2000, 100e3)
    lmb.loading(-1000, LoadType.Heaviside)
    lmb.solution([3, 4, 5, 6])
    lmb.results(output_folder="Results", file_name="Heaviside")
