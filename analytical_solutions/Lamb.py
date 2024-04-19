import os
import sys
import numpy as np
from scipy.optimize import root
from scipy.integrate import trapezoid
import matplotlib.pylab as plt


class LoadType:
    Heaviside = "heaviside"
    Pulse = "pulse"


class Lamb:
    r"""
    Based on Verruijt: An Introduction to Soil Dynamics (Chapter 13: Section 13.2).

    The solution is found for normalised variables: tau = (c_s * t / r)  and u_bar = P / (2 * pi * G * r).
    Only vertical displacement is computed.
    In this way the solution only depends on Poisson ratio.
    The results for different radius and E are found as post-processing.
    """

    def __init__(self, nb_steps=1000, tol=0.005, tau_max=4, step_int=1000, pulse_samples=2):
        """
        Lamb wave solution for vertical displacement

        Args:
            nb_steps: number of steps for tau
            tol: small number for the integration
            tau_max: maximum value of tau
            step_int: number of steps for numerical integration
            pulse_samples: number of samples for pulse load
        """

        self.u = []
        self.poisson = []
        self.young = []
        self.rho = []
        self.time = []
        self.load = []
        self.load_type = []
        self.radius = []
        self.shear_modulus = []  # shear modulus
        self.cs = []  # shear wave velocity
        self.cr = []  # Rayleigh wave velocity
        self.eta = []  # ratio shear wave / compression wave velocity
        self.tau = []
        self.y = []
        self.nb_steps = int(nb_steps)  # for the discretisation of tau
        self.steps_int = int(step_int)  # number of steps for numerical integration
        self.tol = tol  # small number for the integration
        self.tau = np.linspace(0, tau_max, self.nb_steps)  # normalised time
        self.pulse_samples = pulse_samples  # number of samples for pulse load

        self.u = []  # displacement
        self.u_bar = []  # normalised displacement

    def material_properties(self, nu: float, rho: float, young: float):
        r"""
        Material properties

        Args:
            nu: Poisson ratio
            rho: density
            young: Young modulus
        """
        self.poisson = nu
        self.rho = rho
        self.young = young

    def loading(self, p: float, load_type: LoadType):
        r"""
        Load properties

        Args:
            p: load
            load_type: type of load (pulse or heaviside)
        """
        self.load = p
        self.load_type = load_type

    def solution(self, radius: list):
        r"""
        Compute the solution

        Args:
            radius: list of radius
        """
        self.radius = radius
        self.u = np.zeros((len(self.tau), len(radius)))
        self.time = np.zeros((len(self.tau), len(radius)))
        self.u_bar = np.zeros(len(self.tau))

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
        theta = np.linspace(0, np.pi / 2, self.steps_int)

        # for each tau
        for idx, ta in enumerate(self.tau):
            # if tau within these limits: evaluate integral
            if (ta > self.eta) & (ta <= 1):

                # G1 integral
                integral = []
                for t in theta:
                    y = np.sqrt(self.eta**2 + (ta**2 - self.eta**2) * np.sin(t)**2)
                    integral.append((1 - 2 * y**2)**2 * np.sin(t)**2 /
                                    (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))

                # perform integration
                integral = trapezoid(np.array(integral), theta)

                # compute C1
                G1 = (ta**2 - self.eta**2) * integral

                # normalised displacement
                self.u_bar[idx] = -G1 / np.pi**2

    def displacement_before_rayleigh(self):
        r"""
        Compute displacement before arrival of Rayleigh wave
        """

        # limits for integration
        theta = np.linspace(0, np.pi / 2, self.steps_int)

        # for each tau
        for idx, ta in enumerate(self.tau):
            # if tau within these limits: evaluate integral
            if (ta > 1) & (ta <= self.cr):
                # G1 and G2 integrals
                integral_1 = []
                integral_2 = []
                for t in theta:
                    y = np.sqrt(self.eta**2 + (ta**2 - self.eta**2) * np.sin(t)**2)
                    integral_1.append((1 - 2 * y**2)**2 * np.sin(t)**2 /
                                      (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))
                    y = np.sqrt(1 + (ta**2 - 1) * np.sin(t)**2)
                    integral_2.append(y**2 * (y**2 - self.eta**2) * np.sin(t)**2 /
                                      (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))

                # perform integration
                integral_1 = trapezoid(np.array(integral_1), theta)
                integral_2 = trapezoid(np.array(integral_2), theta)

                # compute G1 & G2
                G1 = (ta**2 - self.eta**2) * integral_1
                G2 = 4 * (ta**2 - 1) * integral_2

                # normalised displacement
                self.u_bar[idx] = -(G1 + G2) / np.pi**2

    def displacement_after_rayleigh(self):
        r"""
        Compute displacement after arrival of Rayleigh wave
        """

        # for each tau
        for idx, ta in enumerate(self.tau):
            # if tau within these limits: evaluate integral
            if ta > self.cr:
                # G1 and G2 integrals
                integral_1_1 = []
                integral_1_2 = []
                integral_2_1 = []
                integral_2_2 = []

                # limits for integration
                theta_r1 = np.arcsin(np.sqrt((self.cr**2 - self.eta**2) / (ta**2 - self.eta**2)))
                theta_r1_low = np.linspace(0, theta_r1 - self.tol, self.steps_int)
                theta_r1_hig = np.linspace(theta_r1 + self.tol, np.pi / 2, self.steps_int)

                theta_r2 = np.arcsin(np.sqrt((self.cr**2 - 1) / (ta**2 - 1)))
                theta_r2_low = np.linspace(0, theta_r2 - self.tol, self.steps_int)
                theta_r2_hig = np.linspace(theta_r2 + self.tol, np.pi / 2, self.steps_int)

                for t in range(self.steps_int):
                    y = np.sqrt(self.eta**2 + (ta**2 - self.eta**2) * np.sin(theta_r1_low[t])**2)
                    integral_1_1.append(
                        (1 - 2 * y**2)**2 * np.sin(theta_r1_low[t])**2 /
                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))
                    y = np.sqrt(self.eta**2 + (ta**2 - self.eta**2) * np.sin(theta_r1_hig[t])**2)
                    integral_1_2.append(
                        (1 - 2 * y**2)**2 * np.sin(theta_r1_hig[t])**2 /
                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 * (1 - self.eta**2) * y**6))

                    y = np.sqrt(1 + (ta**2 - 1) * np.sin(theta_r2_low[t])**2)
                    integral_2_1.append(y**2 * (y**2 - self.eta**2) * np.sin(theta_r2_low[t])**2 /
                                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 *
                                         (1 - self.eta**2) * y**6))
                    y = np.sqrt(1 + (ta**2 - 1) * np.sin(theta_r2_hig[t])**2)
                    integral_2_2.append(y**2 * (y**2 - self.eta**2) * np.sin(theta_r2_hig[t])**2 /
                                        (1 - 8 * y**2 + 8 * (3 - 2 * self.eta**2) * y**4 - 16 *
                                         (1 - self.eta**2) * y**6))

                # perform integration
                integral_1_1 = trapezoid(np.array(integral_1_1), theta_r1_low)
                integral_1_2 = trapezoid(np.array(integral_1_2), theta_r1_hig)
                integral_2_1 = trapezoid(np.array(integral_2_1), theta_r2_low)
                integral_2_2 = trapezoid(np.array(integral_2_2), theta_r2_hig)

                # compute G1 & G2
                G1 = (ta**2 - self.eta**2) * (integral_1_1 + integral_1_2)
                G2 = 4 * (ta**2 - 1) * (integral_2_1 + integral_2_2)

                # normalised displacement
                self.u_bar[idx] = -(G1 + G2) / np.pi**2

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
        # determine Rayleigh wave speed
        # root finder of Rayleigh wave
        dd = root(self.d_function, (1 - self.poisson) / 8, tol=1e-12)
        d = dd.x[0]
        self.cr = np.sqrt(1 + d)

    def results(self, plots=True, output_folder="./", file_name="results"):
        r"""
        Post-processing of the results

        Args:
            plots: if plots
            output_folder: output folder
            file_name: file name
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

        return

    def d_function(self, d):
        r"""
        Function to find Rayleigh wave speed
        """
        return d - ((1 - self.poisson) / 8) / ((1 + d) * (d + self.poisson))


if __name__ == "__main__":
    lmb = Lamb()
    lmb.material_properties(0.2, 2000, 100e3)
    lmb.loading(-1000, LoadType.Pulse)
    lmb.solution([3, 4, 5, 6])
    lmb.results(output_folder="Results", file_name="Pulse")

    lmb = Lamb()
    lmb.material_properties(0.2, 2000, 100e3)
    lmb.loading(-1000, LoadType.Heaviside)
    lmb.solution([3, 4, 5, 6])
    lmb.results(output_folder="Results", file_name="Heaviside")
