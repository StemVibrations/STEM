import numpy as np


class TwoDofVehicle:
    """
    Computes the displacement of a two DOF vehicle crossing a simple supported beam.
    Displacement computed at middle spam. Solution assuming only first mode.

    Based on Biggs "Introduction do Structural Dynamics", pp pg 322
    """

    def __init__(self, t_step=0.001):
        """
        Initialise

        :param t_step: time step (default 0.001)
        """

        self.t_step = t_step  # time step

        # vehicle properties
        self.mv = []  # mass of unsprung vehicle part
        self.mw = []  # mass of sprung vehicle part
        self.speed = []  # speed
        self.k_vehicle = []  # stiffness spring vehicle
        self.c_vehicle = []  # damping spring vehicle

        # beam properties
        self.EI = []  # bending stiffness of the beam
        self.mass = []  # unit mass of the beam
        self.length = []  # length of the beam
        self.eig = []  # eigen frequency n mode

        # results
        self.time = []  # time
        self.displacement = []  # displacement (2d array: [beam ; vehicle])

        # constants
        self.g = 9.81  # gravity
        # newmark constants
        self.beta = 0.25
        self.gamma = 0.5
        return

    def vehicle(self, m1, m2, speed, k, c):
        """
        Define vehicle properties

        :param m1: mass of sprung vehicle part
        :param m2: mass of unsprung vehicle part
        :param speed: vehicle speed
        :param k: vehicle stiffness
        :param c: vehicle damping
        """

        self.mv = m1
        self.mw = m2
        self.speed = speed
        self.k_vehicle = k
        self.c_vehicle = c
        return

    def beam(self, E, I, rho, A, L):
        """
        Beam properties

        :param E: Young modulus
        :param I: Inertia
        :param rho: Density
        :param A: Area
        :param L: Length
        :return:
        """
        self.EI = E * I
        self.mass = rho * A
        self.length = L

        return

    def eigen_freq(self, n):
        """
        Computes eigen frequency for a simple supported beam

        :param n: n mode
        """
        self.eig = n**2 * np.pi**2 * np.sqrt(self.EI / (self.mass * self.length**4))
        return

    def compute(self):
        """
        Computes displacement of middle point of beam and vehicle
        """

        # computes first eigen frequency
        self.eigen_freq(1)

        # initial conditions
        acc = np.array([0, 0])
        vel = np.array([0, 0])

        # maximum time: corresponding to one passage
        time_max = self.length / self.speed

        # define time
        self.time = np.linspace(0, time_max, int(time_max / self.t_step))

        # result variable
        res = np.zeros((len(self.time), 2))

        # time step
        delta_t = self.time[1] - self.time[0]

        # constants for newmark
        a1 = 1 / (self.beta * delta_t**2)
        a2 = 1 / (self.beta * delta_t)
        a3 = 1 / (2 * self.beta) - 1
        a4 = self.gamma / (self.beta * delta_t)
        a5 = 1 - (self.gamma / self.beta)
        a6 = (1 - self.gamma / (2 * self.beta)) * delta_t

        # for each time
        for idx in range(1, len(self.time)):
            # auxiliar variable
            sin = np.sin(np.pi * self.speed * self.time[idx] / self.length)

            # stiffness matrix
            k11 = a1 * (self.mass * self.length / 2 +
                        self.mw * sin**2) + self.mass * self.length / 2 * self.eig**2 + self.k_vehicle * sin**2
            k12 = -self.k_vehicle * sin
            k21 = -self.k_vehicle * sin - self.c_vehicle * a4 * sin
            k22 = a1 * self.mv + self.k_vehicle + self.c_vehicle * a4
            kappa = np.array([[k11, k12], [k21, k22]])

            f1 = (self.mv + self.mw) * self.g * sin + (a1 * res[idx - 1, 0] + a2 * vel[0] +
                                                       a3 * acc[0]) * (self.mass * self.length / 2 + self.mw * sin**2)
            f2 = -self.mv * (-a1 * res[idx - 1, 1] - a2 * vel[1] - a3 * acc[1]) - \
                 self.c_vehicle * (-a4 * res[idx - 1, 1] + a5 * vel[1] + a6 * acc[1] - (-a4 * res[idx - 1, 0] + a5 * vel[0] + a6 * acc[0]) * sin)
            force = np.array([f1, f2])

            res[idx, :] = np.dot(np.linalg.inv(kappa), force)

            v = a4 * (res[idx, :] - res[idx - 1, :]) + a5 * vel + a6 * acc
            a = a1 * (res[idx, :] - res[idx - 1, :]) - a2 * vel - a3 * acc

            vel = v
            acc = a

        # assign to disp
        self.displacement = res
        return
