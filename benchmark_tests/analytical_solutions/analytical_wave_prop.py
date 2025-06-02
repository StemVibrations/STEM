import matplotlib.pylab as plt
import numpy as np
import json
import os


class OneDimWavePropagation:
    def __init__(self, nb_cycles=10, nb_terms=250):
        """
        Analytical solution for the wave propagation on an elastic solid of finite length, when subjected to an pressure.

        Calculation of the solution of the wave equation is based on the analytical solution:
        Churchill, R.V. Operational Mathematics. 3rd edition. McGraw-Hill Book Company. 1972. pp: 253-257.
        """
        self.L = []  # Length of the solid column.
        self.k = []  # Bulk modulus
        self.rho = []  # Density
        self.p0 = []  # pressure boundary
        self.nb_elements = []  # number of elements
        self.nb_cycles = nb_cycles  # Number of cycles of the wave traveling wave
        self.nb_fourier_terms = nb_terms  # number of Fourier series terms

        self.c = []  # wave speed
        self.u = []  # displacements
        self.v = []  # velocities
        self.p = []  # stress
        self.time = []  # time
        self.result = []
        return

    def properties(self, rho, K, p0, L, nb_ele, time=None):
        r"""
        Assigns properties

        :param rho: Density solid
        :param K: Bulk modulus solid
        :param p0: Initial pressure on the solid boundary
        :param L: Length of the solid column
        :param nb_ele: Number of elements
        """
        self.rho = rho
        self.K = K
        self.p0 = p0
        self.L = L
        self.nb_elements = nb_ele
        self.time = time
        return

    def solution(self):
        """
        Solution following Fourier expansion
        """

        # wave speed
        self.c = np.sqrt(self.K / self.rho)
        # solid discretisation
        H_discre = np.linspace(0, self.L, int(self.nb_elements))
        # time discretisation
        if self.time is None:
            self.time = np.linspace(0, (self.nb_cycles * self.L / self.c), int(np.ceil(self.c / self.L) * 10))

        # variable initialisation: u = displacement; p = pressure
        self.u = np.zeros((H_discre.shape[0], self.time.shape[0]))
        self.v = np.zeros((H_discre.shape[0], self.time.shape[0]))
        self.p = np.zeros((H_discre.shape[0], self.time.shape[0]))

        for id_t, t in enumerate(self.time):
            for id_x, x, in enumerate(H_discre):
                summation = 0
                summation_p = 0
                summation_v = 0
                for k in range(1, self.nb_fourier_terms):
                    # Fourier terms
                    lambda_k = (2 * k - 1) * np.pi / (2 * self.L)
                    summation += (-1) ** k / (2 * k - 1) ** 2 * np.sin(lambda_k * x) * np.cos(lambda_k * self.c * t)
                    summation_p += (-1) ** k * lambda_k / (2 * k - 1) ** 2 * np.cos(lambda_k * x) * np.cos(
                        lambda_k * self.c * t)
                    summation_v -= (-1) ** k * lambda_k * self.c / (2 * k - 1) ** 2 * np.sin(lambda_k * x) * np.sin(
                        lambda_k * self.c * t)

                self.u[id_x, id_t] = self.p0 / self.K * (x + 8 * self.L / np.pi ** 2 * summation)
                self.p[id_x, id_t] = self.p0 / self.K * (1 + 8 * self.L / np.pi ** 2 * summation_p) * self.K
                self.v[id_x, id_t] = self.p0 / self.K * (8 * self.L / np.pi ** 2 * summation_v)

        return

    def write_results(self, output="./results_temp.json"):
        """
        Writes and saves output in a json file

        :param output: path to write json file (default "./results.json")
        """
        # create dictionary for results
        self.result = {"time": self.time.tolist(),
                       "v": self.v.tolist(),
                       "u": self.u.tolist()}

        # dump results
        with open(output, "w") as f:
            json.dump(self.result, f, indent=2)

        return


if __name__ == '__main__':

    p = OneDimWavePropagation(nb_terms=50)
    p.properties(1500, 30e6, 1000, 10, 20)
    p.solution()
    p.write_results()

    import matplotlib.pylab as plt
    plt.plot(p.time, p.u[19, :])
    plt.show()
