from scipy import special
import numpy as np


class Boussinesq:
    """
    Analytical solutions for the Boussinesq equations.
    """

    def __init__(self, young_modulus, poisson_ratio, load_radius, surface_load_value):
        """
        Initialise the Boussinesq solution.

        :param young_modulus: Young's modulus
        :param poisson_ratio: Poisson's ratio
        """
        self.E = young_modulus
        self.nu = poisson_ratio
        self.load_radius = load_radius
        self.q = surface_load_value

    def calculate_vertical_stress_below_load_centre(self, depth):
        """
        Calculate vertical stress below the centre of a circular load. (Craig, 2013)

        :param depth: Depth below the load
        :return: Vertical stress
        """

        if depth == 0:
            return self.q

        return self.q * (1 - (1 / ((1 + (self.load_radius / depth)**2)**(3 / 2))))

    def calculate_vertical_displacement_on_surface(self, radial_distance):
        """
        Calculate vertical displacement below a point load.

        :param radial_distance: Radial distance from the load
        :param load_value: Load value
        :return: Vertical displacement
        """

        # to avoid division by zero at the edge of the load area
        if radial_distance == self.load_radius:
            radial_distance -= 1e-12

        # case outside the loaded area
        if radial_distance > self.load_radius:
            k = self.load_radius / radial_distance
            c1 = (radial_distance / self.load_radius * special.ellipe(k**2) -
                  (radial_distance**2 - self.load_radius**2) /
                  (radial_distance * self.load_radius) * special.ellipk(k**2))
            displacement = 4 * self.q * self.load_radius * (1 - self.nu**2) / (self.E * np.pi) * c1
        # case inside the loaded area
        else:
            k = radial_distance / self.load_radius
            displacement = 4 * self.q * self.load_radius * (1 - self.nu**2) / (self.E * np.pi) * special.ellipe(k**2)

        return displacement


if __name__ == '__main__':

    # example usage
    young_modulus = 20e6
    poisson_ratio = 0.3
    load_radius = 0.1
    surface_load_value = -10e3

    boussinesq_solution = Boussinesq(young_modulus, poisson_ratio, load_radius, surface_load_value)

    eps = 1e-7
    radial_distance = 0.2 + eps  # just outside the load radius

    radial_distances = np.linspace(0, 0.5, 1000)

    vertical_displacements = []
    for radial_distance in radial_distances:
        vertical_displacement = boussinesq_solution.calculate_vertical_displacement_on_surface(radial_distance)
        vertical_displacements.append(vertical_displacement)

    depths = np.linspace(1e-3, 2.0, 1000)
    vertical_stresses = []
    for depth in depths:
        vertical_stress = boussinesq_solution.calculate_vertical_stress_below_load_centre(depth)
        vertical_stresses.append(vertical_stress)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(radial_distances, vertical_displacements, label='Vertical Displacement')
    axes[0].set_xlabel('Radial Distance (m)')
    axes[0].set_ylabel('Vertical Displacement (m)')
    axes[1].plot(depths, vertical_stresses, label='Vertical Stress', color='orange')
    axes[1].set_xlabel('Depth (m)')
    axes[1].set_ylabel('Vertical Stress (Pa)')
    plt.show()

    # print(f"Vertical displacement at radial distance {radial_distance} m: {vertical_displacement} m")
