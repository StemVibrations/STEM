from scipy import special
import numpy as np


class Boussinesq:
    """
    Analytical solutions for the Boussinesq equations.

    Attributes:
        - E (float): Young's modulus of the material [Pa]
        - nu (float): Poisson's ratio of the material [-]
        - load_radius (float): Radius of the circular load [m]
        - q (float): Surface load value [Pa]
    """

    def __init__(self, young_modulus: float, poisson_ratio: float, load_radius: float, surface_load_value: float):
        """
        Initialise the Boussinesq solution.

        Args:
            - young_modulus (float): Young's modulus of the material [Pa]
            - poisson_ratio (float): Poisson's ratio of the material [-]
            - load_radius (float): Radius of the circular load [m]
            - surface_load_value (float): Surface load value [Pa]
        """
        self.E: float = young_modulus
        self.nu: float = poisson_ratio
        self.load_radius: float = load_radius
        self.q: float = surface_load_value

    def calculate_vertical_stress_below_load_centre(self, depth: float) -> float:
        """
        Calculate vertical stress below the centre of a circular load. (Craig, 2013)

        Args:
            - depth (float): Depth below the load [m]

        Returns:
            - float: Vertical stress at the given depth [Pa]
        """

        if depth == 0:
            return self.q

        return self.q * (1 - (1 / ((1 + (self.load_radius / depth)**2)**(3 / 2))))

    def calculate_vertical_displacement_on_surface(self, radial_distance: float) -> float:
        """
        Calculate vertical displacement at the surface

        Args:
            - radial_distance (float): Radial distance from the load [m]

        Returns:
            - float: Vertical displacement at the given radial distance [m]
        """

        # to avoid division by zero when radial_distance == load_radius
        if radial_distance == self.load_radius:
            radial_distance -= 1e-12

        # case outside the loaded area ( Love, A. E. H. (1927) ?)
        if radial_distance > self.load_radius:
            k = self.load_radius / radial_distance
            c1 = (radial_distance / self.load_radius * special.ellipe(k**2) -
                  (radial_distance**2 - self.load_radius**2) /
                  (radial_distance * self.load_radius) * special.ellipk(k**2))
            displacement = 4 * self.q * self.load_radius * (1 - self.nu**2) / (self.E * np.pi) * c1

        # case inside the loaded area ( Timoshenko, S. P., & Goodier, J. N. (1951/1970). Theory of Elasticity. ? )
        else:
            k = radial_distance / self.load_radius
            displacement = 4 * self.q * self.load_radius * (1 - self.nu**2) / (self.E * np.pi) * special.ellipe(k**2)

        return displacement
