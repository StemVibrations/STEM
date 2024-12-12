import numpy as np


class HertzianContact():
    """
    Hertzian contact model

    Attributes:
        - contact_coeff (float): contact coefficient
        - contact_power (float): contact power
    """

    def __init__(self):
        """
        Constructor of the HertzianContact class
        """
        self.contact_coeff = None  # contact coefficient
        self.contact_power = None  # contact power

    def calculate_contact_force(self, du: np.ndarray) -> np.ndarray:
        """
        Calculate contact force

        Args:
            - du (np.ndarray): contact deformation

        Returns:
            - np.ndarray: contact force
        """

        contact_force = np.sign(-du) * np.nan_to_num((1 / self.contact_coeff * -du)**self.contact_power)
        return contact_force

    def calculate_contact_deformation(self, F: np.ndarray) -> np.ndarray:
        """
        Calculate contact deformation

        Args:
            - F (np.ndarray): contact force

        Returns:
            - np.ndarray: contact deformation
        """

        return np.sign(F) * self.contact_coeff * np.abs(F)**(1 / self.contact_power)
