import numpy as np


class HertzianContact():

    def __init__(self):
        self.contact_coeff = None  # contact coefficient
        self.contact_power = None  # contact power


    def calculate_contact_force(self, du):
        """
        Calculate contact force

        :param du: differential displacement
        :return:
        """

        contact_force = np.sign( -du) * np.nan_to_num((1 /self.contact_coeff * -du) ** self.contact_power)
        return contact_force


    def calculate_contact_deformation(self, F):
        """
        Calculate contact deformation

        :param F: contact force
        :return: contact deformation
        """
        
        return np.sign(F) * self.contact_coeff * np.abs(F) ** (1 / self.contact_power)
