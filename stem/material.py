
class Material:
    """
    Class containing material information about a body part, e.g. a soil layer or track components

    Attributes:
        name (str): name of the material
        parameters (dict): dictionary containing the material parameters

    """

    def __init__(self):
        self.name = ""
        self.parameters = {}