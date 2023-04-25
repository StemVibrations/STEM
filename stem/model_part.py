
from stem.material import Material

class ModelPart:
    """
    One part of the complete model, this can be a boundary condition, a loading or another special process
    like excavation.

    Attributes:
        name (str): name of the model part
        nodes (np.array or None): node id followed by node coordinates in an array
        elements (np.array or None): element id followed by connectivities in an array
        conditions (np.array or None): condition id followed by connectivities in an array
        parameters (dict): dictionary containing the model part parameters

    """
    def __init__(self):
        self.name = None

        self.nodes = None
        self.elements = None
        self.conditions = None
        self.parameters = {}

        pass


class BodyModelPart(ModelPart):
    """
    This class contains model parts which are part of the body, e.g. a soil layer or track components.

    Attributes:
        name (str): name of the model part
        nodes (np.array or None): node id followed by node coordinates in an array
        elements (np.array or None): element id followed by connectivities in an array
        conditions (np.array or None): condition id followed by connectivities in an array
        parameters (dict): dictionary containing the model part parameters
        material (Material): material of the model part

    """

    def __init__(self):
        super().__init__()

        self.material = Material()
