
from stem.material import Material

class ModelPart:
    """
    One part of the complete model, this can be a boundary condition, a loading or another special process
    like excavation
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
    """

    def __init__(self):
        super().__init__()

        self.material = Material()
