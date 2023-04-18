
from stem.material import Material

class ModelPart:
    def __init__(self):
        self.name = None

        self.nodes = None
        self.elements = None
        self.conditions = None
        self.parameters = {}

        pass


class BodyModelPart(ModelPart):

    def __init__(self):
        super().__init__()

        self.material = Material()
