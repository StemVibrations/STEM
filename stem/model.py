from typing import List

from stem.model_part import ModelPart, BodyModelPart


class Model:
    """
    Main model

    """
    def __init__(self):

        self.project_parameters = None
        self.solver = None
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []

