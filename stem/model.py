from typing import List

from model_part import ModelPart, BodyModelPart



class Model:
    def __init__(self):

        self.project_parameters = None
        self.solver = None
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []


