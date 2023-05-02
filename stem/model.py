from typing import List

from stem.model_part import ModelPart, BodyModelPart


class Model:
    """
    A class to represent the main model.

    Attributes:
        project_parameters (dict): A dictionary containing the project parameters.
        solver (Solver): The solver used to solve the problem.
        body_model_parts (list): A list containing the body model parts.
        process_model_parts (list): A list containing the process model parts.

    """
    def __init__(self):

        self.project_parameters = None
        self.solver = None
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []


