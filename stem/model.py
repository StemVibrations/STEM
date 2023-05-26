from typing import List

from stem.model_part import ModelPart, BodyModelPart
from stem.geometry import Geometry

from gmsh_utils.gmsh_IO import GmshIO
import numpy as np

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
        self.geometry = Geometry()


    def generate_track(self, sleeper_distance: float, n_sleepers: int):
        """
        Generates a track geometry. With rail and railpads.

        Args:
            sleeper_distance (float): distance between sleepers
            n_sleepers (int): number of sleepers

        Returns:

        """

        origin_point = np.array([1, 1, 1])
        direction_vector = np.array([1, 2, 0])

        geo_data = self.geometry.create_track_geometry(sleeper_distance, n_sleepers, origin_point, direction_vector)


        return geo_data


if __name__ == '__main__':
    model = Model()

    rail_nodes = model.generate_track(0.6, 10)



    print(rail_nodes)

