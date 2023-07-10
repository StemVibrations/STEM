from typing import List

from stem.model_part import ModelPart, BodyModelPart

from gmsh_utils import gmsh_IO

class Model:
    """
    A class to represent the main model.

    Attributes:
        - project_parameters (dict): A dictionary containing the project parameters.
        - solver (Solver): The solver used to solve the problem.
        - body_model_parts (list): A list containing the body model parts.
        - process_model_parts (list): A list containing the process model parts.

    """
    def __init__(self):
        self.ndim = None
        self.project_parameters = None
        self.solver = None
        self.geometry = None
        self.mesh = None
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []


    def add_soil_layer(self, coordinates, material_parameters, name, extrusion_length=None):
        """
        Adds a soil layer to the model.

        Args:
            - coordinates (np.array): The coordinates of the soil layer.
            - material_parameters (dict): A dictionary containing the material parameters.

        """
        gmsh_io = gmsh_IO.GmshIO()
        if self.ndim == 2:
            gmsh_io.make_geometry_2d(coordinates, name)

        if self.ndim == 3 and extrusion_length is None:
            raise ValueError("extrusion_length must be specified for 3D models")

        body_model_part = BodyModelPart()
        body_model_part.name = name
        body_model_part.material = material_parameters

        body_model_part.get_geometry_from_geo_data(gmsh_io.geo_data, name)

        self.body_model_parts.append(body_model_part)


if __name__ == '__main__':
    coordinates = [[0, 0,0], [1, 0,0], [1, 1,0], [0, 1,0]]





