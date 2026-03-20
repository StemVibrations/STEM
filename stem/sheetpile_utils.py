from typing import Sequence, List

from gmsh_utils import gmsh_IO
import numpy as np

from stem.structural_material import *
from stem.model_part import BodyModelPart


class SheetPileUtils:

    @staticmethod
    def add_sheetpile_by_coordinates(coordinates: Sequence[Sequence[float]], material_parameters: EulerBeam, name: str,
                                     gmsh_io_instance: gmsh_IO, body_model_parts: List[BodyModelPart]):
        """
        Add a sheet pile to the model by providing the coordinates of the two points defining the sheet pile.

        """
        if len(coordinates) != 2:
            raise ValueError("Coordinates for sheet pile should be a list of two points")

        gmsh_input = {name: {"coordinates": coordinates, "ndim": 1}}

        gmsh_io_instance.generate_geometry(gmsh_input, "")

        # create body model part
        body_model_part = BodyModelPart(name)
        material = StructuralMaterial(name, material_parameters)
        body_model_part.material = material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(gmsh_io_instance.geo_data, name)

        body_model_parts.append(body_model_part)

    @staticmethod
    def add_anchor_by_coordinates(coordinates: Sequence[Sequence[float]], material_parameters: Anchor,
                                  prestress_value: float, name: str, gmsh_io_instance: gmsh_IO,
                                  body_model_parts: List[BodyModelPart]):
        """
        Add an anchor to the model by providing the coordinates of the two points defining the anchor.

        """
        if len(coordinates) != 2:
            raise ValueError("Coordinates for anchor should be a list of two points")

        gmsh_input = {name: {"coordinates": coordinates, "ndim": 1}}

        gmsh_io_instance.generate_geometry(gmsh_input, "")

        orrientation_vector = np.array(coordinates[1]) - np.array(coordinates[0])
        orrientation_vector = orrientation_vector / np.linalg.norm(orrientation_vector)

        axial_stiffness = material_parameters.YOUNG_MODULUS

        x_stiffness = axial_stiffness * orrientation_vector[0]**2
        y_stiffness = axial_stiffness * orrientation_vector[1]**2

        equivalent_spring_damper_parameters = ElasticSpringDamper([x_stiffness, y_stiffness, 0], [0, 0, 0], [0, 0, 0],
                                                                  [0, 0, 0])
        equivalent_spring_damper_parameters._end_coordinates = coordinates

        # create body model part
        body_model_part = BodyModelPart(name)
        material = StructuralMaterial(name, equivalent_spring_damper_parameters)
        body_model_part.material = material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(gmsh_io_instance.geo_data, name)
        body_model_parts.append(body_model_part)

    @staticmethod
    def add_grout_by_coordinates(coordinates: Sequence[Sequence[float]], grout_parameters: EulerBeam,
                                 interface_parameters, name: str, gmsh_io_instance: gmsh_IO,
                                 body_model_parts: List[BodyModelPart]):
        """
        Add a grout to the model by providing the coordinates of the two points defining the grout.

        """
        if len(coordinates) != 2:
            raise ValueError("Coordinates for grout should be a list of two points")

        gmsh_input = {name: {"coordinates": coordinates, "ndim": 1}}

        gmsh_io_instance.generate_geometry(gmsh_input, "")

        # create body model part
        body_model_part = BodyModelPart(name)
        material = StructuralMaterial(name, grout_parameters)
        body_model_part.material = material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(gmsh_io_instance.geo_data, name)

        body_model_parts.append(body_model_part)

    @staticmethod
    def add_point_element_by_coordinates(coordinates: Sequence[Sequence[float]], material_parameters: NodalConcentrated,
                                         name: str, gmsh_io_instance: gmsh_IO, body_model_parts: List[BodyModelPart]):
        """
        Add a point element to the model by providing the coordinates of the point.

        """
        if len(coordinates) != 1:
            raise ValueError("Coordinates for point element should be a list of one point")

        gmsh_input = {name: {"coordinates": coordinates, "ndim": 0}}

        gmsh_io_instance.generate_geometry(gmsh_input, "")

        # create body model part
        body_model_part = BodyModelPart(name)
        material = StructuralMaterial(name, material_parameters)
        body_model_part.material = material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(gmsh_io_instance.geo_data, name)

        body_model_parts.append(body_model_part)
