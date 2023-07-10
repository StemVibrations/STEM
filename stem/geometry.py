from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

from stem.IO.kratos_io import KratosIO

class GeometricalObjectABC(ABC):
    """
    An abstract base class for all geometrical objects.
    """

    @property
    @abstractmethod
    def id(self) -> int:
        """
        Abstract property for returning the id of the object.

        Raises:
            - Exception: cannot call abstract method.

        """
        raise Exception("Cannot call abstract method.")


class Point(GeometricalObjectABC):
    """
    A class to represent a point in space.

    Inheritance:
        - GeometricalObjectABC

    Attributes:
        - __id (Optional[int]): A unique identifier for the point.
        - coordinates (Iterable or None): An iterable of floats representing the x, y and z coordinates of the point.
    """
    def __init__(self, id):
        """
        Constructor for the point class.

        Args:
            id (int): The id of the point.
        """
        self.__id: int = id
        self.coordinates: List[float] = []

    @property
    def id(self) -> int:
        """
        Getter for the id of the point.

        Returns:
            - int: The id of the point.

        """
        return self.__id

    @id.setter
    def id(self, value: int):
        """
        Setter for the id of the point.

        Args:
            - value (int): The id of the point.

        """
        self.__id = value


class Line(GeometricalObjectABC):
    """
    A class to represent a line in space.

    Inheritance:
        - GeometricalObjectABC

    Attributes:
        - id (int or None): A unique identifier for the line.
        - point_ids (Iterable or None): An Iterable of two integers representing the ids of the points that make up the\
            line.
    """

    def __init__(self, id: int):
        """
        Constructor for the line class.

        Args:
            id (int): The id of the line.
        """
        self.__id: int = id
        self.point_ids: List[int] = []

    @property
    def id(self) -> int:
        """
        Getter for the id of the line.

        Returns:
            - int: The id of the line.
        """
        return self.__id

    @id.setter
    def id(self, value: int):
        """
        Setter for the id of the line.

        Args:
            - value (int): The id of the line.

        """
        self.__id = value


class Surface(GeometricalObjectABC):
    """
    A class to represent a surface in space.

    Inheritance:
        - GeometricalObjectABC

    Attributes:
        - __id (int): A unique identifier for the surface.
        - line_ids (Iterable or None): An Iterable of three or more integers representing the ids of the lines that make\
            up the surface.
    """
    def __init__(self, id: int):
        self.__id: int = id
        self.line_ids: List[int] = []

    @property
    def id(self) -> int:
        """
        Getter for the id of the surface.

        Returns:
            - int: The id of the surface.
        """
        return self.__id

    @id.setter
    def id(self, value: int):
        """
        Setter for the id of the surface.

        Args:
            - value (int): The id of the surface.

        """
        self.__id = value


class Volume(GeometricalObjectABC):
    """
    A class to represent a volume in a three-dimensional space.

    Inheritance:
        - GeometricalObjectABC

    Attributes:
        - __id (int): A unique identifier for the volume.
        - surface_ids (List[int]): An Iterable of four or more integers representing the ids of the surfaces that\
            make up the volume.
    """
    def __init__(self, id: int):
        self.__id: int = id
        self.surface_ids: List[int] = []

    @property
    def id(self) -> int:
        """
        Getter for the id of the volume.

        Returns:
            - int: The id of the volume.
        """
        return self.__id

    @id.setter
    def id(self, value: int):
        """
        Setter for the id of the volume.

        Args:
            - value (int): The id of the volume.

        """
        self.__id = value


class Geometry:
    """
    A class to represent a collection of geometric objects in a two- or three-dimensional space.

    Attributes:
        - points (Optional[List[Point]]): An Iterable of Point objects representing the points in the geometry.
        - lines (Optional[List[Line]]): An Iterable of Line objects representing the lines in the geometry.
        - surfaces (Optional[List[Surface]]): An Iterable of Surface objects representing the surfaces in the geometry.
        - volumes (Optional[List[Volume]]): An Iterable of Volume objects representing the volumes in the geometry.
    """
    def __init__(self, points: List[Point] = None, lines: List[Line] = None, surfaces: List[Surface] = None,
                 volumes: List[Volume] = None):
        self.points: Optional[List[Point]] = points
        self.lines: Optional[List[Line]] = lines
        self.surfaces: Optional[List[Surface]] = surfaces
        self.volumes: Optional[List[Volume]] = volumes

    @staticmethod
    def __get_unique_entities_by_ids(entities: List[GeometricalObjectABC]):
        """
        Returns a list of unique entities by their ids.

        Args:
            - entities (List[Union[Volume,Surface,Line,Point]): An Iterable of entities.

        Returns:
            - unique_entities (List[GeometricalObjectABC): A list of unique entities.

        """
        unique_entity_ids = []
        unique_entities = []
        for entity in entities:
            if entity.id not in unique_entity_ids:
                unique_entity_ids.append(entity.id)
                unique_entities.append(entity)
        return unique_entities

    @staticmethod
    def __create_surface(geo_data: Dict[str, Any], surface_id: int):
        """
        Creates a surface from the geometry data.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - surface_id (int): The id of the surface to create.

        Returns:
            - surface (Surface): The surface object.
        """

        # Initialise point and line lists
        points = []
        lines = []

        # create surface and lower dimensional objects
        surface = Surface(abs(surface_id))
        surface.line_ids = geo_data["surfaces"][surface.id]
        for line_id in surface.line_ids:
            line = Line(abs(line_id))
            line.point_ids = geo_data["lines"][line.id]
            for point_id in line.point_ids:
                point = Point(point_id)
                point.coordinates = geo_data["points"][point.id]
                points.append(point)
            lines.append(line)

        return surface, lines, points

    @classmethod
    def create_geometry_from_gmsh_group(cls, geo_data: Dict[str, Any], group_name: str):
        """
        Initialises the geometry by parsing the geometry data from the geo_data dictionary.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - group_name (str): The name of the group to create the geometry from.

        Returns:
            - geometry (Geometry): A Geometry object containing the geometric objects in the group.
        """

        # initialize point, line, surface and volume lists
        points = []
        lines = []
        surfaces = []
        volumes = []

        group_data = geo_data["physical_groups"][group_name]
        ndim_group = group_data["ndim"]

        if ndim_group == 3:
            # Create volumes and lower dimensional objects
            for id in group_data["geometry_ids"]:
                volume = Volume(id)
                volume.surface_ids = geo_data["volumes"][volume.id]

                # create surfaces and lower dimensional objects which are part of the current volume
                for surface_id in volume.surface_ids:
                    surface, surface_lines, surface_points = Geometry.__create_surface(geo_data, surface_id)
                    surfaces.append(surface)
                    lines.extend(surface_lines)
                    points.extend(surface_points)
                volumes.append(volume)

        elif ndim_group == 2:
            # create surfaces and lower dimensional objects
            for id in group_data["geometry_ids"]:

                surface, lines, points = Geometry.__create_surface(geo_data, id)
                surfaces.append(surface)
                lines.extend(lines)
                points.extend(points)

        # remove duplicates from points, lines, surfaces, volumes
        unique_volumes = Geometry.__get_unique_entities_by_ids(volumes)
        unique_surfaces = Geometry.__get_unique_entities_by_ids(surfaces)
        unique_lines = Geometry.__get_unique_entities_by_ids(lines)
        unique_points = Geometry.__get_unique_entities_by_ids(points)

        return cls(unique_points, unique_lines, unique_surfaces, unique_volumes)



if __name__ == '__main__':
    expected_points = {1: [0., 0., 0.], 2: [0.5, 0., 0.], 3: [0.5, 1., 0.], 4: [0., 1., 0.], 11: [0., 2., 0.],
                       12: [0.5, 2., 0.], 13: [0., 0., -0.5], 14: [0.5, 0., -0.5], 18: [0.5, 1., -0.5],
                       22: [0., 1., -0.5], 23: [0., 2., -0.5], 32: [0.5, 2., -0.5]}
    expected_lines = {5: [1, 2], 6: [2, 3], 7: [3, 4], 8: [4, 1], 13: [4, 11], 14: [11, 12], 15: [12, 3],
                      19: [13, 14], 20: [14, 18], 21: [18, 22], 22: [22, 13], 24: [1, 13], 25: [2, 14],
                      29: [3, 18], 33: [4, 22], 41: [23, 22], 43: [18, 32], 44: [32, 23], 46: [11, 23],
                      55: [12, 32]}
    expected_surfaces = {10: [5, 6, 7, 8], 17: [-13, -7, -15, -14], 26: [5, 25, -19, -24], 30: [6, 29, -20, -25],
                         34: [7, 33, -21, -29], 38: [8, 24, -22, -33], 39: [19, 20, 21, 22],
                         48: [-13, 33, -41, -46], 56: [-15, 55, -43, -29], 60: [-14, 46, -44, -55],
                         61: [41, -21, 43, 44]}
    expected_volumes = {1: [-10, 39, 26, 30, 34, 38], 2: [-17, 61, -48, -34, -56, -60]}
    expected_physical_groups = {'group_1': {'geometry_ids': [1], 'id': 1, 'ndim': 3},
                                'group_2': {'geometry_ids': [2], 'id': 2, 'ndim': 3}}

    geo_data = {"points": expected_points,
                "lines": expected_lines,
                "surfaces": expected_surfaces,
                "volumes": expected_volumes,
                "physical_groups": expected_physical_groups}

    geom = Geometry.create_geometry_from_gmsh_group(geo_data, "group_1")

    a=1+1