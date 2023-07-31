from typing import Dict, List, Any, Optional, Sequence
from abc import ABC, abstractmethod

import numpy as np


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
        - :class:`GeometricalObjectABC`

    Attributes:
        - __id (int): A unique identifier for the point.
        - coordinates (Sequence[float]): A sequence of floats representing the x, y and z coordinates of the point.
    """
    def __init__(self, id: int):
        """
        Constructor for the point class.

        Args:
            id (int): The id of the point.
        """
        self.__id: int = id
        self.coordinates: Sequence[float] = []

    @classmethod
    def create(cls, coordinates: Sequence[float], id: int):
        """
        Creates a point object from a list of coordinates and a point id.

        Args:
            - coordinates (Sequence[float]): An iterable of floats representing the x, y and z coordinates of the point.
            - id (int): The id of the point.

        Returns:
            - :class:`Point`: A point object.

        """
        point = cls(id)
        point.coordinates = coordinates
        return point

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
        - :class:`GeometricalObjectABC`

    Attributes:
        - id (int): A unique identifier for the line.
        - point_ids (Sequence[int]): A sequence of two integers representing the ids of the points that make up the\
            line.
    """

    def __init__(self, id: int):
        """
        Constructor for the line class.

        Args:
            id (int): The id of the line.
        """
        self.__id: int = id
        self.point_ids: Sequence[int] = []

    @classmethod
    def create(cls, point_ids: Sequence[int], id: int):
        """
        Creates a line object from a list of point ids and a line id.

        Args:
            - point_ids (Sequence[int]): A sequence of two integers representing the ids of the points that make up the\
                line.
            - id (int): The id of the line.

        Returns:
            - :class:`Line`: A line object.

        """
        line = cls(id)
        line.point_ids = point_ids
        return line

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
        - :class:`GeometricalObjectABC`

    Attributes:
        - __id (int): A unique identifier for the surface.
        - line_ids (Sequence[int]): A sequence of three or more integers representing the ids of the lines that make\
            up the surface.
    """
    def __init__(self, id: int):
        self.__id: int = id
        self.line_ids: Sequence[int] = []

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

    @classmethod
    def create(cls, line_ids: Sequence[int], id: int):
        """
        Creates a surface object from a list of line ids and a surface id.

        Args:
            - line_ids (Sequence[int]): A sequence of three or more integers representing the ids of the lines that make\
                up the surface.
            - id (int): The id of the surface.

        Returns:
            - :class:`Surface`: A surface object.

        """
        surface = cls(id)
        surface.line_ids = line_ids
        return surface


class Volume(GeometricalObjectABC):
    """
    A class to represent a volume in a three-dimensional space.

    Inheritance:
        - :class:`GeometricalObjectABC`

    Attributes:
        - __id (int): A unique identifier for the volume.
        - surface_ids (Sequence[int]): A sequence of four or more integers representing the ids of the surfaces that\
            make up the volume.
    """
    def __init__(self, id: int):
        self.__id: int = id
        self.surface_ids: Sequence[int] = []

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

    @classmethod
    def create(cls, surface_ids: Sequence[int], id: int):
        """
        Creates a volume object from a list of surface ids and a volume id.

        Args:
            - surface_ids (Sequence[int]): A sequence of four or more integers representing the ids of the surfaces that\
                make up the volume.
            - id (int): The id of the volume.

        Returns:
            - :class:`Volume`: A volume object.

        """
        volume = cls(id)
        volume.surface_ids = surface_ids
        return volume


class Geometry:
    """
    A class to represent a collection of geometric objects in a zero-, one-, two- or three-dimensional space.

    Attributes:
        - points (Dict[int, :class:`Point`]): An dictionary of Point objects representing the points in the geometry.
        - lines (Dict[int, :class:`Line`]): A dictionary of Line objects representing the lines in the geometry.
        - surfaces (Dict[int, :class:`Surface`]): A dictionary of Surface objects representing the surfaces in the \
          geometry.
        - volumes (Dict[int, :class:`Volume`]): A dictionary of Volume objects representing the volumes in the geometry.
    """
    def __init__(self, points: Dict[int, Point] = {}, lines: Dict[int, Line] = {},
                 surfaces: Dict[int, Surface] = {}, volumes: Dict[int, Volume] = {}):
        self.points: Dict[int, Point] = points
        self.lines: Dict[int, Line] = lines
        self.surfaces: Dict[int, Surface] = surfaces
        self.volumes: Dict[int, Volume] = volumes

    @staticmethod
    def __get_unique_entities_by_ids(entities: Sequence[GeometricalObjectABC]):
        """
        Returns a list of unique entities by their ids.

        Args:
            - entities (Sequence[:class:`GeometricalObjectABC`]): An Sequence of geometrical entities.

        Returns:
            - Sequence[:class:`GeometricalObjectABC`]: A sequence of unique geometrical entities entities.

        """
        unique_entity_ids = []
        unique_entities = []
        for entity in entities:
            if entity.id not in unique_entity_ids:
                unique_entity_ids.append(entity.id)
                unique_entities.append(entity)
        return unique_entities

    @staticmethod
    def __set_point(geo_data: Dict[str, Any], point_id: int):
        """
        Creates a point from the geometry data.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - point_id (int): The id of the line to create.

        Returns:
            - :class:`Point`: The point object.
        """

        # create point
        return Point.create(geo_data["points"][point_id],point_id)

    @staticmethod
    def __set_line(geo_data: Dict[str,Any], line_id: int):
        """
        Creates a line from the geometry data.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - line_id (int): The id of the line to create.

        Returns:
            - Tuple[:class:`Line`, Sequence[:class:`Point`]]: The line object and the points that make up the line.
        """

        # create line and lower dimensional objects
        line_id = abs(line_id)
        line = Line.create(geo_data["lines"][line_id], line_id)

        # Using list comprehension to get the points
        points = [Geometry.__set_point(geo_data, point_id) for point_id in line.point_ids]

        return line, points

    @staticmethod
    def __create_surface(geo_data: Dict[str, Any], surface_id: int):
        """
        Creates a surface from the geometry data.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - surface_id (int): The id of the surface to create.

        Returns:
            - Tuple[:class:`Surface`, Sequence[:class:`Line`], Sequence[:class:`Point`]]: The surface object, \
                the lines that make up the surface and the points that make up the lines.
        """

        # Initialise point and line lists
        points = []
        lines = []

        # create surface and lower dimensional objects
        surface_id = abs(surface_id)
        surface = Surface.create(geo_data["surfaces"][surface_id], surface_id)
        for line_id in surface.line_ids:
            line, line_points = Geometry.__set_line(geo_data, line_id)

            lines.append(line)
            points.extend(line_points)

        return surface, lines, points

    @classmethod
    def create_geometry_from_geo_data(cls, geo_data: Dict[str,Any]):
        """
        Creates the geometry from gmsh geo_data

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.

        Returns:
            - :class:`Geometry`: The geometry object.
        """

        # initialise geometry dictionaries
        points = {}
        lines = {}
        surfaces = {}
        volumes = {}

        # add volumes to geometry
        for key, value in geo_data["volumes"].items():
            volumes[key] = Volume.create(value,key)

        # add surfaces to geometry
        for key, value in geo_data["surfaces"].items():
            surfaces[key] = Surface.create(value, key)

        # add lines to geometry
        for key, value in geo_data["lines"].items():
            lines[key] = Line.create(value,key)

        # add points to geometry
        for key, value in geo_data["points"].items():
            points[key] = Point.create(value,key)

        # create the geometry class
        return cls(points, lines, surfaces, volumes)

    @classmethod
    def create_geometry_from_gmsh_group(cls, geo_data: Dict[str, Any], group_name: str):
        """
        Initialises the geometry by parsing the geometry data from the geo_data dictionary.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - group_name (str): The name of the group to create the geometry from.

        Returns:
            - :class:`Geometry`: A Geometry object containing the geometric objects in the group.
        """

        # initialize point, line, surface and volume dictionaries
        points = {}
        lines = {}
        surfaces = {}
        volumes = {}

        group_data = geo_data["physical_groups"][group_name]
        ndim_group = group_data["ndim"]

        if ndim_group == 0:
            # create points
            for id in group_data["geometry_ids"]:
                points[id] = Geometry.__set_point(geo_data, id)
        elif ndim_group == 1:
            # create lines and lower dimensional objects
            for id in group_data["geometry_ids"]:
                line, line_points = Geometry.__set_line(geo_data, id)

                lines[id] = line
                for point in line_points:
                    points[point.id] = point

        elif ndim_group == 2:
            # create surfaces and lower dimensional objects
            for id in group_data["geometry_ids"]:

                surface, surface_lines, surface_points = Geometry.__create_surface(geo_data, id)
                surfaces[id] = surface
                for line in surface_lines:
                    lines[line.id] = line
                for point in surface_points:
                    points[point.id] = point

        elif ndim_group == 3:
            # Create volumes and lower dimensional objects
            for id in group_data["geometry_ids"]:
                volume = Volume.create(geo_data["volumes"][id], id)

                # create surfaces and lower dimensional objects which are part of the current volume
                for surface_id in volume.surface_ids:
                    surface, surface_lines, surface_points = Geometry.__create_surface(geo_data, surface_id)
                    surfaces[abs(surface_id)] = surface
                    for line in surface_lines:
                        lines[line.id] = line
                    for point in surface_points:
                        points[point.id] = point

                volumes[id] = volume

        return cls(points, lines, surfaces, volumes)
