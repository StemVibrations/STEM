from typing import Dict, List, Any, Optional, Sequence
from abc import ABC, abstractmethod


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
        - coordinates (List[float]): An iterable of floats representing the x, y and z coordinates of the point.
    """
    def __init__(self, id: int):
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
        - :class:`GeometricalObjectABC`

    Attributes:
        - id (int): A unique identifier for the line.
        - point_ids (List[int]): An Iterable of two integers representing the ids of the points that make up the\
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
        - :class:`GeometricalObjectABC`

    Attributes:
        - __id (int): A unique identifier for the surface.
        - line_ids (List[int]): An Iterable of three or more integers representing the ids of the lines that make\
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
        - :class:`GeometricalObjectABC`

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
        - points (Optional[List[:class:`Point`]]): An Iterable of Point objects representing the points in the geometry.
        - lines (Optional[List[:class:`Line`]]): An Iterable of Line objects representing the lines in the geometry.
        - surfaces (Optional[List[:class:`Surface`]]): An Iterable of Surface objects representing the surfaces in the geometry.
        - volumes (Optional[List[:class:`Volume`]]): An Iterable of Volume objects representing the volumes in the geometry.
    """
    def __init__(self, points: Optional[List[Point]] = None, lines: Optional[List[Line]] = None,
                 surfaces: Optional[List[Surface]] = None, volumes: Optional[List[Volume]] = None):
        self.points: Optional[List[Point]] = points
        self.lines: Optional[List[Line]] = lines
        self.surfaces: Optional[List[Surface]] = surfaces
        self.volumes: Optional[List[Volume]] = volumes

    @staticmethod
    def __get_unique_entities_by_ids(entities: Sequence[GeometricalObjectABC]):
        """
        Returns a list of unique entities by their ids.

        Args:
            - entities (Sequence[:class:`GeometricalObjectABC`]): An Sequence of geometrical entities.

        Returns:
            - unique_entities (List[:class:`GeometricalObjectABC`): A list of unique geometrical entities entities.

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
            - point (:class:`Point`): The point object.
        """

        # create point
        point = Point(point_id)
        point.coordinates = geo_data["points"][point.id]
        return point

    @staticmethod
    def __set_line(geo_data: Dict[str,Any], line_id: int):
        """
        Creates a line from the geometry data.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - line_id (int): The id of the line to create.

        Returns:
            - line (:class:`Line`): The line object.
        """

        # Initialise point list
        points = []

        # create line and lower dimensional objects
        line = Line(abs(line_id))
        line.point_ids = geo_data["lines"][line.id]
        for point_id in line.point_ids:
            points.append(Geometry.__set_point(geo_data, point_id))
        return line, points

    @staticmethod
    def __create_surface(geo_data: Dict[str, Any], surface_id: int):
        """
        Creates a surface from the geometry data.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - surface_id (int): The id of the surface to create.

        Returns:
            - surface (:class:`Surface`): The surface object.
        """

        # Initialise point and line lists
        points = []
        lines = []

        # create surface and lower dimensional objects
        surface = Surface(abs(surface_id))
        surface.line_ids = geo_data["surfaces"][surface.id]
        for line_id in surface.line_ids:
            line, line_points = Geometry.__set_line(geo_data, line_id)

            lines.append(line)
            points.extend(line_points)

        return surface, lines, points

    @classmethod
    def create_geometry_from_gmsh_group(cls, geo_data: Dict[str, Any], group_name: str):
        """
        Initialises the geometry by parsing the geometry data from the geo_data dictionary.

        Args:
            - geo_data (Dict[str, Any]): A dictionary containing the geometry data as provided by gmsh_utils.
            - group_name (str): The name of the group to create the geometry from.

        Returns:
            - geometry (:class:`Geometry`): A Geometry object containing the geometric objects in the group.
        """

        # initialize point, line, surface and volume lists
        points = []
        lines = []
        surfaces = []
        volumes = []

        group_data = geo_data["physical_groups"][group_name]
        ndim_group = group_data["ndim"]

        if ndim_group == 0:
            # create points
            for id in group_data["geometry_ids"]:
                points.append(Geometry.__set_point(geo_data, id))
        elif ndim_group == 1:
            # create lines and lower dimensional objects
            for id in group_data["geometry_ids"]:
                line, line_points = Geometry.__set_line(geo_data, id)

                lines.append(line)
                points.extend(line_points)

        elif ndim_group == 2:
            # create surfaces and lower dimensional objects
            for id in group_data["geometry_ids"]:

                surface, surface_lines, surface_points = Geometry.__create_surface(geo_data, id)
                surfaces.append(surface)
                lines.extend(surface_lines)
                points.extend(surface_points)

        elif ndim_group == 3:
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

        # remove duplicates from points, lines, surfaces, volumes
        unique_volumes = Geometry.__get_unique_entities_by_ids(volumes)
        unique_surfaces = Geometry.__get_unique_entities_by_ids(surfaces)
        unique_lines = Geometry.__get_unique_entities_by_ids(lines)
        unique_points = Geometry.__get_unique_entities_by_ids(points)

        return cls(unique_points, unique_lines, unique_surfaces, unique_volumes)
