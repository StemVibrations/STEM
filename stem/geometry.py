from typing import Dict, Any, Sequence, List
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npty


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
            - id (int): The id of the point.
        """
        self.__id: int = id
        self.coordinates: Sequence[float] = []

    def __getattribute__(self, item: str) -> Any:
        """
        Overrides the getattribute method of the object class.

        Args:
            - item (str): The name of the attribute.

        Raises:
            - AttributeError: Cannot call create method on an initialised point instance.

        Returns:
            - Any: The attribute.

        """
        # make sure that the create method cannot be called on an initialised point instance
        if item == "create":
            raise AttributeError("Cannot call create method on an initialised point instance.")
        else:
            return super().__getattribute__(item)

    @classmethod
    def create(cls, coordinates: Sequence[float], id: int) -> "Point":
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

    def __getattribute__(self, item: str) -> Any:
        """
        Overrides the getattribute method of the object class.

        Args:
            - item (str): The name of the attribute.

        Raises:
            - AttributeError: Cannot call create method on an initialised line instance.

        Returns:
            - Any: The attribute.

        """
        # make sure that the create method cannot be called on an initialised line instance
        if item == "create":
            raise AttributeError("Cannot call create method on an initialised line instance.")
        else:
            return super().__getattribute__(item)

    @classmethod
    def create(cls, point_ids: Sequence[int], id: int) -> "Line":
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

    def __getattribute__(self, item: str) -> Any:
        """
        Overrides the getattribute method of the object class.

        Args:
            - item (str): The name of the attribute.

        Raises:
            - AttributeError: Cannot call create method on an initialised surface instance.

        Returns:
            - Any: The attribute.

        """
        # make sure that the create method cannot be called on an initialised surface instance
        if item == "create":
            raise AttributeError("Cannot call create method on an initialised surface instance.")
        else:
            return super().__getattribute__(item)

    @property
    def id(self) -> int:
        """
        Getter for the id of the surface.

        Returns:
            - int: The id of the surface.
        """
        return self.__id

    @classmethod
    def create(cls, line_ids: Sequence[int], id: int) -> "Surface":
        """
        Creates a surface object from a list of line ids and a surface id.

        Args:
            - line_ids (Sequence[int]): A sequence of three or more integers representing the ids of the lines that\
              make up the surface.
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

    def __getattribute__(self, item: str) -> Any:
        """
        Overrides the getattribute method of the object class.

        Args:
            - item (str): The name of the attribute.

        Raises:
            - AttributeError: Cannot call create method on an initialised volume instance.

        Returns:
            - Any: The attribute.

        """
        # make sure that the create method cannot be called on an initialised volume instance
        if item == "create":
            raise AttributeError("Cannot call create method on an initialised volume instance.")
        else:
            return super().__getattribute__(item)

    @property
    def id(self) -> int:
        """
        Getter for the id of the volume.

        Returns:
            - int: The id of the volume.
        """
        return self.__id

    @classmethod
    def create(cls, surface_ids: Sequence[int], id: int) -> "Volume":
        """
        Creates a volume object from a list of surface ids and a volume id.

        Args:
            - surface_ids (Sequence[int]): A sequence of four or more integers representing the ids of the surfaces \
                that make up the volume.
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

    def __init__(self,
                 points: Dict[int, Point] = {},
                 lines: Dict[int, Line] = {},
                 surfaces: Dict[int, Surface] = {},
                 volumes: Dict[int, Volume] = {}):
        self.points: Dict[int, Point] = points
        self.lines: Dict[int, Line] = lines
        self.surfaces: Dict[int, Surface] = surfaces
        self.volumes: Dict[int, Volume] = volumes

    def __getattribute__(self, item: str):
        """
        Overrides the getattribute method of the object class.

        Args:
            - item (str): The name of the attribute.

        Returns:
            - Any: The attribute.

        """
        # Make sure that the create_geometry_from_geo_data method  and the create_geometry_from_gmsh_group cannot be
        # called on an initialised geometry instance
        if item == "create_geometry_from_geo_data" or item == "create_geometry_from_gmsh_group":
            raise AttributeError(f"Cannot call class method: {item} from an initialised geometry instance.")
        else:
            return super().__getattribute__(item)

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
        return Point.create(geo_data["points"][point_id], point_id)

    @staticmethod
    def __set_line(geo_data: Dict[str, Any], line_id: int):
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
    def create_geometry_from_geo_data(cls, geo_data: Dict[str, Any]):
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
            volumes[key] = Volume.create(value, key)

        # add surfaces to geometry
        for key, value in geo_data["surfaces"].items():
            surfaces[key] = Surface.create(value, key)

        # add lines to geometry
        for key, value in geo_data["lines"].items():
            lines[key] = Line.create(value, key)

        # add points to geometry
        for key, value in geo_data["points"].items():
            points[key] = Point.create(value, key)

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

    def get_ordered_points_from_surface(self, surface_id: int) -> List[Point]:
        """
        Returns the points that make up the surface in the correct order, i.e. the order in which the points are
        connected by the lines that make up the surface.

        Args:
            - surface_id (int): The id of the surface.

        Returns:
            - List[:class:`Point`]: A sequence of points that make up the surface in the correct order.
        """
        surface = self.surfaces[abs(surface_id)]

        # initialize list of surface point ids
        surface_point_ids: List[int] = []

        # Get ordered list of point ids from the line connectivities for the surface
        for line_k in surface.line_ids:

            # get current line
            line = self.lines[abs(line_k)]

            # reverse line connectivity if line is defined in opposite direction
            line_connectivities = line.point_ids[::-1] if line_k < 0 else line.point_ids

            surface_point_ids.extend(
                [point_id for point_id in line_connectivities if point_id not in surface_point_ids])

        return [self.points[point_id] for point_id in surface_point_ids]

    def calculate_length_line(self, line_id: int) -> float:
        """
        Calculate the length of a line.

        Args:
            - line_id (int): The id of the line.

        Returns:
            - float: The length of the line.
        """
        point_coordinates = np.array(
            [self.points[point_id].coordinates for point_id in self.lines[abs(line_id)].point_ids])

        length_line = np.linalg.norm(point_coordinates[0, :] - point_coordinates[1, :])
        return float(length_line)

    def calculate_centroid_of_line(self, line_id: int) -> npty.NDArray[np.float64]:
        """
        Calculate the centroid of a line.

        Args:
            - line_id (int): The id of the line.

        Returns:
            - npty.NDArray[np.float64]: The coordinates of the centroid of the line.
        """
        point_coordinates = np.array(
            [self.points[point_id].coordinates for point_id in self.lines[abs(line_id)].point_ids], dtype=np.float64)

        centroid: npty.NDArray[np.float64] = np.mean(point_coordinates, axis=0)
        return centroid

    def calculate_centroid_of_surface(self, surface_id: int) -> npty.NDArray[np.float64]:
        """
        Calculate the centroid of a surface.

        Args:
            - surface_id (int): The id of the surface.

        Returns:
            - npty.NDArray[np.float64]: The coordinates of the centroid of the surface.
        """
        points = self.get_ordered_points_from_surface(surface_id)
        coordinates = np.array([point.coordinates for point in points])
        centroid: npty.NDArray[np.float64] = np.mean(coordinates, axis=0)

        return centroid

    def calculate_centre_of_mass_surface(self, surface_id: int) -> npty.NDArray[np.float64]:
        """
        Calculate the centre of mass of a surface. Where mass is given to the length of the lines that make up the
        surface. The centre of mass is then given by the 'O' in the following diagram:

        x---x---x---x---x
        |               |
        x       O       x
        |               |
        x---------------x

        Args:
            - surface_id (int): The id of the surface.

        Returns:
            - npty.NDArray[np.float64]: The coordinates of the centre of mass of the surface.
        """
        # initialize centre of mass and circumference
        centre_of_mass = np.zeros(3, dtype=np.float64)
        circumference = 0.0

        # calculate the centre of mass of the surface
        for line_k in self.surfaces[abs(surface_id)].line_ids:

            # calculate length current line
            weight = self.calculate_length_line(line_k)
            # calculate centroid of current line
            line_centroid = self.calculate_centroid_of_line(line_k)

            # add weighted centroid to centre of mass
            centre_of_mass += weight * line_centroid
            circumference += weight

        centre_of_mass /= circumference

        return centre_of_mass

    def calculate_area_surface(self, surface_id: int) -> float:
        """
        Calculate the area of a convex or concave surface in a 3D space using the shoelace algorithm.

        Args:
            - surface_id (int): The id of the surface.

        Returns:
            - float: The area of the surface.
        """

        points = self.get_ordered_points_from_surface(surface_id)
        coordinates = np.array([point.coordinates for point in points])

        # set the origin point as the first point of the first line
        origin_point = coordinates[0]

        # calculate the cross product of the vectors formed by the origin point and the other points
        cross_products = np.cross(coordinates[1:-1] - origin_point, coordinates[2:] - origin_point)

        # calculate the norms of the cross products
        norms = np.linalg.norm(cross_products, axis=1)

        # get first non-collinear points
        non_zero_norm_index = (norms != 0).argmax()
        non_collinear_points = [
            origin_point, coordinates[non_zero_norm_index + 1], coordinates[non_zero_norm_index + 2]
        ]

        # calculate the normal vector of the surface
        normal_vector = np.cross(non_collinear_points[1] - non_collinear_points[0],
                                 non_collinear_points[2] - non_collinear_points[0])

        # calculate the dot product of the cross products and the normal vector in order to determine the sign of the
        # area increment
        cross_dot_normal = np.dot(cross_products, normal_vector)

        # calculate the absolute value of the area of the surface
        area: float = abs(norms.dot(np.sign(cross_dot_normal)) * 0.5)

        return area

    def calculate_centre_of_mass_volume(self, volume_id: int) -> npty.NDArray[np.float64]:
        """
        Calculate the centre of mass of a volume. Where mass is given to the area of the surfaces that make up the
        volume.

        Args:
            - volume_id (int): The id of the volume.

        Returns:
            - npty.NDArray[np.float64]: The coordinates of the centre of mass of the volume.
        """
        # initialize centre of mass and volume
        centre_of_mass = np.zeros(3, dtype=np.float64)
        surface_area = 0.0

        # calculate the centre of mass of the volume
        for surface_id in self.volumes[abs(volume_id)].surface_ids:

            # calculate area current surface
            weight = self.calculate_area_surface(surface_id)
            # calculate centroid of current surface
            surface_centroid = self.calculate_centre_of_mass_surface(surface_id)

            # add weighted centroid to centre of mass
            centre_of_mass += weight * surface_centroid
            surface_area += weight

        centre_of_mass /= surface_area

        return centre_of_mass

    def get_all_coordinates(self) -> npty.NDArray[np.float64]:
        """
        Returns all coordinates of the points in the geometry.

        Returns:
            - npty.NDArray[np.float64]: A 2D array of shape (n_points, 3) containing the coordinates of the points.
        """
        return np.array([point.coordinates for point in self.points.values()], dtype=np.float64)
