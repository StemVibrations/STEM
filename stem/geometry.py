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

        # Initialise point list
        points = []

        # create line and lower dimensional objects
        line_id = abs(line_id)
        line = Line.create(geo_data["lines"][line_id], line_id)
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

        # initialise geometry lists
        points = []
        lines = []
        surfaces = []
        volumes = []

        # add volumes to geometry
        for key, value in geo_data["volumes"].items():
            volumes.append(Volume.create(value,key))

        # add surfaces to geometry
        for key, value in geo_data["surfaces"].items():
            surfaces.append(Surface.create(value, key))

        # add lines to geometry
        for key, value in geo_data["lines"].items():
            lines.append(Line.create(value,key))

        # add points to geometry
        for key, value in geo_data["points"].items():
            points.append(Point.create(value,key))

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
                volume = Volume.create(geo_data["volumes"][id], id)

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
    def create_track_geometry(self, sleeper_distance: float, n_sleepers: int, origin_point, direction_vector):
        """
        Generates a track geometry. With rail and railpads.

        Args:
            sleeper_distance (float): distance between sleepers
            n_sleepers (int): number of sleepers
            origin_point (Point): origin point of the track
            direction_vector (np.array): direction vector of the track

        Returns:

        """

        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

        rotation_matrix = np.diag(normalized_direction_vector)


        rail_length = sleeper_distance * n_sleepers
        rail_end_coords = np.array([origin_point,
                                    origin_point + normalized_direction_vector * rail_length])

        # # rail_end_coords = np.array([rail_end_local_distance, y_local_coords, z_local_coords]).T
        # rail_end_coords = np.array([origin_point, end_global_coordinates]).T

        rail_local_distance = np.linspace(0, sleeper_distance * n_sleepers, n_sleepers + 1)
        sleeper_local_coords = np.copy(rail_local_distance)

        # todo kratos allows for a 0 thickness rail pad height, however gmsh needs to deal with fragmentation,
        # so we add a small height to prevent wrong fragmentation. Investigate the possibility to reset the thickness to
        # zero after the mesh is generated

        rail_pad_height = 0.1

        # todo transfer from local to global coordinates, currently local coordinates are used
        # global rail coordinates

        rail_global_coords = rail_local_distance[:,None].dot(normalized_direction_vector[None,:]) + origin_point

        # global sleeper coordinates

        sleeper_global_coords = sleeper_local_coords[:,None].dot(normalized_direction_vector[None,:]) + origin_point

        # y coord is vertical direction
        vertical_direction = 1
        sleeper_global_coords[:,vertical_direction] -= rail_pad_height


        gmsh_io = GmshIO()
        gmsh_io.create_lines_by_coordinates(rail_end_coords, name_label="rail")

        # add railpad lines in a loop such that the lines are not connected
        for rail_coord, sleeper_coord in zip(rail_global_coords, sleeper_global_coords):
            gmsh_io.create_lines_by_coordinates([rail_coord, sleeper_coord], name_label="railpads")

        # todo connect railpad lines to rail lines, gmsh provides functions to do this

        gmsh_io.extract_geo_data()
        geo_data = gmsh_io.geo_data

        # todo transfer gmsh geo data to kratos model part

        return geo_data

