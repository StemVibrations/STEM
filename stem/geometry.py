import numpy as np
from gmsh_utils.gmsh_IO import GmshIO

from stem.kratos_IO import KratosIO


class Point:
    """
    A class to represent a point in space.

    Attributes:
        id (int or None): A unique identifier for the point.
        coordinates (Iterable or None): An iterable of floats representing the x, y and z coordinates of the point.
    """
    def __init__(self):
        self.id = None
        self.coordinates = None


class Line:
    """
    A class to represent a line in space.

    Attributes:
        id (int or None): A unique identifier for the line.
        point_ids (Iterable or None): An Iterable of two integers representing the ids of the points that make up the
            line.
    """

    def __init__(self):
        self.id = None
        self.point_ids = None


class Surface:
    """
    A class to represent a surface in space.

    Attributes:
        id (int or None): A unique identifier for the surface.
        line_ids (Iterable or None): An Iterable of three or more integers representing the ids of the lines that make
            up the surface.
    """
    def __init__(self):
        self.id = None
        self.line_ids = None


class Volume:
    """
    A class to represent a volume in a three-dimensional space.

    Attributes:
        id (int or None): A unique identifier for the volume.
        surface_ids (Iterable or None): An Iterable of four or more integers representing the ids of the surfaces that
            make up the volume.
    """
    def __init__(self):
        self.id = None
        self.surface_ids = None


class Geometry:
    """
    A class to represent a collection of geometric objects in a two- or three-dimensional space.

    Attributes:
        points (Iterable or None): An Iterable of Point objects representing the points in the geometry.
        lines (Iterable or None): An Iterable of Line objects representing the lines in the geometry.
        surfaces (Iterable or None): An Iterable of Surface objects representing the surfaces in the geometry.
        volumes (Iterable or None): An Iterable of Volume objects representing the volumes in the geometry.
    """
    def __init__(self):

        self.points = None
        self.lines = None
        self.surfaces = None
        self.volumes = None

    def get_geometry_data_from_gmsh(self):
        #todo connect  to gmsh io and populate points, lines, surfaces, volumes
        pass


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

