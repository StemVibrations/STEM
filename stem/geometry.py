import numpy as np

from stem.gmsh_IO import GmshIO
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


