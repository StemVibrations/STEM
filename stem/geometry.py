from dataclasses import dataclass
from abc import ABC
from typing import List, Iterable, Dict, Any


@dataclass
class Point:
    """
    A class to represent a point in space.

    Attributes:
        id (int or None): A unique identifier for the point.
        coordinates (Iterable or None): An iterable of floats representing the x, y and z coordinates of the point.
    """
    id: int
    coordinates: Iterable


@dataclass
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
        point_ids (Iterable or None): An Iterable of three or more integers representing
            the points that make up the surface (2D element).
    """

    def __init__(self):
        self.id = None
        self.point_ids = None


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


@dataclass
class Geometry:
    """
    A class to represent a collection of geometric objects in a two- or three-dimensional space.

    Attributes:
        points (List[Node]): List of Point objects representing the points in the
            geometry.
        lines (List[Line]): List of Line objects representing the lines in the geometry.
        surfaces (List[Surface]): List of Surface objects representing the surfaces in the geometry.
        volumes (List[Volume]): List of Volume objects representing the volumes in the geometry.
    """

    points: List[Point]
    lines: List[Line]
    surfaces: List[Surface]
    volumes: List[Volume]

    def get_geometry_data_from_gmsh(self):
        # todo connect  to gmsh io and populate points, lines, surfaces, volumes
        pass
