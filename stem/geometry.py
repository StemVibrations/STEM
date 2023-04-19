
from stem.gmsh_IO import GmshIO
from stem.kratos_IO import KratosIO

import numpy as np


class Point:
    def __init__(self):
        self.id = None
        self.coordinates = None

class Line:
    def __init__(self):
        self.id = None
        self.point_ids = None


class Surface:
    def __init__(self):
        self.id = None
        self.line_ids = None

class Volume:
    def __init__(self):
        self.id = None
        self.surface_ids = None

class Geometry:
    def __init__(self):

        self.points = None
        self.lines = None
        self.surfaces = None
        self.volumes = None

    def get_geometry_data_from_gmsh(self):
        #todo connect  to gmsh io and populate points, lines, surfaces, volumes
        pass


