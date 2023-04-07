import gmsh
import sys
from points_input import points, lc


# #define the points as a list of tuples
# points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
# #define the mesh size
# lc = 0.1


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Curve:
    def __init__(self, *args):
        for i, arg in enumerate(args):
            setattr(self, f'l{i + 1}', arg)


# gmsh functions
def create_point(input):
    x = input[0]
    y = input[1]
    z = input[2]
    lc = input[3]
    gmsh.model.geo.addPoint(x, y, z, lc)


def create_line(input):
    point1 = input[0]
    point2 = input[1]
    l = gmsh.model.geo.addLine(point1, point2)
    return l


#
def create_surface(shape):
    gmsh.model.geo.addCurveLoop(shape, 1)
    s = gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.setPhysicalName(2, s, "Soil layer")  # set a name label for the surface
    return s


def make_geometry(points, point_pairs):
    for i in range(len(points)):
        p = [points[i][0], points[i][1], points[i][2], lc]
        create_point(p)

    line_lists = []
    for i in range(len(point_pairs)):
        l = [point_pairs[i][0], point_pairs[i][1]]
        print(l)
        line_lists.append(i + 1)
        create_line(l)

    create_surface(line_lists)


def submit_callback(points, point_pairs):
    gmsh.initialize()
    gmsh.model.add("geometry")

    make_geometry(points, point_pairs)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("geometry.msh")
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


# initialize an empty list for pairs of consecutive points
point_pairs = []
for i in range(len(points) - 1):
    # select two consecutive points and store them in an array
    point_pair = [i + 1, i + 2]
    point_pairs.append(point_pair)

point_pairs.append([len(points), 1])
print("point pairs:", point_pairs)

submit_callback(points, point_pairs)

