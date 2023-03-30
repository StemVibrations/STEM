import gmsh
import sys

"""
test points
p1 = [0, 0, 0, lc]
p2 = [1, 0, 0, lc]
p3 = [1, 3, 0, lc]
p4 = [0, 3, 0, lc]

test lines
l1 = create_line(1, 2)
l2 = create_line(2, 3)
l3 = create_line(3, 4)
l4 = create_line(4, 1)

test surface
s = [1, 2, 3, 4]
"""

# coordinates separated by tab, defined in points.txt file
lc = 0.1


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


with open("points.txt", "r") as f:
    file1 = f.read()

lines = file1.split("\n")
points = []

# loop through each line in the file
for line in lines:
    coords = line.split()
    point = (float(coords[0]), float(coords[1]), float(coords[2]))
    points.append(point)

# print the points
print("points:", points)

# initialize an empty list for pairs of consecutive points
point_pairs = []
for i in range(len(points) - 1):
    # select two consecutive points and store them in an array
    point_pair = [i + 1, i + 2]
    point_pairs.append(point_pair)

point_pairs.append([len(points), 1])
print("point pairs:", point_pairs)

submit_callback(points, point_pairs)

