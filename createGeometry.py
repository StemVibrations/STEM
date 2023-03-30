import gmsh
lc = 1e-1

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
    def __init__(self, l1, l2, l3, l4):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

def make_geometry(points, np, lc, lines, nl, curves, nc):

    for i in range(np):
        p = [points[i].x, points[i].y, points[i].z, lc]
        create_point(p)
    for i in range(nl):
        l = [lines[i].p1, lines[i].p2]
        create_line(l)
    for i in range(nc):
        s = [curves[i].l1, curves[i].l2, curves[i].l3, curves[i].l4]
        print(s)
        create_surface(s)

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
    gmsh.model.geo.addCurveLoop([shape[0], shape[1], shape[2], shape[3]],1)
    s = gmsh.model.geo.addPlaneSurface([1],1)
    return s


# Initialize Gmsh
gmsh.initialize()

# Create some points
p1 = Point(0, 0, 0)
p2 = Point(1, 0, 0)
p3 = Point(1, 3, 0)
p4 = Point(0, 3, 0)

# Create some lines
l1 = Line([p1, p2])
l2 = Line([p2, p3])
l3 = Line([p3, p4])
l4 = Line([p4, p1])

# Create a surface
s = Surface([l4, l1, l2, l3])

# Create the entities in Gmsh
# p1.create()
# p2.create()
# p3.create()
# p4.create()
# l1.create()
# l2.create()
# l3.create()
# l4.create()
# s.create()

# Generate the mesh and save the Gmsh file
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("createGeometry.msh")

# Finalize Gmsh
gmsh.finalize()