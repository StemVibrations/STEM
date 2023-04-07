import gmsh
import sys

# from points_input import points, lc

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
def create_point(input,gmsh_model):
    x = input[0]
    y = input[1]
    z = input[2]
    lc = input[3]
    gmsh_model.geo.addPoint(x, y, z, lc)


def create_line(input, gmsh_model):
    point1 = input[0]
    point2 = input[1]
    l = gmsh_model.geo.addLine(point1, point2)
    return l


def create_surface(shape, gmsh_model):
    gmsh.model.geo.addCurveLoop(shape, 1)
    s = gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh_model.setPhysicalName(2, s, "Soil layer")  # set a name label for the surface
    return s


def make_geometry(points, point_pairs, lc, gmsh_model):
    for i in range(len(points)):
        p = [points[i][0], points[i][1], points[i][2], lc]
        create_point(p, gmsh_model)

    line_lists = []
    for i in range(len(point_pairs)):
        l = [point_pairs[i][0], point_pairs[i][1]]
        line_lists.append(i + 1)
        #print("l=",l)
        create_line(l, gmsh_model)

    create_surface(line_lists, gmsh_model)


def extract_mesh_data(shape):
    # entities = gmsh.model.getEntities()
    # print("entities", entities)
    # for e in entities:
    #     # Dimension and tag of the entity:
    #     # print("e=", e)
    #     dim = e[0]
    #     tag = e[1]
    #     # print("dim=", dim)
    #     # print("tag=", tag)
    #     nodeTags, nodeCoords, nodeParams  = gmsh.model.mesh.getNodes(dim, tag)
    #     print(nodeCoords)
    #     type = gmsh.model.getType(dim, tag)
    #     print(type)

    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()  # nodes, elements
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
    numElem = sum(len(i) for i in elemTags)
    print(" - Mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
          " elements")

    coord = []
    for i in range(len(nodeCoords)):
        if i % 3 == 0:
            coord.append([nodeCoords[i], nodeCoords[i + 1], nodeCoords[i + 2]])

    nodetag1D = []
    for i in range(int(len(elemNodeTags[0]))):
        if i % 2 == 0:
            nodetag1D.append([elemNodeTags[0][i], elemNodeTags[0][i+1]])

    nodetag2D = []
    print(elemNodeTags)
    for i in range(int(len(elemNodeTags[1]))): # elemNodeTags[1] means 2D element node tags
        if shape == 3:
            if i % shape == 0:
                nodetag2D.append([elemNodeTags[1][i], elemNodeTags[1][i + 1], elemNodeTags[1][i + 2]])
        if shape == 4:
            if i % shape == 0:
                nodetag2D.append([elemNodeTags[1][i], elemNodeTags[1][i + 1], elemNodeTags[1][i + 2],\
                                  elemNodeTags[1][i + 3]])

    return coord, nodeTags, elemTypes, elemTags, nodetag2D, nodetag1D


def submit_callback(points, point_pairs):
    gmsh.initialize()
    gmsh.model.add("geometry")
    gmsh_model = gmsh.model
    make_geometry(points, point_pairs, lc, gmsh_model)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    if mesh_type == "triangular":
        shape = 3
    if mesh_type == "quad":
        shape = 4
    # extract mesh data
    node_coords, node_tags, elem_types, elemTags, elem_node_tags, nodetag1D = extract_mesh_data(shape)

    # print some mesh data for demonstration
    print("Node coordinates:")
    print(node_coords)
    print("Node tags:")
    print(node_tags)
    print("Element types:")
    print(elem_types)
    print("Element tags:")
    print("element tags 1D:", elemTags[0])
    print("element tags 2D:", elemTags[1])
    print("element tags 0D:", elemTags[2])
    print("Elem Node Tags 1D = lines")
    print(nodetag1D)
    print("Elem Node Tags 2D = Elements")
    print(elem_node_tags)

    gmsh.write("geometry.msh")

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()

if __name__ == '__main__':
    # define the points as a list of tuples
    points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    # define the mesh size
    lc = 2
    mesh_type = "triangular"

    # initialize an empty list for pairs of consecutive points
    point_pairs = []
    for i in range(len(points) - 1):
        # select two consecutive points and store them in an array
        point_pair = [i + 1, i + 2]
        point_pairs.append(point_pair)

    point_pairs.append([len(points), 1])
    print("point_pairs=",point_pairs)

    submit_callback(points, point_pairs)
