import gmsh
import sys

import numpy as np


def create_point(input):
    """
    creates points in gmsh
    :param input: gets points coordinates in order and mesh size from user
    :return: -
    """
    x = input[0]
    y = input[1]
    z = input[2]
    lc = input[3]
    gmsh.model.geo.addPoint(x, y, z, lc)


def create_line(input):
    """
    Creates lines in gmsh
    :param input: gets point tags in order
    :return: -
    """
    point1 = input[0]
    point2 = input[1]
    gmsh.model.geo.addLine(point1, point2)


def create_surface(line_list):
    """
    Creates curve and then surface in gmsh by using line tags
    :param line_list: gets line tags in order
    :return: returns the surface tag
    """
    gmsh.model.geo.addCurveLoop(line_list, 1)
    s = gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.setPhysicalName(2, s, "Soil layer")  # set a name label for the surface
    return s


def create_volume(s, depth):
    """
    Creates volume by extruding 2D surface
    :param s: surface tag
    :param depth: depth of 3D geometry
    :return: -
    """
    gmsh.model.geo.extrude([(2, s)], 0, 0, depth)


def make_geometry_3D(points, point_pairs, lc, depth):
    """
    Creates 3D geometries
    :param points: geometry points coordinates
    :param point_pairs: points paired for lines
    :param lc: mesh size
    :param depth: depth of 3D geometry
    :return: -
    """
    for i in range(len(points)):
        p = [points[i][0], points[i][1], points[i][2], lc]
        create_point(p)

    line_lists = []
    for i in range(len(point_pairs)):
        l = [point_pairs[i][0], point_pairs[i][1]] # begin and end of line with point tag
        line_lists.append(i + 1)
        create_line(l)

    s = create_surface(line_lists)
    create_volume(s, depth)


def make_geometry_2D(points, point_pairs, lc):
    """
    Creates 2D geometries
    :param points: geometry points coordinates
    :param point_pairs: points paired for lines
    :param lc: mesh size
    :return: -
    """
    for i in range(len(points)):
        p = [points[i][0], points[i][1], points[i][2], lc]
        create_point(p)

    line_lists = []
    for i in range(len(point_pairs)):
        l = [point_pairs[i][0], point_pairs[i][1]]
        line_lists.append(i + 1)
        create_line(l)

    create_surface(line_lists)


def extract_mesh_data(mesh_shape, dims):
    """
    Gets gmsh output data
    :param mesh_shape: for mesh_type 'triangular': 'mesh_shape=3' , 'quad': 'mesh_shape=4'
    :return: Geometry and Mesh data: node tags, node coordinates, element types, element tags 0D, 1D, 2D, 3D
    """

    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()  # nodes, elements
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
    numElem = sum(len(i) for i in elemTags)
    print(" - Mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
          " elements")
    #print(elemNodeTags)

    coord = []
    for i in range(len(nodeCoords)):
        if i % 3 == 0:
            coord.append([nodeCoords[i], nodeCoords[i + 1], nodeCoords[i + 2]])

    nodetag1D = []
    for i in range(int(len(elemNodeTags[0]))): # elemNodeTags[0] means 1D element node tags
        if i % 2 == 0:
            nodetag1D.append([elemNodeTags[0][i], elemNodeTags[0][i+1]])

    nodetag2D = []
    for i in range(int(len(elemNodeTags[1]))): # elemNodeTags[1] means 2D element node tags
        if mesh_shape == 3:
            if i % mesh_shape == 0:
                nodetag2D.append([elemNodeTags[1][i], elemNodeTags[1][i + 1], elemNodeTags[1][i + 2]])
        if mesh_shape == 4:
            if i % mesh_shape == 0:
                nodetag2D.append([elemNodeTags[1][i], elemNodeTags[1][i + 1], elemNodeTags[1][i + 2],\
                                  elemNodeTags[1][i + 3]])

    if dims == 3:
        nodetag3D = []
        for i in range(int(len(elemNodeTags[2]))): # elemNodeTags[2] means 3D element node tags
            if i % 4 == 0:
                nodetag3D.append([elemNodeTags[2][i], elemNodeTags[2][i + 1], elemNodeTags[2][i + 2],\
                                  elemNodeTags[2][i + 3]])

        return coord, nodeTags, elemTypes, elemTags, nodetag1D, nodetag2D, nodetag3D
    if dims == 2:
        return coord, nodeTags, elemTypes, elemTags, nodetag1D, nodetag2D

def submit_callback_3D(points, point_pairs, depth, lc, dims):

    """
    Initialize main and output for 3D geometries
    :param points: geometry points coordinates
    :param point_pairs: points paired for lines
    :param depth: depth of 3D geometry
    :param lc: mesh size
    :param dims: geometry dimension (2=2D or 3=3D)
    :return: saves mesh data and opens GMsh interface
    """

    gmsh.initialize()
    gmsh.model.add("geometry")
    make_geometry_3D(points, point_pairs, lc, depth)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dims)

    if mesh_type == "triangular":
        mesh_shape = 3
    if mesh_type == "quad":
        mesh_shape = 4
    # extract mesh data
    node_coords, node_tags, elem_types, elemTags, nodetag1D, nodetag2D, nodetag3D = extract_mesh_data(mesh_shape, dims)

    # print some mesh data for demonstration
    # print("Node coordinates:")
    # print(node_coords)
    # print("Node tags:")
    # print(node_tags)
    # print("Element types:")
    # print(elem_types)
    # print("Element tags:")
    # print("element tags 1D:", elemTags[0])
    # print("element tags 2D:", elemTags[1])
    # print("element tags 3D:", elemTags[2])
    # print("element tags 0D:", elemTags[3])
    # print("Elem Node Tags 1D = lines")
    # print(nodetag1D)
    # print("Elem Node Tags 2D = Surfaces")
    # print(nodetag2D)
    # print("Elem Node Tags 3D = Volumes")
    # print(nodetag3D)
    nodes = []
    for i in range (int(len(node_tags))):
        nodes.append([node_tags[i], node_coords[i][0], node_coords[i][1], node_coords[i][2]])
    lines = []
    for i in range (int(len(elemTags[0]))):
        lines.append([elemTags[0][i], nodetag1D[i][0], nodetag1D[i][1]])
    surfaces = []
    for i in range(int(len(elemTags[1]))):
        surfaces.append([elemTags[1][i], nodetag2D[i][0], nodetag2D[i][1], nodetag2D[i][1]])
    volumes = []
    for i in range (int(len(elemTags[2]))):
        volumes.append([elemTags[2][i], nodetag3D[i][0], nodetag3D[i][1], nodetag3D[i][2], nodetag3D[i][3]])

    np.savetxt('0.nodes.mdpa', nodes, delimiter=' ')
    np.savetxt('1.lines.mdpa', lines, delimiter=' ')
    np.savetxt('2.surfaces.mdpa', surfaces, delimiter=' ')
    np.savetxt('3.volumes.mdpa', volumes, delimiter=' ')

    # mesh file output
    gmsh.write("geometry.msh")
    # opens gmsh
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def submit_callback_2D(points, point_pairs, lc, dims):

    """
    Initialize main and output for 2D geometries
    :param points: geometry points coordinates
    :param point_pairs: points paired for lines
    :param lc: mesh size
    :param dims: geometry dimension (2=2D or 3=3D)
    :return: saves mesh data and opens GMsh interface
    """

    gmsh.initialize()
    gmsh.model.add("geometry")
    gmsh_model = gmsh.model
    make_geometry_2D(points, point_pairs, lc)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dims)

    if mesh_type == "triangular":
        mesh_shape = 3
    if mesh_type == "quad":
        mesh_shape = 4
    # extract mesh data
    node_coords, node_tags, elem_types, elemTags, nodetag1D, nodetag2D  = extract_mesh_data(mesh_shape, dims)

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
    print(nodetag2D)

    # mesh file output
    gmsh.write("geometry.msh")
    #opens gmsh
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


if __name__ == '__main__':
    # define the points as a list of tuples
    points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0)]
    # define the mesh size
    lc = 2
    # define mesh type
    mesh_type = "triangular"
    # define geometry dimension; input 3 for 3D, input 2 for 2D
    dims = 3
    #input depth of geometry if 3D
    depth = 1

    # initialize an empty list for pairs of consecutive points
    point_pairs = []
    for i in range(len(points) - 1):
        # select two consecutive points and store their tags in an array
        point_pair = [i + 1, i + 2] # creates point pairs by point tags
        point_pairs.append(point_pair)
    # make a pair that connects last point to first point
    point_pairs.append([len(points), 1])
    #print("point_pairs=",point_pairs)

    # dimension of geometry
    if dims == 3:
        extrude_depth = depth
        submit_callback_3D(points, point_pairs, extrude_depth, lc, dims)
    if dims == 2:
        submit_callback_2D(points, point_pairs, lc, dims)
