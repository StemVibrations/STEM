import gmsh
import sys
import pytest
from create_geometry_mesh_data import make_geometry
from create_geometry_mesh_data import extract_mesh_data


def test_make_geometry():
    mesh_type = "triangular"
    lc = 2
    points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    point_pairs = []
    for i in range(len(points) - 1):
        # select two consecutive points and store them in a list
        point_pair = [i + 1, i + 2]
        point_pairs.append(point_pair)

    point_pairs.append([len(points), 1])
    gmsh.initialize()

    make_geometry(points, point_pairs, lc, gmsh.model)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    node_coords, node_tags, elem_types, elemTags, elem_node_tags, nodetag1D = extract_mesh_data(3)

    assert node_coords != [] # check if node_coords is not empty
    assert node_tags != [] # check if node_tags is not empty
    assert elem_types != [] # check if elem_types is not empty
    assert elemTags != [] # check if elemTags is not empty
    assert elem_node_tags != [] # check if elem_node_tags is not empty