import gmsh
import sys
import pytest
from create_geometry_3D import make_geometry_3D, create_point_pairs
from create_geometry_3D import extract_mesh_data


def test_make_geometry():
    input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    mesh_size = 2
    dims = 3
    depth = 2
    name_label = "Soil Layer"

    point_pairs = create_point_pairs(input_points)
    gmsh.initialize()
    make_geometry_3D(input_points, point_pairs, mesh_size, depth, name_label)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dims)
    coord, node_tags, elem_types, elem_tags, node_tag_1D, node_tag_2D, node_tag_3D = extract_mesh_data(dims)

    assert coord != []  # check if node_coords is not empty
    assert node_tags != []  # check if node_tags is not empty
    assert elem_types != []  # check if elem_types is not empty
    assert elem_tags != []  # check if elemTags is not empty
    assert node_tag_1D != []  # check if elem_node_tags is not empty
    assert node_tag_2D != []  # check if elem_node_tags is not empty
    assert node_tag_3D != []  # check if elem_node_tags is not empty

