from create_geometry import generate_gmsh_mesh


def test_generate_mesh_2D():
    """
    checks whether mesh data generated for 2D and 3D geometries is not empty
    :return: -
    """
    # define the points of the surface as a list of tuples
    input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    # define the mesh size
    mesh_size = 0.1
    # set a name label for the surface
    name_label = "Soil Layer"
    # if True, saves mesh data to separate mdpa files
    save_file = False
    # if True, opens gmsh interface
    gmsh_interface = False
    # set a name for mesh output file
    mesh_output_name = "test_2D"

    # test 2D geometry
    # define geometry dimension; input "2" for 2D
    dims = 2
    # input depth of geometry if 3D
    depth = 0

    coord, node_tags, elem_types, elem_tags, node_tag_1D, node_tag_2D = generate_gmsh_mesh(input_points, depth,
                                                                                           mesh_size, dims, save_file,
                                                                                           name_label,
                                                                                           mesh_output_name,
                                                                                           gmsh_interface)

    assert coord != []  # check if node_coords is not empty
    assert node_tags != []  # check if node_tags is not empty
    assert elem_types != []  # check if elem_types is not empty
    assert elem_tags != []  # check if elemTags is not empty
    assert node_tag_1D != []  # check if elem_node_tags is not empty
    assert node_tag_2D != []  # check if elem_node_tags is not empty


def test_generate_mesh_3D():
    """
    checks whether mesh data generated for 2D and 3D geometries is not empty
    :return: -
    """
    
     # define the points of the surface as a list of tuples
    input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    # define the mesh size
    mesh_size = 0.1
    # set a name label for the surface
    name_label = "Soil Layer"
    # if True, saves mesh data to separate mdpa files
    save_file = False
    # if True, opens gmsh interface
    gmsh_interface = False
    # test 3D geometry
    # define geometry dimension; input "3" for 3D to extrude the 2D surface
    dims = 3
    # input depth of geometry if 3D
    depth = 1
    # set a name for mesh output file
    mesh_output_name = "test_3D"

    coord, node_tags, elem_types, elem_tags, node_tag_1D, node_tag_2D, node_tag_3D = generate_gmsh_mesh(input_points,
                                                                                                        depth,
                                                                                                        mesh_size, dims,
                                                                                                        save_file,
                                                                                                        name_label,
                                                                                                        mesh_output_name,
                                                                                                        gmsh_interface)

    assert coord != []  # check if node_coords is not empty
    assert node_tags != []  # check if node_tags is not empty
    assert elem_types != []  # check if elem_types is not empty
    assert elem_tags != []  # check if elemTags is not empty
    assert node_tag_1D != []  # check if elem_node_tags is not empty
    assert node_tag_2D != []  # check if elem_node_tags is not empty
    assert node_tag_3D != []  # check if elem_node_tags is not empty
