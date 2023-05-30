from gmsh_utils.gmsh_IO import GmshIO

def test_generate_mesh_2D():
    """
    Checks whether mesh data generated for 2D geometries is not empty.

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
    # set output directory of the mesh
    mesh_output_dir = "."

    # test 2D geometry
    # define geometry dimension; input "2" for 2D
    dims = 2
    # input depth of geometry if 3D
    extrusion_length = [0, 0, 0]

    gmsh_io = GmshIO()
    gmsh_io.generate_gmsh_mesh(input_points, extrusion_length, mesh_size, dims, name_label,
                               mesh_output_name, mesh_output_dir, save_file,
                               gmsh_interface)

    mesh_data = gmsh_io.mesh_data

    assert mesh_data["nodes"]["coordinates"].size > 0  # check if node_coords is not empty
    assert mesh_data["nodes"]["ids"].size > 0  # check if node_tags is not empty
    assert list(mesh_data["elements"].keys()) == ["LINE_2N", "TRIANGLE_3N",
                                                  "POINT_1N"]  # check if correct elements are present

    # check each element type contains ids and nodes
    for value in mesh_data["elements"].values():
        assert value["element_ids"].size > 0
        assert value["connectivities"].size > 0


def test_generate_mesh_3D():
    """
    Checks whether mesh data generated for 3D geometries is not empty.

    """
    
     # define the points of the surface as a list of tuples
    input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    # define the mesh size
    element_size = 0.1
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
    extrusion_length = [0, 0, 1]
    # set a name for mesh output file
    mesh_output_name = "test_3D"
    # set output directory of the mesh
    mesh_output_dir = "."

    gmsh_io = GmshIO()

    gmsh_io.generate_gmsh_mesh(input_points, extrusion_length, element_size, dims, name_label,
                               mesh_output_name, mesh_output_dir, save_file,
                               gmsh_interface)

    mesh_data = gmsh_io.mesh_data

    assert mesh_data["nodes"]["coordinates"].size > 0  # check if node_coords is not empty
    assert mesh_data["nodes"]["ids"].size > 0  # check if node_tags is not empty
    assert list(mesh_data["elements"].keys()) == ["LINE_2N", "TRIANGLE_3N", "TETRAHEDRON_4N",
                                                  "POINT_1N"]  # check if correct elements are present

    # check each element type contains ids and nodes
    for value in mesh_data["elements"].values():
        assert value["element_ids"].size > 0
        assert value["connectivities"].size > 0


