from copy import deepcopy

import pytest
import numpy as np
import numpy.testing as npt

from stem.mesh import *


class TestMesh:

    def test_create_0d_mesh_from_gmsh_group(self):
        """
        Test the creation of a 0D mesh from a gmsh group.

        """

        # Set up the mesh data
        mesh_data = {
            "ndim": 0,
            "nodes": {
                1: [0, 0, 0],
                2: [0.5, 0, 0]
            },
            "elements": {
                "POINT_1N": {
                    1: [1],
                    2: [2]
                }
            },
            "physical_groups": {
                "points_group": {
                    "ndim": 0,
                    "element_ids": [1, 2],
                    "node_ids": [1, 2],
                    "element_type": "POINT_1N"
                }
            },
        }

        # Create the mesh from the gmsh group
        generated_mesh = Mesh.create_mesh_from_gmsh_group(mesh_data, "points_group")

        # set expected mesh
        expected_nodes = [Node(1, [0, 0, 0]), Node(2, [0.5, 0, 0])]
        expected_elements = [Element(1, "POINT_1N", [1]), Element(2, "POINT_1N", [2])]
        expected_mesh = Mesh(0)
        expected_mesh.nodes = expected_nodes
        expected_mesh.elements = expected_elements

        # Check the generated mesh
        assert generated_mesh.ndim == expected_mesh.ndim

        # Check the nodes
        for generated_node, expected_node in zip(generated_mesh.nodes.values(), expected_mesh.nodes):
            assert generated_node.id == expected_node.id
            assert pytest.approx(generated_node.coordinates) == expected_node.coordinates

        # Check the elements
        for generated_element, expected_element in zip(generated_mesh.elements.values(), expected_mesh.elements):
            assert generated_element.id == expected_element.id
            assert generated_element.element_type == expected_element.element_type
            assert generated_element.node_ids == expected_element.node_ids

    def test_create_1d_mesh_from_gmsh_group(self):
        """
        Test the creation of a 1D mesh from a gmsh group.

        """

        # Set up the mesh data
        mesh_data = {
            "ndim": 1,
            "nodes": {
                1: [0, 0, 0],
                2: [0.5, 0, 0],
                3: [1, 0, 0]
            },
            "elements": {
                "LINE_2N": {
                    1: [1, 2],
                    2: [2, 3]
                }
            },
            "physical_groups": {
                "lines_group": {
                    "ndim": 1,
                    "element_ids": [1, 2],
                    "node_ids": [1, 2, 3],
                    "element_type": "LINE_2N"
                }
            },
        }

        # Create the mesh from the gmsh group
        generated_mesh = Mesh.create_mesh_from_gmsh_group(mesh_data, "lines_group")

        # set expected mesh
        expected_nodes = [Node(1, [0, 0, 0]), Node(2, [0.5, 0, 0]), Node(3, [1, 0, 0])]
        expected_elements = [Element(1, "LINE_2N", [1, 2]), Element(2, "LINE_2N", [2, 3])]
        expected_mesh = Mesh(1)
        expected_mesh.nodes = expected_nodes
        expected_mesh.elements = expected_elements

        # Check the generated mesh
        assert generated_mesh.ndim == expected_mesh.ndim

        # Check the nodes
        for generated_node, expected_node in zip(generated_mesh.nodes.values(), expected_mesh.nodes):
            assert generated_node.id == expected_node.id
            assert pytest.approx(generated_node.coordinates) == expected_node.coordinates

        # Check the elements
        for generated_element, expected_element in zip(generated_mesh.elements.values(), expected_mesh.elements):
            assert generated_element.id == expected_element.id
            assert generated_element.element_type == expected_element.element_type
            assert generated_element.node_ids == expected_element.node_ids

    def test_create_2d_mesh_from_gmsh_group(self):
        """
        Test the creation of a 2D mesh from a gmsh group.

        """

        # Set up the mesh data
        mesh_data = {
            "ndim": 2,
            "nodes": {
                1: [0, 0, 0],
                2: [0.5, 0, 0],
                3: [1, 0, 0],
                4: [0, 0.5, 0],
                5: [0.5, 0.5, 0],
                6: [1, 0.5, 0]
            },
            "elements": {
                "TRIANGLE_3N": {
                    1: [1, 2, 4],
                    2: [2, 3, 5],
                    3: [3, 6, 5]
                }
            },
            "physical_groups": {
                "triangles_group": {
                    "ndim": 2,
                    "element_ids": [1, 2, 3],
                    "node_ids": [1, 2, 3, 4, 5, 6],
                    "element_type": "TRIANGLE_3N",
                }
            },
        }

        # Create the mesh from the gmsh group
        generated_mesh = Mesh.create_mesh_from_gmsh_group(mesh_data, "triangles_group")

        # set expected mesh
        expected_nodes = [
            Node(1, [0, 0, 0]),
            Node(2, [0.5, 0, 0]),
            Node(3, [1, 0, 0]),
            Node(4, [0, 0.5, 0]),
            Node(5, [0.5, 0.5, 0]),
            Node(6, [1, 0.5, 0]),
        ]
        expected_elements = [
            Element(1, "TRIANGLE_3N", [1, 2, 4]),
            Element(2, "TRIANGLE_3N", [2, 3, 5]),
            Element(3, "TRIANGLE_3N", [3, 6, 5]),
        ]

        expected_mesh = Mesh(2)
        expected_mesh.nodes = expected_nodes
        expected_mesh.elements = expected_elements

        # Check the generated mesh
        assert generated_mesh.ndim == expected_mesh.ndim

        # Check the nodes
        for generated_node, expected_node in zip(generated_mesh.nodes.values(), expected_mesh.nodes):
            assert generated_node.id == expected_node.id
            assert pytest.approx(generated_node.coordinates) == expected_node.coordinates

        # Check the elements
        for generated_element, expected_element in zip(generated_mesh.elements.values(), expected_mesh.elements):
            assert generated_element.id == expected_element.id
            assert generated_element.element_type == expected_element.element_type
            assert generated_element.node_ids == expected_element.node_ids

    def test_create_3d_mesh_from_gmsh_group(self):
        """
        Test the creation of a 3D mesh from a gmsh group.

        """

        # Set up the mesh data
        mesh_data = {
            "ndim": 3,
            "nodes": {
                1: [0, 0, 0],
                2: [0.5, 0, 0],
                3: [1, 0, 0],
                4: [0, 0.5, 0],
                5: [0.5, 0.5, 0],
                6: [1, 0.5, 0],
                7: [0, 0, 0.5],
                8: [0.5, 0, 0.5],
                9: [1, 0, 0.5],
                10: [0, 0.5, 0.5],
                11: [0.5, 0.5, 0.5],
                12: [1, 0.5, 0.5],
            },
            "elements": {
                "TETRAHEDRON_4N": {
                    1: [1, 2, 4, 7],
                    2: [2, 3, 5, 8],
                    3: [3, 6, 5, 9],
                    4: [4, 5, 7, 10],
                    5: [5, 6, 8, 11],
                    6: [6, 9, 11, 8],
                }
            },
            "physical_groups": {
                "tetrahedral_group": {
                    "ndim": 3,
                    "element_ids": [1, 2, 3, 4, 5, 6],
                    "node_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    "element_type": "TETRAHEDRON_4N",
                }
            },
        }

        # Create the mesh from the gmsh group
        generated_mesh = Mesh.create_mesh_from_gmsh_group(mesh_data, "tetrahedral_group")

        # set expected mesh
        expected_nodes = [
            Node(1, [0, 0, 0]),
            Node(2, [0.5, 0, 0]),
            Node(3, [1, 0, 0]),
            Node(4, [0, 0.5, 0]),
            Node(5, [0.5, 0.5, 0]),
            Node(6, [1, 0.5, 0]),
            Node(7, [0, 0, 0.5]),
            Node(8, [0.5, 0, 0.5]),
            Node(9, [1, 0, 0.5]),
            Node(10, [0, 0.5, 0.5]),
            Node(11, [0.5, 0.5, 0.5]),
            Node(12, [1, 0.5, 0.5]),
        ]

        expected_elements = [
            Element(1, "TETRAHEDRON_4N", [1, 2, 4, 7]),
            Element(2, "TETRAHEDRON_4N", [2, 3, 5, 8]),
            Element(3, "TETRAHEDRON_4N", [3, 6, 5, 9]),
            Element(4, "TETRAHEDRON_4N", [4, 5, 7, 10]),
            Element(5, "TETRAHEDRON_4N", [5, 6, 8, 11]),
            Element(6, "TETRAHEDRON_4N", [6, 9, 11, 8]),
        ]

        expected_mesh = Mesh(3)
        expected_mesh.nodes = expected_nodes
        expected_mesh.elements = expected_elements

        # Check the generated mesh
        assert generated_mesh.ndim == expected_mesh.ndim

        # Check the nodes
        for generated_node, expected_node in zip(generated_mesh.nodes.values(), expected_mesh.nodes):
            assert generated_node.id == expected_node.id
            assert pytest.approx(generated_node.coordinates) == expected_node.coordinates

        # Check the elements
        for generated_element, expected_element in zip(generated_mesh.elements.values(), expected_mesh.elements):
            assert generated_element.id == expected_element.id
            assert generated_element.element_type == expected_element.element_type
            assert generated_element.node_ids == expected_element.node_ids

    def test_create_mesh_from_non_existing_group(self):
        """
        Test the creation of a mesh from a non-existing gmsh group. Expected to raise a ValueError.

        """

        # Set up the mesh data
        mesh_data = {
            "ndim": 0,
            "nodes": {
                1: [0, 0, 0],
                2: [0.5, 0, 0]
            },
            "elements": {
                "POINT_1N": {
                    1: [1],
                    2: [2]
                }
            },
            "physical_groups": {
                "points_group": {
                    "ndim": 0,
                    "element_ids": [1, 2],
                    "node_ids": [1, 2],
                    "element_type": "POINT_1N"
                }
            },
        }

        # Create the mesh from the gmsh group
        with pytest.raises(ValueError):
            Mesh.create_mesh_from_gmsh_group(mesh_data, "non_existing_group")

    def test_is_node_equal(self):
        """
        Test the equality of two nodes.

        """

        # Create two equal nodes
        node1 = Node(1, [0, 0])
        node2 = Node(1, [0, 0])

        # Check the equality of the two nodes
        assert node1 == node2

        # change node id of node2
        node3 = deepcopy(node2)
        node3.id = 2
        assert node1 != node3

        # change coordinates of node2
        node3 = deepcopy(node2)
        node3.coordinates = [0, 1]
        assert node1 != node3

        # create a non node object
        non_node_object = "I am not a node"

        # Check the equality of the node with the non-node object
        assert node1 != non_node_object

    def test_is_element_equal(self):
        """
        Test the equality of two elements.

        """

        # Create two equal elements
        element1 = Element(1, "TRIANGLE_3N", [1, 2, 3])
        element2 = Element(1, "TRIANGLE_3N", [1, 2, 3])

        # Check the equality of the two elements
        assert element1 == element2

        # change element type of element2
        element3 = deepcopy(element2)
        element3.element_type = "TRIANGLE_6N"
        assert element1 != element3

        # change node ids of element2
        element3 = deepcopy(element2)
        element3.node_ids = [1, 2, 4]
        assert element1 != element3

        # create a non element object
        non_element_object = "I am not an element"

        # Check the equality of the element with the non-element object
        assert element1 != non_element_object

    def test_is_mesh_equal(self):
        """
        Test the equality of two meshes.

        """

        # Create two equal meshes
        mesh1 = Mesh(2)
        mesh1.nodes = {1: Node(1, [0, 0]), 2: Node(2, [1, 1])}
        mesh1.elements = {1: Element(1, "TRIANGLE_3N", [1, 2, 3])}

        mesh2 = Mesh(2)
        mesh2.nodes = {1: Node(1, [0, 0]), 2: Node(2, [1, 1])}
        mesh2.elements = {1: Element(1, "TRIANGLE_3N", [1, 2, 3])}

        # Check the equality of the two meshes
        assert mesh1 == mesh2

        # change dimension of mesh2
        mesh3 = deepcopy(mesh2)
        mesh3.ndim = 3
        assert mesh1 != mesh3

        # change nodes of mesh2
        mesh3 = deepcopy(mesh2)
        mesh3.nodes = {1: Node(1, [0, 0]), 2: Node(2, [1, 2])}
        assert mesh1 != mesh3

        # change elements of mesh2
        mesh3 = deepcopy(mesh2)
        mesh3.elements = {1: Element(1, "TRIANGLE_3N", [1, 2, 4])}
        assert mesh1 != mesh3

        # create a non mesh object
        non_mesh_object = "I am not a mesh"

        # Check the equality of the mesh with the non-mesh object
        assert mesh1 != non_mesh_object

    def test_calculate_centroids_multiple_elements(self):
        """
        Test the calculation of the centroids of multiple elements.

        """

        # Create a 2D mesh
        mesh = Mesh(2)
        mesh.nodes = {
            1: Node(1, [0, 0, 0]),
            2: Node(2, [1, 0, 0]),
            3: Node(3, [0, 1, 0]),
            4: Node(4, [1, 1, 0]),
            5: Node(5, [0.5, 0.5, 0])
        }
        mesh.elements = {
            1: Element(1, "TRIANGLE_3N", [1, 5, 2]),
            2: Element(2, "TRIANGLE_3N", [1, 3, 5]),
            3: Element(3, "TRIANGLE_3N", [3, 5, 4]),
            4: Element(4, "TRIANGLE_3N", [2, 5, 4])
        }

        # Calculate the centroids
        calculated_centroids = mesh.calculate_centroids()

        expected_centroids = np.array([[0.5, 0.16666667, 0.], [0.16666667, 0.5, 0.], [0.5, 0.83333333, 0.],
                                       [0.83333333, 0.5, 0.]])

        npt.assert_array_almost_equal(calculated_centroids, expected_centroids)

    def test_calculate_centroids_single_point_element(self):
        """
        Test the calculation of the centroids of a single point element.

        """

        # Create a 2D mesh
        mesh = Mesh(2)
        mesh.nodes = {1: Node(1, [1, 0, 1])}
        mesh.elements = {1: Element(1, "POINT_1N", [1])}

        # Calculate the centroids
        calculated_centroids = mesh.calculate_centroids()

        expected_centroids = np.array([[1, 0, 1]])

        npt.assert_array_almost_equal(calculated_centroids, expected_centroids)


class TestMeshSettings:
    """
    Test the mesh settings class.
    """

    def test_validation_element_order_at_initialisation_expected_raise(self):
        """
        Test the validation of the element order. Expected to raise a ValueError when the element order is not 1 or 2.

        """

        # test if ValueError is raised when element_order is not 1 or 2
        with pytest.raises(ValueError):
            MeshSettings(element_order=3)

    def test_validation_element_order_after_initialisation_expected_raise(self):
        """
        Test the validation of the element order. Expected to raise a ValueError when the element order is not 1 or 2.

        """

        # test if ValueError is raised when element_order is not 1 or 2
        mesh_settings = MeshSettings()

        with pytest.raises(ValueError):
            mesh_settings.element_order = 3
