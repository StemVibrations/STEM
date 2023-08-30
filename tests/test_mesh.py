import pytest

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
            "nodes": {1: [0, 0, 0], 2: [0.5, 0, 0]},
            "elements": {"POINT_1N": {1: [1], 2: [2]}},
            "physical_groups": {
                "points_group": {"ndim": 0, "element_ids": [1, 2], "node_ids": [1, 2], "element_type": "POINT_1N"}
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
            "nodes": {1: [0, 0, 0], 2: [0.5, 0, 0], 3: [1, 0, 0]},
            "elements": {"LINE_2N": {1: [1, 2], 2: [2, 3]}},
            "physical_groups": {
                "lines_group": {"ndim": 1, "element_ids": [1, 2], "node_ids": [1, 2, 3], "element_type": "LINE_2N"}
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
            "nodes": {1: [0, 0, 0], 2: [0.5, 0, 0], 3: [1, 0, 0], 4: [0, 0.5, 0], 5: [0.5, 0.5, 0], 6: [1, 0.5, 0]},
            "elements": {"TRIANGLE_3N": {1: [1, 2, 4], 2: [2, 3, 5], 3: [3, 6, 5]}},
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
            "nodes": {1: [0, 0, 0], 2: [0.5, 0, 0]},
            "elements": {"POINT_1N": {1: [1], 2: [2]}},
            "physical_groups": {
                "points_group": {"ndim": 0, "element_ids": [1, 2], "node_ids": [1, 2], "element_type": "POINT_1N"}
            },
        }

        # Create the mesh from the gmsh group
        with pytest.raises(ValueError):
            Mesh.create_mesh_from_gmsh_group(mesh_data, "non_existing_group")

    def test_flip_node_order(self):
        """
        Tests that element node ids are flipped in the right way. in 2D.
        """

        mesh = Mesh(ndim=2)

        mesh_data = {
            "ndim": 2,
            "nodes": {1: [0, 0, 0], 2: [1.0, 0, 0], 3: [1, 1.0, 0], 4: [0, 1.0, 0],
                      5: [0.5, 0.0, 0], 6: [1, 0.5, 0], 7: [0.5, 1, 0], 8: [0, 0.5, 0],
                      9: [0.5, 0.5, 0]},
            "elements": {"QUADRANGLE_4N": {1: [1, 2, 3, 4]},
                         "QUADRANGLE_8N": {2: [1, 2, 3, 4, 5, 6, 7, 8]},
                         "TRIANGLE_3N": {3: [1, 2, 3]},
                         "TRIANGLE_6N": {4: [1, 2, 3, 5, 6, 9]},
                         "LINE_2N": {5: [1, 2]},
                         "LINE_3N": {6: [1, 2, 5]}},
            "physical_groups": {
                "quad_linear": {
                    "ndim": 2,
                    "element_ids": [1],
                    "node_ids": [1, 2, 3, 4],
                    "element_type": "QUADRANGLE_4N",
                },
                "quad_quadr": {
                    "ndim": 2,
                    "element_ids": [2],
                    "node_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "element_type": "QUADRANGLE_8N",
                },
                "tri_linear": {
                    "ndim": 2,
                    "element_ids": [3],
                    "node_ids": [1, 2, 3],
                    "element_type": "TRIANGLE_3N",
                },
                "tri_quadr": {
                    "ndim": 2,
                    "element_ids": [4],
                    "node_ids": [1, 2, 3, 5, 6, 9],
                    "element_type": "TRIANGLE_6N",
                },
                "line_linear": {
                    "ndim": 2,
                    "element_ids": [5],
                    "node_ids": [1, 2],
                    "element_type": "LINE_2N",
                },
                "line_quadr": {
                    "ndim": 2,
                    "element_ids": [6],
                    "node_ids": [1, 2, 5],
                    "element_type": "LINE_3N"}
            },
        }

        for group_name, group_data in mesh_data["physical_groups"].items():
            group_element_type = group_data["element_type"]
            element_reversed_ordering_info = Mesh.get_2d_element_info(group_element_type)

            mesh.flip_node_order(group_name, mesh_data, element_reversed_ordering_info)

        expected_ordering = [
            [[4, 3, 2, 1]],
            [[4, 3, 2, 1, 8, 7, 6, 5]],
            [[3, 2, 1]],
            [[3, 2, 1, 9, 6, 5]],
            [[2, 1]],
            [[2, 1, 5]],
        ]

        for (element_name, element_data), expected_nodes_element in zip(mesh_data["elements"].items(),
                                                                    expected_ordering):
            np.testing.assert_equal(list(element_data.values()), expected_nodes_element)

        # not_implemented_elements = ["TETRAHEDRON_4N", "HEXAHEDRON_8N", "TETRAHEDRON_10N", "HEXAHEDRON_20N"]
        #
        # # check for raising errors for unsupported elements
        #
        # for not_implemented_element in not_implemented_elements:
        #     with pytest.raises(NotImplementedError):
        #         mesh.flip_node_order(Element(999, not_implemented_element, [1,2,3]))


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
