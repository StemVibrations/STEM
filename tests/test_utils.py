import re

import numpy as np
import numpy.testing as npt
import pytest

from stem.utils import Utils
from stem.mesh import Mesh, Element
from stem.globals import ELEMENT_DATA

from tests.utils import TestUtils


class TestUtilsStem:

    def test_check_ndim_nnodes_combinations(self):
        """
        Test the check which checks if the combination of n_dim and n_nodes is valid for a certain element type.
        It checks for valid input, invalid n_dim and invalid n_nodes.

        """

        # successful checks
        Utils.check_ndim_nnodes_combinations(2,2, {2: [2]}, "test")

        # non valid n_dim
        with pytest.raises(ValueError, match=re.escape(f"Number of dimensions 3 is not supported for failed_test "
                                                       f"elements. Supported dimensions are [2].")):
            Utils.check_ndim_nnodes_combinations(3,2, {2: [2]},
                                                 "failed_test")

        # non valid n_nodes for dim
        with pytest.raises(ValueError, match=re.escape(f"In 2 dimensions, only [3, 4, 5] noded failed_test_2 elements "
                                                       f"are supported. 2 nodes were provided.")):
            Utils.check_ndim_nnodes_combinations(2,2, {2: [3,4,5]},
                                                 "failed_test_2")


    def test_is_clockwise(self):
        """
        Test the check which checks if coordinates are given in clockwise order
        """

        coordinates = [[0, 0], [2, 0], [2, 2], [0, 2]]

        assert not Utils.are_2d_coordinates_clockwise(coordinates=coordinates)
        assert Utils.are_2d_coordinates_clockwise(coordinates=coordinates[::-1])

    def test_is_clockwise_non_convex(self):
        """
        Test the check which checks if coordinates are given in clockwise order for a non-convex polygon

        """

        coordinates = [[0, 0], [2, 0], [2, 2], [1, -1], [0, 2]]

        assert not Utils.are_2d_coordinates_clockwise(coordinates=coordinates)
        assert Utils.are_2d_coordinates_clockwise(coordinates=coordinates[::-1])

    def test_collinearity_2d(self):
        """
        Check collinearity between 3 points in 2D
        """
        p1 = np.array([0, 0])
        p2 = np.array([-2, -1])

        p_test_1 = np.array([2, 1])
        p_test_2 = np.array([-5, 1])

        assert Utils.is_collinear(point=p_test_1, start_point=p1, end_point=p2)
        assert not Utils.is_collinear(point=p_test_2, start_point=p1, end_point=p2)

    def test_collinearity_3d(self):
        """
        Check collinearity between 3 points in 3D
        """

        p1 = np.array([0, 0, 0])
        p2 = np.array([-2, -2, 2])

        p_test_1 = np.array([2, 2, -2])
        p_test_2 = np.array([2, -2, 2])

        assert Utils.is_collinear(point=p_test_1, start_point=p1, end_point=p2)
        assert not Utils.is_collinear(point=p_test_2, start_point=p1, end_point=p2)

    def test_is_in_between_2d(self):
        """
        Check if point is in between other 2 points in 2D
        """

        p1 = np.array([0, 0])
        p2 = np.array([-2, -2])

        p_test_1 = np.array([2, 2])
        p_test_2 = np.array([-1, -1])

        assert not Utils.is_point_between_points(
            point=p_test_1, start_point=p1, end_point=p2
        )
        assert Utils.is_point_between_points(
            point=p_test_2, start_point=p1, end_point=p2
        )

    def test_is_in_between_3d(self):
        """
        Check if point is in between other 2 points in 3D
        """

        p1 = np.array([0, 0, 0])
        p2 = np.array([-2, -2, 2])

        p_test_1 = np.array([2, 2, -2])
        p_test_2 = np.array([-1, -1, 1])

        assert not Utils.is_point_between_points(
            point=p_test_1, start_point=p1, end_point=p2
        )
        assert Utils.is_point_between_points(
            point=p_test_2, start_point=p1, end_point=p2
        )

    def test_check_sequence_non_string(self):
        """
        Test assertion of non-string sequence
        """

        assert Utils.is_non_str_sequence([1, 2, 3, 4])
        assert Utils.is_non_str_sequence((1, 2, 3, 4))

        assert not Utils.is_non_str_sequence({"Hello world"})
        assert not Utils.is_non_str_sequence({1: {"a": "A"}, 2: {"b": "B"}})

    def test_chain_sequence(self):
        """
        Test chain of sequences
        """

        # test 1: no conflicts in arguments, merge dictionaries with common keys
        # new keys are added, common keys are expanded
        seq1 = [1, 2, 3, 4]
        seq2 = (100, 200, 300)

        actual_sequence = list(Utils.chain_sequence([seq1, seq2]))
        expected_sequence = [1, 2, 3, 4, 100, 200, 300]

        npt.assert_equal(expected_sequence, actual_sequence)

    def test_merge(self):
        """
        Test merging of dictionaries
        """

        # test 1: no conflicts in arguments, merge dictionaries with common keys
        # new keys are added, common keys are expanded
        dict1a = {1: {"a": "A"}, 2: {"b": "B"}}
        dict1b = {2: {"c": "C"}, 3: {"d": "D"}}

        actual_dict_1 = Utils.merge(dict1a, dict1b)
        expected_dict_1 = {1: {"a": "A"}, 2: {"c": "C", "b": "B"}, 3: {"d": "D"}}

        # expect it raises an error
        TestUtils.assert_dictionary_almost_equal(expected_dict_1, actual_dict_1)

        # test 2: conflicts in arguments, merge common keys argument into a list
        # common keys are expanded into list
        dict2a = {1: {"a": [1, 2, 3]}, 2: {"b": "B"}}
        dict2b = {1: {"a": [5, 6]}, 2: {"b": "D"}, 3: {"d": "D"}}

        # expect it raises an error
        with pytest.raises(
                ValueError, match="Conflict of merging keys at 2->b. Two non sequence values have been found."
        ):
            Utils.merge(dict2a, dict2b)

        # test 3: conflicts in arguments, merge common keys argument into a list
        # list of lists is preserved into a list of list + 2 extra elements
        # tuple(int) and list(str) are merged into list(str,int)
        dict3a = {1: {"a": [[1, 2, 3]]}, 2: {"b": ["B"]}}
        dict3b = {1: {"a": [5, 6]}, 2: {"b": (1, 2, 3)}, 3: {"d": "D"}}

        actual_dict_3 = Utils.merge(dict3a, dict3b)
        expected_dict_3 = {
            1: {"a": [[1, 2, 3], 5, 6]},
            2: {"b": ["B", 1, 2, 3]},
            3: {"d": "D"},
        }

        TestUtils.assert_dictionary_almost_equal(expected_dict_3, actual_dict_3)

    def test_flip_node_order(self):
        """
        Tests that element node ids are flipped in the right way.
        """

        # define original mesh data

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

        mesh = Mesh(ndim=2)

        for element_name, element in mesh_data["elements"].items():
            for element_id, node_ids in element.items():
                # create element and add to mesh
                mesh.elements[element_id] = Element(element_id, element_name, node_ids)

                # reverse the node order
                Utils.flip_node_order([mesh.elements[element_id]])

        # set expected node ordering per element
        expected_ordering = [
            [2, 1, 4, 3],
            [2, 1, 4, 3, 5, 8, 7, 6],
            [3, 2, 1],
            [3, 2, 1, 9, 6, 5],
            [2, 1],
            [2, 1, 5],
        ]

        for element, expected_nodes_element in zip(list(mesh.elements.values()), expected_ordering):
            np.testing.assert_equal(element.node_ids, expected_nodes_element)

    def test_flip_node_order_exception(self):
        """
        Tests that an exception is raised if a list with different elements is given to flip_node_order.
        """

        with pytest.raises(ValueError, match="All elements should be of the same type."):
            Utils.flip_node_order([Element(1, "LINE_2N", [1, 2]),
                                   Element(2, "LINE_3N", [2, 3, 4])])

    def test_is_tetrahedron_4n_edge_defined_outwards(self):
        """
        Tests if the 3-node triangle edge of a 4 node tetrahedron is defined outwards. It checks different orientations
        of the edge element and the body element.

        """

        edge_element = Element(1, "TRIANGLE_3N", [1, 2, 3])
        edge_element_reversed = Element(1, "TRIANGLE_3N", [3, 2, 1])
        body_element = Element(2, "TETRAHEDRON_4N", [1, 2, 3, 4])
        body_element_mirrored = Element(2, "TETRAHEDRON_4N", [1, 2, 3, 5])

        nodes = {1: [0, 0, 0], 2: [1.0, 0, 0], 3: [1, 1.0, 0], 4: [0, 0.0, 1.0], 5: [0.0, 0.0, -1.0]}

        # check if edge is defined outwards in both node orders of edge element
        assert not Utils.is_volume_edge_defined_outwards(edge_element, body_element, nodes)
        assert Utils.is_volume_edge_defined_outwards(edge_element_reversed, body_element, nodes)

        # check if edge is defined outwards in both node orders of edge element and a mirrored body element
        assert Utils.is_volume_edge_defined_outwards(edge_element, body_element_mirrored, nodes)
        assert not Utils.is_volume_edge_defined_outwards(edge_element_reversed, body_element_mirrored, nodes)

    def test_is_tetrahedron_10n_edge_defined_outwards(self):
        """
        Tests if the 6-node triangle edge of a 10 node tetrahedron is defined outwards. It checks different orientations
        of the edge element and the body element.

        """

        edge_element = Element(1, "TRIANGLE_6N", [1, 2, 3, 5, 6, 7])
        edge_element_reversed = Element(1, "TRIANGLE_6N", [3, 2, 1, 7, 6, 5])

        body_element = Element(2, "TETRAHEDRON_10N", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        body_element_mirrored = Element(2, "TETRAHEDRON_10N", [1, 2, 3, 11, 5, 6, 7, 12, 9, 13])

        nodes = {1: [0, 0, 0], 2: [1.0, 0, 0], 3: [0.0, 1.0, 0], 4: [0, 0.0, 1.0],# vertices
                 5: [0.5, 0.0, 0.0], 6: [0.5, 0.5, 0.0], 7: [0.0, 5.0, 0.0], # x-y plane midpoints
                 8: [0.0, 0.0, 0.5], 9: [0.0, 0.5, 0.5],  # new x-z plane midpoints
                 10: [0.5, 0.0, 0.5], # new y-z plane midpoint
                 11: [0.0, 0.0, -1.0], 12: [0.0, 0.0, -0.5], 13: [0.5, 0.0, -0.5]} # mirrored y-z plane points

        # check if edge is defined outwards in both node orders of edge element
        assert not Utils.is_volume_edge_defined_outwards(edge_element, body_element, nodes)
        assert Utils.is_volume_edge_defined_outwards(edge_element_reversed, body_element, nodes)

        # check if edge is defined outwards in both node orders of edge element and a mirrored body element
        assert Utils.is_volume_edge_defined_outwards(edge_element, body_element_mirrored, nodes)
        assert not Utils.is_volume_edge_defined_outwards(edge_element_reversed, body_element_mirrored, nodes)

    def test_is_hexahedron_8n_edge_defined_outwards(self):
        """
        Tests if the 4-node quad edge of a 8 node hexahedron is defined outwards. It checks different orientations
        of the edge element and the body element.

        """

        edge_element = Element(1, "QUADRANGLE_4N", [1, 2, 3, 4])
        edge_element_reversed = Element(1, "QUADRANGLE_4N", [4, 3, 2, 1])

        body_element = Element(2, "HEXAHEDRON_8N", [1, 2, 3, 4, 5, 6, 7, 8])
        body_element_mirrored = Element(2, "HEXAHEDRON_8N", [1, 2, 3, 4, 9, 10, 11, 12])

        nodes = {1: [0, 0, 0], 2: [1.0, 0, 0], 3: [1, 1.0, 0], 4: [0, 1.0, 0],
                 5: [0, 0, 1.0], 6: [1, 0, 1.0], 7: [1, 1, 1], 8: [0, 1, 1],
                 9: [0.0, 0.0, -1], 10: [1.0, 0.0, -1], 11: [1, 1, -1], 12: [0.0, 1, -1]}

        # check if edge is defined outwards in both node orders of edge element
        assert not Utils.is_volume_edge_defined_outwards(edge_element, body_element, nodes)
        assert Utils.is_volume_edge_defined_outwards(edge_element_reversed, body_element, nodes)

        # check if edge is defined outwards in both node orders of edge element and a mirrored body element
        assert Utils.is_volume_edge_defined_outwards(edge_element, body_element_mirrored, nodes)
        assert not Utils.is_volume_edge_defined_outwards(edge_element_reversed, body_element_mirrored, nodes)

    @pytest.mark.skip("Hexahedron 20n is not correctly implemented yet.")
    def test_is_hexahedron_20n_edge_defined_outwards(self):
        """
        Tests if the 8-node quad edge of a 20 node hexahedron is defined outwards. It checks different orientations
        of the edge element and the body element.

        """

        edge_element_1 = Element(1, "QUADRANGLE_8N", [1, 2, 3, 4, 9, 12, 14, 10])
        edge_element_1_reversed = Element(1, "QUADRANGLE_8N", [4, 3, 2, 1, 10, 14, 12, 9])

        body_element = Element(2, "HEXAHEDRON_20N", [1, 2, 3, 4,
                                                     5, 6, 7, 8,
                                                     9, 10, 11, 12,
                                                     13, 14, 15, 16,
                                                     17, 18, 19, 20])

        body_element_mirrored = Element(2, "HEXAHEDRON_20N", [1, 2, 3, 4,
                                                              21, 22, 23, 24,
                                                              9, 10, 25, 12,
                                                              26, 14, 27, 28,
                                                              29, 32, 30, 31])

        nodes = {1: [0, 0, 0], 2: [1.0, 0, 0], 3: [1, 1.0, 0], 4: [0, 1.0, 0],
                 5: [0, 0, 1.0], 6: [1, 0, 1.0], 7: [1, 1, 1], 8: [0, 1, 1],
                 9: [0.5, 0.0, 0], 12: [1.0, 0.5, 0.0], 14: [0.5, 1, 0], 10: [0, 0.5, 0.0],
                 11: [0, 0, 0.5], 13: [1, 0, 0.5], 15: [1, 1, 0.5], 16: [0, 1, 0.5],
                 17: [0.5, 0.0, 1], 19: [1, 0.5, 1], 20: [0.5, 1, 1], 18: [0, 0.5, 1],
                 21: [0, 0, -1.0], 22: [1, 0, -1.0], 23: [1, 1, -1], 24: [0, 1, -1], # mirrored vertices
                 25: [0, 0, -0.5], 26: [1, 0, -0.5], 27: [1, 1, -0.5], 28: [0, 1, -0.5], # mirrored midpoints
                 29: [0.5, 0.0, -1], 30: [1, 0.5, -1], 31: [0.5, 1, -1], 32: [0, 0.5, -1]} # mirrored midpoints

        # check if edge is defined outwards in both node orders of edge element
        assert not Utils.is_volume_edge_defined_outwards(edge_element_1, body_element, nodes)
        assert Utils.is_volume_edge_defined_outwards(edge_element_1_reversed, body_element, nodes)

        # check if edge is defined outwards in both node orders of edge element and a mirrored body element
        assert Utils.is_volume_edge_defined_outwards(edge_element_1, body_element_mirrored, nodes)
        assert not Utils.is_volume_edge_defined_outwards(edge_element_1_reversed, body_element_mirrored, nodes)

    def test_is_volume_edge_defined_outwards_exceptions(self):
        """
        Tests exceptions of is_volume_edge_defined_outwards

        """

        edge_element_1 = Element(1, "LINE_2N", [1, 2])
        edge_element_2 = Element(2, "TRIANGLE_3N", [1, 2, 3])
        edge_element_3 = Element(3, "TRIANGLE_3N", [1, 2, 5])


        body_element_1 = Element(3, "TETRAHEDRON_4N", [1, 2, 3, 4])
        body_element_2 = Element(4, "TRIANGLE_3N", [1, 2, 3])

        nodes = {1: [0, 0, 0], 2: [1.0, 0, 0], 3: [1, 1.0, 0], 4: [0, 0.0, 1.0], 5: [0.0, 0.0, -1.0]}

        # expected raise as edge element is not a 2D element
        with pytest.raises(ValueError, match="Edge element should be a 2D element."):
            Utils.is_volume_edge_defined_outwards(edge_element_1,body_element_1, nodes)

        # expected raise as body element is not 3D
        with pytest.raises(ValueError, match="Body element should be a 3D element."):
            Utils.is_volume_edge_defined_outwards(edge_element_2, body_element_2, nodes)

        # expected raise as not all nodes of edge element are in body element
        with pytest.raises(ValueError, match="All nodes of the edge element should be part of the body element."):
            Utils.is_volume_edge_defined_outwards(edge_element_3, body_element_1, nodes)


