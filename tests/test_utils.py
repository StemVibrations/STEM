import re

import numpy as np
import numpy.testing as npt
import pytest

from stem.geometry import Geometry, Point, Line
from stem.utils import Utils
from stem.mesh import Mesh, Element, Node
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

    def test_create_sigmoid_tiny_expr(self):
        """
        Test the creation of the sigmoid function tiny expr, which can be evaluated in c++

        """

        # define input values
        start_time = 5
        dt_slope = 3
        final_value = -10000
        initial_value = -3000

        # define time vector, note that it is required for the test that this variable is called "t"
        t = np.linspace(start_time, start_time + dt_slope, 10)

        # get full sigmoid function
        full_function = Utils.create_sigmoid_tiny_expr(start_time, dt_slope, initial_value, final_value, False)

        # get half sigmoid function
        half_function = Utils.create_sigmoid_tiny_expr(start_time, dt_slope, initial_value, final_value, True)

        # replace e^ with np.exp, such that python can evaluate the string
        python_full_func_str = full_function.replace(r"e^", "np.exp")
        python_half_func_str = half_function.replace(r"e^", "np.exp")

        # evaluate string
        full_eval = eval(python_full_func_str)
        half_eval = eval(python_half_func_str)

        # define expected functions
        expected_half_eval = ((1 / (1 + np.exp(-6/dt_slope * (t - start_time))) - 0.5)
                              * (final_value - initial_value) * 2 + initial_value)
        expected_full_eval = ((1 / (1 + np.exp(-12/dt_slope
                                              * (t - dt_slope/2 - start_time))))
                              * (final_value - initial_value ) + initial_value)

        # check if expected and actual results are almost equal
        np.testing.assert_almost_equal(full_eval, expected_full_eval)
        np.testing.assert_almost_equal(half_eval, expected_half_eval)

    def test_check_lines_geometry_are_path(self):
        """
        Tests that the lines in a geometry are connected and aligned along one path (no branching)

        """

        # None object passed (geometry not initialised)
        msg = "No geometry has been provided."
        with pytest.raises(ValueError, match=msg):
            Utils.check_lines_geometry_are_path(None)

        geo1 = Geometry()

        # test undefined lines in geometry (empty)
        msg = "The geometry doesn't contain lines to check."
        with pytest.raises(ValueError, match=msg):
            Utils.check_lines_geometry_are_path(geo1)

        # test normal path
        geo1.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([0.5, 0, 0], 2),
            3: Point.create([1, 0, 0], 3),
            4: Point.create([1.5, 1, 0], 4)
        }

        geo1.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([2, 3], 2),
            3: Line.create([3, 4], 3),
        }

        # assert geometry is aligned on a path
        assert Utils.check_lines_geometry_are_path(geo1)
        # test for discontinuities
        geo2 = Geometry()

        geo2.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([0.5, 0, 0], 2),
            3: Point.create([1, 0, 0], 3),
            4: Point.create([1.5, 1, 0], 4)
        }

        geo2.lines = {
            1: Line.create([1, 2], 1),
            3: Line.create([3, 4], 3),
        }

        # assert path are disconnected  (not a path)
        assert not Utils.check_lines_geometry_are_path(geo2)

        # test for loops
        geo3 = Geometry()

        geo3.points = {
            1: Point.create([0, 0.5, 0], 1),
            2: Point.create([0.5, 0.5, 0], 2),
            3: Point.create([1, 0.5, 0], 3),
            4: Point.create([0.5, 1, 0], 4),
            5: Point.create([0.5, 0, 0], 5)
        }

        geo3.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([2, 3], 2),
            3: Line.create([3, 4], 3),
            4: Line.create([4, 2], 4),
            5: Line.create([2, 5], 5)
        }

        # assert loop is present (not a path)
        assert not Utils.check_lines_geometry_are_path(geo3)

        # test for loops
        geo4 = Geometry()

        geo4.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([0, 1, 0], 2),
            3: Point.create([1, 0, 0], 3),
            4: Point.create([-1, 0, 0], 4)
        }

        geo4.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([1, 3], 2),
            3: Line.create([1, 4], 3)
        }

        # assert branching point is present (not a path)
        assert not Utils.check_lines_geometry_are_path(geo4)

    def test_is_point_aligned_and_between_any_of_points(self):
        """
        Checks that any of the points is aligned with at least one of the points in a list of pairs of
        coordinates.
        """
        point_coordinates = [[(0.0, 0, 0), (1, 0, 0)],
                             [(1, 0, 0), (2, 0, 0)],
                             [(2, 0, 0), (4, 0, 0)]]

        origin_correct = (0.5, 0, 0)
        origin_wrong = (3, 1, 0)

        # check that the function returns true for correct definition of points/origin
        assert Utils.is_point_aligned_and_between_any_of_points(point_coordinates, origin_correct)

        # check that the function returns false for incorrect definition of points/origin
        assert not Utils.is_point_aligned_and_between_any_of_points(point_coordinates, origin_wrong)

    def test_replace_extension(self):
        """
        Tests that the extension of a filename is replaced correctly.

        """
        filename1 = "outputfile"
        filename2 = "outputfile.csv"
        filename3 = "outputfile.tmp.csv"

        desired_filename = "outputfile.json"

        assert Utils.replace_extensions(filename1, ".json") == desired_filename
        assert Utils.replace_extensions(filename2, ".json") == desired_filename
        assert Utils.replace_extensions(filename3, ".json") == desired_filename

    def test_find_nodes_close_to_geometry_points(self):
        """
        Tests that nodes close to given geometry points are correctly identified.

        """

        # define geometry
        geometry = Geometry()
        geometry.points = {
            1: Point.create([1, 0, 0], 1),
            2: Point.create([3, 0, 0], 2),
            3: Point.create([5, 0, 0], 3)
        }

        # define mesh
        mesh = Mesh(ndim=2)
        mesh.nodes = {
            1: Node(id=1, coordinates=[1, 0, 0]),
            2: Node(id=2, coordinates=[2, 0, 0]),
            3: Node(id=3, coordinates=[3, 0, 0]),
            4: Node(id=4, coordinates=[4, 0, 0]),
            5: Node(id=5, coordinates=[5, 0, 0])
        }
        expected_ids = [1, 3, 5]
        actual_ids = Utils.find_node_ids_close_to_geometry_nodes(mesh=mesh, geometry=geometry)

        np.testing.assert_equal(actual=actual_ids, desired=expected_ids)
