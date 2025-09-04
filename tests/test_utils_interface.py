import pytest
import plotly.graph_objects as go

from stem.utils_interface import UtilsInterface
from stem.mesh import Node


class TestUtilsInterface:

    def test_vertical_interface_2d(self):
        """
        Test vertical interface: left column (stable) vs right column (changing).

        1──────────6  7──────────4
        │          │  │          │  1(0,2)
        │          │  │          │  2(0,0)
        │          │  │          │  3(2,0)
        │          │  │          │  4(2,2)
        │          │  │          │  5(1,0)
        │          │  │          │  6(1,1)
        │          │  │          │  7(1,1)
        │          │  │          │  8(1,0)
        2──────────5  8──────────3

        Expect node ordering [5, 6, 7, 8].
        """
        nodes_stable_parts = [5, 6]
        nodes_for_element = [
            Node(id=5, coordinates=[1.0, 0.0, 0.0]),
            Node(id=6, coordinates=[1.0, 1.0, 0.0]),
            Node(id=7, coordinates=[1.0, 1.0, 0.0]),
            Node(id=8, coordinates=[1.0, 0.0, 0.0])
        ]
        order = UtilsInterface.get_quad4_node_order(nodes_stable_parts, nodes_for_element)
        assert order == [5, 6, 7, 8]

    def test_horizontal_interface_2d(self):
        """
        Test horizontal interface: left column (stable) vs right column (changing).

        7──────────4
        │          │ 1(0,1)
        │          │ 2(0,0)
        │          │ 3(1,1)
        │          │ 4(1,2)
        8──────────3 5(1,0)
        1──────────6 6(1,1)
        │          │ 7(0,2)
        │          │ 8(0,1)
        │          │
        │          │
        2──────────5
        Expect node ordering [6, 1, 8, 3].
        """
        nodes_stable_parts = [1, 6]
        nodes_for_element = [
            Node(id=8, coordinates=[0.0, 1.0, 0.0]),
            Node(id=3, coordinates=[1.0, 1.0, 0.0]),
            Node(id=6, coordinates=[1.0, 1.0, 0.0]),
            Node(id=1, coordinates=[0.0, 1.0, 0.0])
        ]
        order = UtilsInterface.get_quad4_node_order(nodes_stable_parts, nodes_for_element)
        assert order == [6, 1, 8, 3]

    def test_with_angle(self):
        """
        Test with angle the interface is not horizontal or vertical.
        The model parts are also not horizontal or vertical.
        Expect node ordering [7, 3, 8, 4].
        """
        nodes_stable_parts = [7, 3]
        nodes = [
            Node(id=7, coordinates=[0.1, 0.6, 0.0]),
            Node(id=4, coordinates=[0.1, 0.6, 0.0]),
            Node(id=3, coordinates=[0.7, 0.7, 0.0]),
            Node(id=8, coordinates=[0.7, 0.7, 0.0])
        ]
        order = UtilsInterface.get_quad4_node_order(nodes_stable_parts, nodes)
        assert order == [7, 3, 8, 4]

    def test_vertical_interface_3d(self):
        """
        Test vertical interface in 3D.
        """
        # Stable part nodes (bottom triangle of the wedge)
        nodes_stable_parts = {1, 2, 3}
        # All nodes for the interface element
        nodes_for_element = [
            Node(id=1, coordinates=[0.0, 0.0, 0.0]),
            Node(id=2, coordinates=[1.0, 0.0, 0.0]),
            Node(id=3, coordinates=[0.0, 1.0, 0.0]),
            Node(id=4, coordinates=[0.0, 0.0, 0.0]),
            Node(id=5, coordinates=[1.0, 0.0, 0.0]),
            Node(id=6, coordinates=[0.0, 1.0, 0.0])
        ]
        # Expected order: stable nodes first, then corresponding nodes from the other part
        expected_order = [1, 2, 3, 4, 5, 6]
        order = UtilsInterface.get_prism6_node_order(nodes_stable_parts, nodes_for_element)
        assert order == expected_order

    def test_horizontal_interface_3d(self):
        """
        Test horizontal interface in 3D.
        """
        # Stable part nodes
        nodes_stable_parts = {1, 2, 4}
        # All nodes for the interface element
        nodes_for_element = [
            Node(id=1, coordinates=[0.0, 0.0, 0.0]),
            Node(id=2, coordinates=[1.0, 0.0, 0.0]),
            Node(id=3, coordinates=[0.0, 0.0, 0.0]),
            Node(id=4, coordinates=[0.0, 1.0, 0.0]),
            Node(id=5, coordinates=[1.0, 0.0, 0.0]),
            Node(id=6, coordinates=[0.0, 1.0, 0.0])
        ]
        # Expected order: stable nodes first, then corresponding nodes from the other part
        expected_order = [1, 2, 4, 3, 5, 6]
        order = UtilsInterface.get_prism6_node_order(nodes_stable_parts, nodes_for_element)
        assert order == expected_order

    def test_angled_interface_3d(self):
        """
        Test angled interface in 3D.
        """
        # Stable part nodes
        nodes_stable_parts = {1, 2, 5}
        # All nodes for the interface element
        nodes_for_element = [
            Node(id=1, coordinates=[0.0, 0.0, 0.0]),
            Node(id=2, coordinates=[1.0, 0.0, 0.0]),
            Node(id=3, coordinates=[0.0, 0.0, 0.0]),
            Node(id=4, coordinates=[1.0, 0.0, 0.0]),
            Node(id=5, coordinates=[0.5, 1.0, 0.0]),
            Node(id=6, coordinates=[0.5, 1.0, 0.0])
        ]
        # Expected order: stable nodes first, then corresponding nodes from the other part
        expected_order = [1, 2, 5, 3, 4, 6]
        order = UtilsInterface.get_prism6_node_order(nodes_stable_parts, nodes_for_element)
        assert order == expected_order
