from typing import List, Set
import numpy as np

from stem.mesh import Node


class UtilsInterface:
    """
    Utility class for handling node ordering and interface management in a mesh.
    """

    @staticmethod
    def get_quad4_node_order(node_ids_part_1: Set[int], nodes_for_element: List[Node]) -> List[int]:
        """
        Order 4 interface nodes using the following order:
        1. Start with the initial nodes that belong to the stable parts so the ones that are not changed
            during the creation of the new element.
        2. The next node is the one that has the same coordinates as the last node of the initial nodes.
        3. The last node is the one that is not part of the initial nodes and has the same
            coordinates as the last node of the initial nodes.

                v
                ^
                |
          3-----------2
          |     |     |
          |     |     |
          |     +---- | --> u
          |           |
          |           |
          0-----------1

        Args:
            - node_ids_part_1 (Set[int]): list of stable nodes ids
            - nodes_for_element (List[Node]): list of nodes for the element to be ordered
        Returns:
            - List[int]: ordered node ids in the order [bottom-left, bottom-right, top-right, top-left]
        """
        # Create a new element with the node ids
        # Step 1: Get the nodes from the initial element so the ones that are in the stable parts and
        # in the nodes_for_element_dict
        quad4_node_collection = {node.id: node for node in nodes_for_element if node.id in node_ids_part_1}
        # Step 2: The next element is the one that has the same coordinates as the last node of the initial nodes
        equal_node = {
            node.id: node
            for node in nodes_for_element if node.id not in list(quad4_node_collection.keys())
            and np.allclose(node.coordinates,
                            list(quad4_node_collection.values())[-1].coordinates)
        }
        quad4_node_collection.update(equal_node)
        # Step 3: Get the final node that the key is not part of the initial nodes
        for node in nodes_for_element:
            if node.id not in list(quad4_node_collection.keys()):
                last_node = node
                break
        quad4_node_collection[last_node.id] = last_node
        return list(quad4_node_collection.keys())
