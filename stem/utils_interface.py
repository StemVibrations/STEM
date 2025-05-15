from typing import Dict, List, Tuple
import numpy as np

from stem.mesh import Node
from stem.model_part import BodyModelPart


class UtilsInterface:
    """
    Utility class for handling node ordering and interface management in a mesh.
    """

    @staticmethod
    def get_quadratic_order_nodes(nodes_stable_parts: List[int], nodes_for_element: List[Node]) -> List[int]:
        """
        Order 4 interface nodes using the following order:
        1. Start with the initial nodes that belong to the stable parts so the ones that are not changed during the creation of the new element.
        2. The next node is the one that has the same coordinates as the last node of the initial nodes.
        3. The last node is the one that is not part of the initial nodes and has the same coordinates as the last node of the initial nodes.

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
            - nodes_stable_parts (List[int]): list of stable nodes ids
            - nodes_for_element (List[Node]): list of nodes for the element to be ordered
        Returns:
            - List[int]: ordered node ids in the order [bottom-left, bottom-right, top-right, top-left]
        """
        # Create a new element with the node ids
        # Step 1: Get the nodes from the initial element so the ones that are in the stable parts and in the nodes_for_element_dict
        initial_nodes = {node.id: node for node in nodes_for_element if node.id in nodes_stable_parts}
        # Step 2: The next element is the one that has the same coordinates as the last node of the initial nodes
        equal_node = {
            node.id: node
            for node in nodes_for_element
            if node.id not in list(initial_nodes.keys()) and np.allclose(node.coordinates,
                                                                         list(initial_nodes.values())[-1].coordinates)
        }
        initial_nodes.update(equal_node)
        # Step 3: Get the final node that the key is not part of the initial nodes
        for node in nodes_for_element:
            if node.id not in list(initial_nodes.keys()):
                last_node = node
                break
        initial_nodes[last_node.id] = last_node
        return list(initial_nodes.keys())
