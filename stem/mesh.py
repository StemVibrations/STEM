from typing import Dict, List, Sequence, Any
from enum import Enum

import numpy as np
import numpy.typing as npty

from stem.globals import ELEMENT_DATA
from stem.utils import Utils


class ElementShape(Enum):
    """
    Enum class for the element shape. TRIANGLE for triangular elements and tetrahedral elements, QUADRILATERAL for
    quadrilateral elements and hexahedral elements.

    """

    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"


class MeshSettings:
    """
    A class to represent the mesh settings.

    Attributes:
        - element_size (float): The element size (default -1, which means that gmsh determines the size).
        - element_shape (:class:`ElementShape`): The element shape. TRIANGLE for triangular elements and \
            tetrahedral elements,  QUADRILATERAL for quadrilateral elements and hexahedral elements. (default TRIANGLE)
        - __element_order (int): The element order. 1 for linear elements, 2 for quadratic elements. (default 1)
    """

    def __init__(self,
                 element_size: float = -1,
                 element_order: int = 1,
                 element_shape: ElementShape = ElementShape.TRIANGLE):
        """
        Initialize the mesh settings.

        Args:
            - element_size (float): The element size (default -1, which means that gmsh determines the size).
            - element_order (int): The element order. 1 for linear elements, 2 for quadratic elements. (default 1)
            - element_shape (:class:`ElementShape`): The element shape. TRIANGLE for triangular elements and \
            tetrahedral elements,  QUADRILATERAL for quadrilateral elements and hexahedral elements. (default TRIANGLE)
        """
        self.element_size: float = element_size
        self.element_shape: ElementShape = element_shape

        if element_order not in [1, 2]:
            raise ValueError("The element order must be 1 or 2. Higher order elements are not supported.")

        self.__element_order: int = element_order

    @property
    def element_order(self) -> int:
        """
        Get the element order.

        Returns:
            - int: element order
        """
        return self.__element_order

    @element_order.setter
    def element_order(self, element_order: int):
        """
        Set the element order. The element order must be 1 or 2.

        Args:
            - element_order (int): element order

        Raises:
            - ValueError: If the element order is not 1 or 2.
        """

        if element_order not in [1, 2]:
            raise ValueError("The element order must be 1 or 2. Higher order elements are not supported.")

        self.__element_order = element_order


class Node:
    """
    Class containing information about a node

    Attributes:
        - id (int): node id
        - coordinates (Sequence[float]): node coordinates

    """

    def __init__(self, id: int, coordinates: Sequence[float]):
        """
        Initialize the node.

        Args:
            - id (int): Node id
            - coordinates (Sequence[float]): Node coordinates
        """
        self.id: int = id
        self.coordinates: Sequence[float] = coordinates

    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal. Nodes are considered equal if their ids and coordinates are equal.

        Args:
            - other (object): The other node.

        Returns:
            - bool: True if the nodes are equal, False otherwise.

        """
        # check if the other object is an instance of the Node class
        if not isinstance(other, Node):
            return False

        # check if the id and coordinates are equal
        return self.id == other.id and np.isclose(self.coordinates, other.coordinates).all().item()


class Element:
    """
    Class containing information about an element

    Attributes:
        - id (int): element id
        - element_type (str): Gmsh element type
        - node_ids (Sequence[int]): node ids

    """

    def __init__(self, id: int, element_type: str, node_ids: List[int]):
        """
        Initialize the element.

        Args:
            - id (int): Element id
            - element_type (str): Gmsh-element type
            - node_ids (List[int]): Node connectivities
        """
        self.id: int = id
        self.element_type: str = element_type
        self.node_ids: List[int] = node_ids

    def __eq__(self, other: object) -> bool:
        """
        Check if two elements are equal. Elements are considered equal if their ids, element types and node ids are
        equal.

        Args:
            - other (object): The other element.

        Returns:
            - bool: True if the elements are equal, False otherwise.
        """

        # check if the other object is an instance of the Element class
        if not isinstance(other, Element):
            return False

        # check if the id, element type and node ids are equal
        return self.id == other.id and self.element_type == other.element_type and self.node_ids == other.node_ids


class Mesh:
    """
    Class containing information about the mesh

    Args:
        - ndim (int): number of dimensions of the mesh

    Attributes:
        - ndim (int): number of dimensions of the mesh
        - nodes (Dict[int, :class:`Node`]): dictionary of node ids followed by the node object
        - elements (Dict[int, :class:`Element`]): dictionary of element ids followed by the element object

    """

    def __init__(self, ndim: int):
        """
        Initialize the mesh.

        Args:
            - ndim (int): number of dimensions of the mesh
        """

        self.ndim: int = ndim
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}

    def __getattribute__(self, item: str) -> Any:
        """
        Overrides the getattribute method of the object class.

        Args:
            - item (str): The name of the attribute.

        Returns:
            - Any: The attribute.

        """
        # Make sure that the create_mesh_from_gmsh_group method cannot be
        # called on an initialised mesh instance
        if item == "create_mesh_from_gmsh_group":
            raise AttributeError(f"Cannot call class method: {item} from an initialised mesh instance.")
        else:
            return super().__getattribute__(item)

    def __eq__(self, other: object) -> bool:
        """
        Check if two meshes are equal. Two meshes are considered equal if their nodes and elements are equal.

        Args:
            - other (object): The other mesh.

        Returns:
            - bool: True if the meshes are equal, False otherwise.

        """
        # check if the other object is an instance of the Mesh class
        if not isinstance(other, Mesh):
            return False

        # check if the number of dimensions, nodes and elements are equal
        return self.ndim == other.ndim and self.nodes == other.nodes and self.elements == other.elements

    @classmethod
    def create_mesh_from_gmsh_group(cls, mesh_data: Dict[str, Any], group_name: str) -> "Mesh":
        """
        Creates a mesh object from gmsh group

        Args:
            - mesh_data (Dict[str, Any]): dictionary of mesh data
            - group_name (str): name of the group

        Raises:
            - ValueError: If the group name is not found in the mesh data

        Returns:
            - :class:`Mesh`: mesh object
        """

        if group_name not in mesh_data["physical_groups"]:
            raise ValueError(f"Group {group_name} not found in mesh data")

        # create mesh object
        group_data = mesh_data["physical_groups"][group_name]

        group_element_ids = group_data["element_ids"]
        group_node_ids = group_data["node_ids"]
        group_element_type = group_data["element_type"]

        gmsh_elements = mesh_data["elements"][group_element_type]

        # create node per node id
        nodes: Dict[int, Node] = {node_id: Node(node_id, mesh_data["nodes"][node_id]) for node_id in group_node_ids}

        # create element per element id
        elements: Dict[int, Element] = {
            element_id: Element(element_id, group_element_type, mesh_data["elements"][group_element_type][element_id])
            for element_id in group_data["element_ids"]
        }

        # In 2D check if vertices of element are clockwise and flip element if they are
        if len(group_element_ids) > 0 and group_data["ndim"] == 2:
            element_info = ELEMENT_DATA[group_element_type]

            # only check the first element in the group. The rest of the elements have the same node order
            node_ids_element = gmsh_elements[group_element_ids[0]]
            coordinates = [nodes[ii].coordinates for ii in node_ids_element]

            # check if vertices are clockwise and flip if they are
            if Utils.are_2d_coordinates_clockwise(coordinates[:element_info["n_vertices"]]):

                # flip the node order in of each element in the group
                Utils.flip_node_order(list(elements.values()))

                # also flip the node order in the mesh data
                for element_id, element in elements.items():
                    mesh_data["elements"][group_element_type][element_id] = element.node_ids

        # add nodes and elements to mesh object
        mesh = cls(group_data["ndim"])
        mesh.nodes = nodes
        mesh.elements = elements

        return mesh

    def calculate_centroids(self) -> npty.NDArray[np.float64]:
        """
        Calculate the centroids of all elements

        Returns:
            - npty.NDArray[np.float64]: centroids of all elements
        """

        centroids: npty.NDArray[np.float64] = np.mean([[self.nodes[nid].coordinates for nid in el.node_ids]
                                                       for el in self.elements.values()],
                                                      axis=1)

        return centroids

    def find_elements_connected_to_nodes(self) -> Dict[int, List[int]]:
        """
        Creates a dictionary of node ids as keys and a list of element ids which are connected to the node as values.


        Returns:
            - Dict[int, List[int]]: dictionary containing node ids as keys and  a list of element ids which are
            connected to the node as values.

        """

        # find which elements are connected to each node
        node_to_elements: Dict[int, List[int]] = {node_id: [] for node_id in self.nodes.keys()}

        for element_id, element in self.elements.items():
            for node_id in element.node_ids:
                node_to_elements[node_id].append(element_id)

        return node_to_elements
