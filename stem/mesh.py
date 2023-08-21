from typing import Dict, List, Sequence, Any
from enum import Enum


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

    def __init__(self, element_size: float = -1, element_order: int = 1,
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
            id (int): Node id
            coordinates (Sequence[float]): Node coordinates
        """
        self.id: int = id
        self.coordinates: Sequence[float] = coordinates


class Element:
    """
    Class containing information about an element

    Attributes:
        - id (int): element id
        - element_type (str): Gmsh element type
        - node_ids (Sequence[int]): node ids

    """

    def __init__(self, id: int, element_type: str, node_ids: Sequence[int]):
        """
        Initialize the element.

        Args:
            id (int): Element id
            element_type (str): Gmsh-element type
            node_ids (Sequence[int]): Node connectivities
        """
        self.id: int = id
        self.element_type: str = element_type
        self.node_ids: Sequence[int] = node_ids


class Mesh:
    """
    Class containing information about the mesh

    Args:
        - ndim (int): number of dimensions of the mesh

    Attributes:
        - ndim (int): number of dimensions of the mesh
        - nodes (List[Node]): node id followed by node coordinates in a list
        - elements (List[Element]): element id followed by connectivities in a list

    """
    def __init__(self, ndim: int):
        """
        Initialize the mesh.

        Args:
            ndim (int): number of dimensions of the mesh
        """

        self.ndim: int = ndim
        self.nodes: List[Node] = []
        self.elements: List[Element] = []

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

        element_type_data = mesh_data["elements"][group_element_type]

        # create element per element id
        elements = [Element(element_id, group_element_type, element_type_data[element_id])
                    for element_id in group_element_ids]

        # create node per node id
        nodes = [Node(node_id, mesh_data["nodes"][node_id]) for node_id in group_node_ids]

        # add nodes and elements to mesh object
        mesh = cls(group_data["ndim"])
        mesh.nodes = nodes
        mesh.elements = elements

        return mesh
