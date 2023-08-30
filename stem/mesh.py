import re
from typing import Dict, List, Tuple, Sequence, Union, Any, Optional
from enum import Enum

import numpy as np
import numpy.typing as npt

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

    def __init__(
        self, element_size: float = -1, element_order: int = 1, element_shape: ElementShape = ElementShape.TRIANGLE
    ):
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
        - nodes (Dict[int, Node]): dictionary of node ids followed by node coordinates in an array
        - elements (Dict[int, Element]): dictionary of element ids followed by connectivities in an array

    """

    def __init__(self, ndim: int):
        """
        Initialize the mesh.

        Args:
            ndim (int): number of dimensions of the mesh
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
        # add each element, but first check if counterclockwise
        # revert the node id order if it is not
        # create element per element id

        # flip_node_order = False
        #
        # if len(group_element_ids) == 0


        # In 2D check if vertices of element are clockwise and flip element if they are
        if len(group_element_ids) > 0 and group_data["ndim"] == 2:
            element_info = Mesh.get_2d_element_info(group_element_type)
            node_ids_element = gmsh_elements[group_element_ids[0]]
            # node_ids_element = mesh_data["elements"][group_element_ids[0]]["element_nodes"][group_element_ids[0]]
            coordinates = [nodes[ii].coordinates for ii in node_ids_element]

            # check if vertices are clockwise and flip if they are
            if Utils.are_2d_coordinates_clockwise(coordinates[:element_info["n_vertices"]]):

                Mesh.flip_node_order(group_name, mesh_data, element_info)

        # create element per element id
        elements: Dict[int, Element] = {element_id: Element(element_id, group_element_type,
                                                            mesh_data["elements"][group_element_type][element_id])
                                        for element_id in group_data["element_ids"]}

        # for element_id in group_element_ids:
        #     node_ids_element = element_type_data[element_id]
        #     elements[element_id] = Element(element_id, group_element_type, node_ids_element)
        # #
        # if flip_node_order:


        #
        #     # set all element connectivities in numpy array
        #     element_connectivies = np.zeros((len(group_element_ids), len(element_type_data[group_element_ids[0]])),
        #                                     dtype=int)
        #
        #     for i, element_id in enumerate(group_element_ids):
        #         node_ids_element = element_type_data[element_id]
        #         element_connectivies[i, :] = node_ids_element
        #
        #     # flip the elements nodes
        #     element_connectivies = element_connectivies[:, element_info["reversed_order"]]
        #
        #     a=1+1
        #
        #
        #
        #
        # mesh.flip_node_order(element=elements[group_element_ids[0]])

        #
        #
        # for element_id in group_element_ids:
        #     node_ids_element = element_type_data[element_id]
        #     elements[element_id] = Element(element_id, group_element_type, node_ids_element)
        #
        #     # check of nodes for 2D mesh
        #     if group_data["ndim"] == 2:
        #         element_info = mesh.get_element_info(element=elements[element_id])
        #         # flip the element nodes if they are not anti-clockwise, and only if mesh entity has more than 2 nodes.
        #         if element_info["n_vertices"] > 2:
        #             coordinates = [nodes[ii].coordinates for ii in node_ids_element]
        #             if Utils.are_2d_coordinates_clockwise(coordinates):
        #                 mesh.flip_node_order(element=elements[element_id])

        # add nodes and elements to mesh object
        mesh = cls(group_data["ndim"])
        mesh.nodes = nodes
        mesh.elements = elements

        return mesh

    @staticmethod
    def get_2d_element_info(gmsh_element_type):
        """
        Returns:
        """

        element_mapping_dict = {"POINT_1N": {"ndim": 0,
                                             "order": 1,
                                             "n_vertices": 1,
                                             "reversed_order": [0]},
                                "LINE_2N": {"ndim": 1,
                                            "order": 1,
                                            "n_vertices": 2,
                                            "reversed_order": [1, 0]},
                                "LINE_3N": {"ndim": 1,
                                            "order": 2,
                                            "n_vertices": 2,
                                            "reversed_order": [1, 0, 2]},
                                "TRIANGLE_3N": {"ndim": 2,
                                                "order": 1,
                                                "n_vertices": 3,
                                                "reversed_order": [2, 1, 0]},
                                "TRIANGLE_6N": {"ndim": 2,
                                                "order": 2,
                                                "n_vertices": 3,
                                                "reversed_order": [2, 1, 0, 5, 4, 3]},
                                "QUADRANGLE_4N": {"ndim": 2,
                                                  "order": 1,
                                                  "n_vertices": 4,
                                                  "reversed_order": [3, 2, 1, 0]},
                                "QUADRANGLE_8N": {"ndim": 2,
                                                  "order": 2,
                                                  "n_vertices": 4,
                                                  "reversed_order": [3, 2, 1, 0, 7, 6, 5, 4]},
                                "TETRAHEDRON_4N": {"ndim": 3,
                                                    "order": 1,
                                                    "n_vertices": 4,
                                                    "reversed_order": [3, 2, 1, 0]}}

        # find element order
        if gmsh_element_type not in element_mapping_dict.keys():
            raise NotImplementedError(f"No reversed order defined for the element type: {gmsh_element_type}")

        return element_mapping_dict[gmsh_element_type]

    @staticmethod
    # def flip_node_order(group_name, mesh_data, element_info, elements):
    def flip_node_order(element_info,elements):
        """
        Returns:
        """

        # group_data = mesh_data["physical_groups"][group_name]
        # group_element_ids = group_data["element_ids"]
        # elements = mesh_data["elements"][group_data["element_type"]]

        ids =[element.id for element in elements]
        element_connectivies = np.array([element.node_ids for element in elements])


        # # set all element connectivities in numpy array
        # element_connectivies = np.zeros((len(elements), len(elements[group_element_ids[0]])),
        #                                 dtype=int)

        # get the connectivities for each element
        # for i, element_id in enumerate(ids):
        #     element_connectivies[i, :] = elements[element_id]

        # flip the elements nodes
        element_connectivies = element_connectivies[:, element_info["reversed_order"]]

        # set the flipped connectivities back to the mesh data

        for i, (id, element_connectivity) in enumerate(zip(ids, element_connectivies)):
            elements[i].node_ids = list(element_connectivity)

        # for i, element_id in enumerate(ids):
        #     elements[element_id] = list(element_connectivies[i, :])

    @staticmethod
    def prepare_data_for_kratos(mesh_data: Dict[str, Any]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Prepares mesh data for Kratos

        Args:
            - mesh_data (Dict[str, Any]): dictionary of mesh data


        Returns:
            - nodes (npt.NDArray[np.float64]): node id followed by node coordinates in an array
            - elements (npt.NDArray[np.int64]): element id followed by connectivities in an array
        """

        # create array of nodes where each row is represented by [id, x,y,z]
        nodes = np.concatenate((mesh_data["nodes"]["ids"][:, None], mesh_data["nodes"]["coordinates"]), axis=1)

        all_elements_list = []
        # create array of elements where each row is represented by [id, node connectivities]
        for v in mesh_data["elements"].values():
            all_elements_list.append(np.concatenate((v["element_ids"][:, None], v["element_nodes"]), axis=1))

        all_elements = np.array(all_elements_list).astype(int)

        return nodes, all_elements


if __name__ == '__main__':

    a = np.array([5.0, 7.0, 9.0])

    new_order = [1, 0, 2]

    # set values in order of list of indices
    a = a[new_order]

    print(a)