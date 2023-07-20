from typing import Dict, List, Tuple, Union, Any, Optional
from enum import Enum

import numpy as np
import numpy.typing as npt

from stem.IO.kratos_io import KratosIO


class Node:
    """
    Class containing information about a node

    Attributes:
        - id (int): node id
        - coordinates (np.array): node coordinates

    """
    def __init__(self, id, coordinates):
        self.id = id
        self.coordinates = coordinates

# class ElementType(Enum):
#
#
# def get_element_type_from

class Element:
    """
    Class containing information about an element

    Attributes:
        - id (int): element id
        - element_type (str): element type
        - node_ids (Union[List[int], npt.NDArray[np.int64]]): node ids

    """
    def __init__(self, id: int, element_type: str, node_ids: Union[List[int], npt.NDArray[np.int64]]):
        self.id: int = id
        self.element_type: str = element_type
        self.node_ids: Union[List[int], npt.NDArray[np.int64]] = node_ids


class Mesh:
    """
    Class containing information about the mesh

    Args:
        - ndim (int): number of dimensions of the mesh

    Attributes:
        - ndim (int): number of dimensions of the mesh
        - nodes (np.array or None): node id followed by node coordinates in an array
        - elements (np.array or None): element id followed by connectivities in an array
        - conditions (np.array or None): condition id followed by connectivities in an array

    """
    def __init__(self, ndim: int):

        self.ndim: Optional[int] = None
        self.nodes = None
        self.elements = None

    @classmethod
    def read_mesh_from_gmsh(cls, mesh_file_name: str) -> None:
        #todo implement this method to read mesh from gmsh file and create a mesh object with the data read from the
        # file.
        pass

    @classmethod
    def create_mesh_from_mesh_data(cls, mesh_data: Dict[str, Any]):
        """
        Creates a mesh object from mesh data

        Args:
            - mesh_data (Dict[str, Any]): dictionary of mesh data

        Returns:
            - :class:`Mesh`: mesh object
        """

        # create mesh object

        node_data = mesh_data["nodes"]
        element_data = mesh_data["elements"]

        nodes = []
        for node_id, coordinates in node_data.items():
            node = Node(node_id, coordinates)
            nodes.append(node)

        elements = []
        for element_type, element_type_data in element_data.items():
            for element_id, element_node in element_type_data.items():
                element = Element(element_id, element_type, element_node)
                elements.append(element)

        mesh = cls(mesh_data["ndim"])
        mesh.nodes = nodes
        mesh.elements = elements

        return mesh

    @classmethod
    def create_mesh_from_gmsh_group(cls, mesh_data, group_name):
        """
        Creates a mesh object from gmsh group

        Args:
            - mesh_data (Dict[str, Any]): dictionary of mesh data
            - group_name (str): name of the group

        Returns:
            - :class:`Mesh`: mesh object
        """
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
        mesh = cls(mesh_data["ndim"])
        mesh.nodes = nodes
        mesh.elements = elements

        return mesh


    def prepare_data_for_kratos(self, mesh_data: Dict[str, Any]) \
            -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
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


    def write_mesh_to_kratos_structure(self, mesh_data: Dict[str, Any], filename: str) -> None:
        """
        Writes mesh data to the structure which can be read by Kratos

        Args:
            - mesh_data (Dict[str, Any]): dictionary of mesh data
            - filename (str): filename of the kratos mesh file

        Returns:
        """

        nodes, elements = self.prepare_data_for_kratos(mesh_data)

        kratos_io = KratosIO(self.ndim)
        kratos_io.write_mesh_to_mdpa(filename)




