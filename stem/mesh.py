from typing import Dict

import numpy as np

# todo create GmshIO its own package
from stem.gmsh_IO import GmshIO
from stem.kratos_IO import KratosIO

class Node:
    """
    Class containing information about a node

    Attributes:
        id (int): node id
        coordinates (np.array): node coordinates

    """
    def __init__(self, id, coordinates):
        self.id = id
        self.coordinates = coordinates

class Element:
    """
    Class containing information about an element

    Attributes:
        id (int): element id
        element_type (str): element type
        node_ids (np.array): node ids

    """
    def __init__(self, id, element_type, node_ids):
        self.id = id
        self.element_type =  element_type
        self.node_ids = node_ids

class Condition:
    """
    Class containing information about a condition

    Attributes:
        id (int): condition id
        element_type (str): element type
        node_ids (np.array): node ids

    """
    def __init__(self, id, element_type, node_ids):
        self.id = id
        self.element_type = element_type
        self.node_ids = node_ids

class Mesh:
    """
    Class containing information about the mesh

    Attributes
    ----------
        ndim (int or None): number of dimensions of the mesh
        nodes (np.array or None): node id followed by node coordinates in an array
        elements (np.array or None): element id followed by connectivities in an array
        conditions (np.array or None): condition id followed by connectivities in an array

    """
    def __init__(self):

        self.ndim = None
        self.nodes = None
        self.elements = None
        self.conditions = None

        pass

    def prepare_data_for_kratos(self, mesh_data) -> (np.ndarray, np.ndarray):
        """
        Prepares mesh data for Kratos

        Args:
            mesh_data (Dict): dictionary of mesh data


        Returns:
            nodes (np.array): node id followed by node coordinates in an array
            elements (np.array): element id followed by connectivities in an array

        """

        # create array of nodes where each row is represented by [id, x,y,z]
        nodes = np.concatenate((mesh_data["nodes"]["ids"][:, None], mesh_data["nodes"]["coordinates"]), axis=1)

        all_elements = []
        # create array of elements where each row is represented by [id, node connectivities]
        for v in mesh_data["elements"].values():
            all_elements.append(np.concatenate((v["element_ids"][:, None], v["element_nodes"]), axis=1))

        return nodes, all_elements


    def write_mesh_to_kratos_structure(self, mesh_data, filename):
        """
        Writes mesh data to the structure which can be read by Kratos

        Args:
            mesh_data (Dict): dictionary of mesh data
            filename (str): filename of the kratos mesh file

        Returns:
            -
        """

        nodes, elements = self.prepare_data_for_kratos(mesh_data)

        kratos_io = KratosIO()
        kratos_io.write_mesh_to_mdpa(nodes, elements, filename)




