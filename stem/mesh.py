import numpy as np

# todo create GmshIO its own package
from stem.gmsh_IO import GmshIO
from stem.kratos_IO import KratosIO

class Node:
    def __init__(self, id, coordinates):
        self.id = id
        self.coordinates = coordinates

class Element:
    def __init__(self, id, element_type, node_ids):
        self.id = id
        self.element_type =  element_type
        self.node_ids = node_ids

class Condition:
    def __init__(self, id, element_type, node_ids):
        self.id = id
        self.element_type = element_type
        self.node_ids = node_ids

class Mesh:
    def __init__(self):

        self.ndim = None
        self.nodes = None
        self.elements = None
        self.conditions = None

        pass

    def prepare_data_for_kratos(self, mesh_data):
        """
        gets mesh data for Kratos
        :param mesh_data: dictionary of mesh data
        :return: node id followed by node coordinates and element id followed by node id in an array
        """

        # create array of nodes where each row is represented by [id, x,y,z]
        nodes = np.concatenate((mesh_data["nodes"]["ids"][:, None], mesh_data["nodes"]["coordinates"]), axis=1)

        all_elements=[]
        # create array of elements where each row is represented by [id, node connectivities]
        for v in mesh_data["elements"].values():
            all_elements.append(np.concatenate((v["element_ids"][:, None], v["element_nodes"]), axis=1))

        return nodes, all_elements


    def write_mesh_to_kratos_structure(self, mesh_data, filename):
        """
        Writes mesh data to the structure which can be read by Kratos

        :param mesh_data: dictionary of mesh data
        :param filename: filename of the kratos mesh file
        :return:
        """
        nodes, elements = self.prepare_data_for_kratos(mesh_data)

        kratos_io = KratosIO()
        kratos_io.write_mesh_to_mdpa(nodes, elements, filename)




