import numpy as np

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


        self.elements_1d = None
        self.elements_2d = None
        self.elements_3d = None

        pass

    def prepare_data_for_kratos(self, node_coords, node_tags, elem_tags, node_tag_1D, node_tag_2D, node_tag_3D = None):
        """
        gets mesh data for Kratos
        :param node_coords: node coordinates
        :param node_tags: node tags
        :param elem_tags: all element tags in an array separated by element type
        :param node_tag_1D: node tags of start and end of line
        :param node_tag_2D: node tags of surface
        :return: node tag followed by node coordinates and element tag followed by node tags in an array
        """
        nodes = np.concatenate((node_tags[:, None], np.array(node_coords)), axis=1)
        elements_1d = np.concatenate((elem_tags[0][:, None], np.array(node_tag_1D)), axis=1)
        elements_2d = np.concatenate((elem_tags[1][:, None], np.array(node_tag_2D)), axis=1)

        if node_tag_3D is not None:
            elements_3d = np.concatenate((elem_tags[2][:, None], np.array(node_tag_3D)), axis=1)
        else:
            elements_3d = None

        return nodes, elements_1d, elements_2d, elements_3d


    def write_mesh_to_kratos_structure(self, filename):

        self.prepare_data_for_kratos()

        kratos_io = KratosIO()

        kratos_io.

        kratos_io.write_mesh_to_mdpa()




