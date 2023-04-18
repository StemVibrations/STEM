
from stem.gmsh_IO import GmshIO
from stem.kratos_IO import KratosIO

import numpy as np


class Geometry:
    def __init__(self):



        pass

    def get_gmsh_data(self):
        pass


def get_data_for_kratos(node_coords, node_tags, elem_tags, node_tag_1D, node_tag_2D, node_tag_3D = None):
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


if __name__ == '__main__':
    input_points, depth, mesh_size, dims, save_file, name_label, mesh_output_name, gmsh_interface = init()
    generate_gmsh_mesh(input_points, depth, mesh_size, dims, save_file, name_label, mesh_output_name, gmsh_interface)
