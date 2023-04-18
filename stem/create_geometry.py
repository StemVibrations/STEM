
from stem.gmsh_IO import GmshIO
from stem.kratos_IO import KratosIO

import numpy as np


class Geometry:
    def __init__(self):



        pass

    def get_gmsh_data(self):
        pass



def init():
    """
    gets user input
    :return: input_points, depth, mesh_size, dims, save_file, name_label, mesh_output_name, gmsh_interface
    """
    # define the points of the surface as a list of tuples
    input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
    # define the mesh size
    mesh_size = 2
    # define geometry dimension; input "3" for 3D to extrude the 2D surface, input "2" for 2D
    dims = 3
    # if 3D, input depth of geometry to be extruded from 2D surface
    depth = 2
    # set a name label for the surface
    name_label = "Soil Layer"
    # if "True", saves mesh data to separate mdpa files; otherwise "False"
    save_file = True
    # if "True", opens gmsh interface; otherwise "False"
    gmsh_interface = True
    # set a name for mesh output file
    mesh_output_name = "geometry"

    return input_points, depth, mesh_size, dims, save_file, name_label, mesh_output_name, gmsh_interface


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
