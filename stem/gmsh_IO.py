from typing import Dict, List, Union, Type
from enum import Enum
import re

import gmsh
import numpy as np
import numpy.typing as npt


#todo Put this file in its own package, e.g. GmshUtils

class ElementType(Enum):
    """
    Enum of the element types as present in Gmsh, where the enum value corresponds to the element type number in gmsh

    """

    LINE_2N = 1
    TRIANGLE_3N = 2
    QUADRANGLE_4N = 3
    TETRAHEDRON_4N = 4
    HEXAHEDRON_8N = 5
    LINE_3N = 8
    TRIANGLE_6N = 9
    TETRAHEDRON_10N = 11
    POINT_1N = 15
    QUADRANGLE_8N = 16
    HEXAHEDRON_20N = 17


class GmshIO:
    """
    Class for reading and writing mesh data to and from Gmsh

    Attributes
    ----------
    mesh_data : Dict
        Dictionary containing the mesh data, i.e. nodal ids and coordinates; and elemental ids, connectivity's
        and element types.

    """

    def __init__(self):
        self.__mesh_data = {}

    @property
    def mesh_data(self) -> Dict[str, object]:
        """
        Returns the mesh data dictionary

        Returns:
            Dict: Dictionary containing the mesh data, i.e. nodal ids and coordinates; and elemental ids, connectivity's
            and element types.
        """

        return self.__mesh_data

    def create_point(self, coordinates: Union[List[float], npt.NDArray[np.float64]], element_size: float) -> None:
        """
        Creates points in gmsh.

        Args:
            coordinates (Union[List[float], npt.NDArray[float]]): An Iterable of point x,y,z coordinates.
            mesh_size (float): The element size.

        Returns
        -------
        None
        """
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        gmsh.model.geo.addPoint(x, y, z, element_size)

    def create_line(self, point_ids: Union[List[int], npt.NDArray[np.int_]]) -> None:
        """
        Creates lines in gmsh.

        Args:
            point_ids (Union[List[int], npt.NDArray[int]]): A list of point tags in order.

        Returns:
            None
        """

        point1 = point_ids[0]
        point2 = point_ids[1]
        gmsh.model.geo.addLine(point1, point2)


    def create_surface(self, line_ids: Union[List[int], npt.NDArray[np.int_]], name_label: str) -> int:
        """
        Creates curve and then surface in gmsh by using line tags.

        Args:
            line_ids (Union[List[int], npt.NDArray[int]]): A list of line tags in order.
            name_label (str): The surface name label provided by user input.

        Returns:
            int: surface id
        """

        gmsh.model.geo.addCurveLoop(line_ids, 1)
        surface_id: int = gmsh.model.geo.addPlaneSurface([1], 1)

        surface_ndim = 2
        gmsh.model.setPhysicalName(surface_ndim, surface_id, name_label)
        return surface_id

    def create_volume_by_extruding_surface(self, surface_id: int,
                                           extrusion_length: Union[List[float], npt.NDArray[np.float64]]) -> None:
        """
        Creates volume by extruding a 2D surface

        Args:
            surface_id (int): The surface tag.
            extrusion_length (Union[List[float], npt.NDArray[float]]): The extrusion length in x, y and z direction.

        Returns:
            None
        """

        surface_ndim = 2
        gmsh.model.geo.extrude([(surface_ndim, surface_id)], extrusion_length[0], extrusion_length[1],
                               extrusion_length[2])

    def make_geometry_2D(self, point_coordinates: Union[List[List[float]], npt.NDArray[np.float64]],
                         point_pairs: Union[List[List[int]], npt.NDArray[np.int_]],
                         element_size: float, name_label: str) -> int:
        """
        Takes point_pairs and puts their tags as the beginning and end of line in gmsh to create line,
        then creates surface to make 2D geometry.

        Args:
            point_coordinates (Union[List[float], npt.NDArray[np.float64]]): A list of point coordinates.
            point_pairs (Union[List[List[int]], npt.NDArray[npt.NDArray[int]]]): A list of point tags of two
                consecutive points in an array.
            element_size (float): The mesh size provided by user.
            name_label (str): The surface name label provided by user input.

        Returns:
            int: Surface id.
        """

        for point in point_coordinates:
            coordinate = [point[0], point[1], point[2]]
            self.create_point(coordinate, element_size)

        line_lists = []

        for i in range(len(point_pairs)):
            line = [point_pairs[i][0], point_pairs[i][1]]
            line_lists.append(i + 1)
            self.create_line(line)

        surface_id: int = self.create_surface(line_lists, name_label)
        return surface_id

    def make_geometry_3D(self, point_coordinates: Union[List[List[float]], npt.NDArray[np.float64]],
                         point_pairs: Union[List[List[int]], npt.NDArray[np.int_]],
                         mesh_size: float, extrusion_length: Union[List[float],
                                                                   npt.NDArray[np.float64]], name_label: str) -> None:
        """
        Creates 3D geometries by extruding the 2D surface

        Args:
            point_coordinates (Union[List[float], npt.NDArray[float]]): Geometry points coordinates.
            point_pairs (Union[List[List[int]], npt.NDArray[npt.NDArray[int]]]): A list of point tags of two consecutive
                points in an array.
            mesh_size (float): Mesh size.
            extrusion_length (Union[List[float], npt.NDArray[float]]): The extrusion length in x, y and z direction.
            name_label (str): surface name label from user input.

        Returns:
            None
        """

        surfaces = self.make_geometry_2D(point_coordinates, point_pairs, mesh_size, name_label)
        self.create_volume_by_extruding_surface(surfaces, extrusion_length)

    def generate_point_pairs(self, point_coordinates: Union[List[List[float]], npt.NDArray[np.float64]]) \
            -> List[List[int]]:
        """
        Generates pairs of point IDs which form a line

        Args:
            point_coordinates (Union[List[List[float]], npt.NDArray[float]]): A list of geometry points coordinates.

        Returns:
            List[List[int]]: A list of pairs of point IDs which create a line.
        """

        # puts two consecutive points tags as the beginning and end of line in an array
        point_pairs = [[i+1, i+2] for i in range(len(point_coordinates)-1)]

        # make a pair that connects last point to first point
        point_pairs.append([len(point_coordinates), 1])

        return point_pairs

    @staticmethod
    def get_num_nodes_from_elem_type(elem_type: int) -> int:
        """
        Gets number of nodes from element types

        Args:
            elem_type (int): An integer that defines the type of element.

        Returns:
            int: The number of nodes needed for a type of element.
        """

        # get name from element type enum
        element_name = ElementType(elem_type).name

        # get number of nodes from the enum name
        num_nodes = int(re.findall(r'\d+', element_name)[0])

        return num_nodes

    def generate_gmsh_mesh(self, point_coordinates: Union[List[List[float]], npt.NDArray[np.float64]],
                           extrusion_length: Union[List[float], npt.NDArray[np.float64]], mesh_size: float, dims: int,
                           name_label: str, mesh_name: str, mesh_output_dir: str, save_file: bool = False,
                           open_gmsh_gui: bool = False) -> None:
        """
        Creates point pairs by storing point tags of two consecutive points in an array,
        then generates mesh for geometries in gmsh.

        Args:
            point_coordinates (Union[List[List[float]], npt.NDArray[np.float64]]): User input points of the surface as
                a list or ndarray.
            extrusion_length (Union[List[float], npt.NDArray[float]]): The depth of 3D geometry.
            mesh_size (float): The mesh size provided by user.
            dims (int): The dimension of geometry (2=2D or 3=3D).
            name_label (str): The surface name label provided by user input.
            mesh_name (str): Name of gmsh model and mesh output file.
            mesh_output_dir (str): Output directory of mesh file.
            save_file (bool, optional): If True, saves mesh data to gmsh msh file. (default is False)
            open_gmsh_gui (bool, optional): User indicates whether to open gmsh interface (default is False)

        Returns:
            None
        """

        #todo add check for clockwise or anticlockwise
        point_pairs = self.generate_point_pairs(point_coordinates)

        gmsh.initialize()
        gmsh.model.add(mesh_name)

        if dims == 3:
            self.make_geometry_3D(point_coordinates, point_pairs, mesh_size, extrusion_length, name_label)

        elif dims == 2:
            self.make_geometry_2D(point_coordinates, point_pairs, mesh_size, name_label)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(dims)

        # extracts mesh data from gmsh
        self.__mesh_data = self.extract_mesh_data(gmsh.model.mesh)

        if save_file:
            # writes mesh file output in .msh format
            file_extension = ".msh"
            mesh_output_file = mesh_output_dir + mesh_name + file_extension
            gmsh.write(mesh_output_file)

        # opens Gmsh interface
        if open_gmsh_gui:
            gmsh.fltk.run()

        gmsh.finalize()

    def extract_element_data(self, elem_type: int, elem_tags: List[int], element_connectivities: List[int]) -> \
            Dict[str, object]:
        """
        Extracts element data from gmsh mesh

        Args:
            elem_type (int): Element type.
            elem_tags (List[int]): Element tags.
            element_connectivities (List[int]): Element node tags.

        Returns:
            dict: Dictionary which contains element data.
        """

        element_name = ElementType(elem_type).name
        n_nodes_per_element = self.get_num_nodes_from_elem_type(elem_type)
        num_elements = len(elem_tags)
        connectivities = np.reshape(element_connectivities, (num_elements, n_nodes_per_element))

        return {element_name: {"element_ids": elem_tags,
                               "connectivities": connectivities}}


    def extract_mesh_data(self, gmsh_mesh: Type[gmsh.model.mesh]):
        """
        Gets gmsh output data

        Args:
            gmsh_mesh (gmsh.model.mesh): The mesh as generated by gmsh.

        Returns:
            dict: Dictionary which contains nodal and elemental information.
        """

        mesh_data: Dict[str, Dict[str, object]] = {"nodes": {},
                                                   "elements": {}}

        # get nodal information
        node_tags, node_coords, node_params = gmsh_mesh.getNodes()  # nodes, elements

        # reshape nodal coordinate array to [num nodes, 3]
        num_nodes = len(node_tags)
        node_coordinates = np.reshape(node_coords, (num_nodes, 3))

        mesh_data["nodes"]["coordinates"] = node_coordinates
        mesh_data["nodes"]["ids"] = node_tags

        # get all elemental information
        elem_types, elem_tags, elem_node_tags = gmsh_mesh.getElements()

        # todo, this is unhandy for the future and the connection to kratos, handier would be to group elements by physical group
        for elem_type, elem_tag, elem_node_tag in zip(elem_types, elem_tags, elem_node_tags):
            element_dict = self.extract_element_data(elem_type, elem_tag, elem_node_tag)
            mesh_data["elements"].update(element_dict)

        return mesh_data
