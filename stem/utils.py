from typing import Sequence, Dict, Any, List, Union, Optional, Generator, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stem.mesh import Element


class Utils:
    """
    Class containing utility methods.

    """
    @staticmethod
    def check_ndim_nnodes_combinations(n_dim: int, n_nodes_element: Optional[int],
                                       available_combinations: Dict[int, List[Any]],
                                       class_name: str):
        """
        Check if the combination of number of global dimensions and number of nodes per element is supported.

        Args:
            - n_dim (int): number of dimensions
            - n_nodes_element (int): number of nodes per element or condition-element
            - available_combinations (Dict[int, List[int]]): dictionary containing the supported combinations of number\
               of dimensions and number of nodes per element or condition-element
            - class_name (str): name of the class to be checked

        Raises:
            - ValueError: when the number of dimensions is not supported.
            - ValueError: when the combination of number of dimensions and number of nodes per element is not supported.

        """
        # check if the number of dimensions is supported
        if n_dim not in available_combinations.keys():
            raise ValueError(f"Number of dimensions {n_dim} is not supported for {class_name} elements. Supported "
                             f"dimensions are {list(available_combinations.keys())}.")

        # check if the number of nodes per element is supported
        if n_nodes_element not in available_combinations[n_dim]:
            raise ValueError(
                f"In {n_dim} dimensions, only {available_combinations[n_dim]} noded {class_name} elements are "
                f"supported. {n_nodes_element} nodes were provided."
            )

    @staticmethod
    def are_2d_coordinates_clockwise(coordinates: Sequence[Sequence[float]]) -> bool:
        """
        Checks if the 2D coordinates are given in clockwise order. If the signed area is positive, the coordinates
        are given in clockwise order.

        Args:
            - coordinates (Sequence[Sequence[float]]): coordinates of the points of a surface

        Returns:
            - bool: True if the coordinates are given in clockwise order, False otherwise.
        """

        # calculate signed area of polygon
        signed_area = 0.0
        for i in range(len(coordinates) - 1):
            signed_area += (coordinates[i + 1][0] - coordinates[i][0]) * (coordinates[i + 1][1] + coordinates[i][1])

        signed_area += (coordinates[0][0] - coordinates[-1][0]) * (coordinates[0][1] + coordinates[-1][1])

        # if signed area is positive, the coordinates are given in clockwise order
        return signed_area > 0.0

    @staticmethod
    def check_dimensions(points:Sequence[Sequence[float]]) -> None:
        """

        Check if points have the same dimensions (2D or 3D).

        Args:
            - points: (Sequence[Sequence[float]]): points to be tested

        Raises:
            - ValueError: when the points have different dimensions.
            - ValueError: when the dimension is not either 2 or 3D.

        Returns:
            - None
        """

        lengths = [len(point) for point in points]
        if len(np.unique(lengths)) != 1:
            raise ValueError("Mismatch in dimension of given points!")

        if any([ll not in [2, 3] for ll in lengths]):
            raise ValueError("Dimension of the points should be 2D or 3D.")

    @staticmethod
    def is_collinear(point: Sequence[float], start_point: Sequence[float], end_point: Sequence[float],
                     a_tol: float = 1e-06) -> bool:
        """
        Check if point is aligned with the other two on a line. Points must have the same dimension (2D or 3D)

        Args:
            - point (Sequence[float]): point coordinates to be tested
            - start_point (Sequence[float]): coordinates of first point of a line
            - end_point (Sequence[float]): coordinates of second point of a line
            - a_tol (float): absolute tolerance to check collinearity (default 1e-6)

        Raises:
            - ValueError: when there is a dimension mismatch in the point dimensions.

        Returns:
            - bool: whether the point is aligned or not
        """

        # check dimensions of points for validation
        Utils.check_dimensions([point, start_point, end_point])

        vec_1 = np.asarray(point) - np.asarray(start_point)
        vec_2 = np.asarray(end_point) - np.asarray(start_point)

        # cross product of the two vector
        cross_product = np.cross(vec_1, vec_2)

        # It should be smaller than tolerance for points to be aligned
        is_collinear: bool = np.sum(np.abs(cross_product)) < a_tol
        return is_collinear

    @staticmethod
    def is_point_between_points(point:Sequence[float], start_point:Sequence[float], end_point:Sequence[float]) -> bool:
        """
        Check if point is between the other two. Points must have the same dimension (2D or 3D).

        Args:
            - point (Sequence[float]): point coordinates to be tested
            - start_point (Sequence[float]): first extreme coordinates of the line
            - end_point (Sequence[float]): second extreme coordinates of the line

        Raises:
            - ValueError: when there is a dimension mismatch in the point dimensions.

        Returns:
            - bool: whether the point is between the other two or not
        """

        # check dimensions of points for validation
        Utils.check_dimensions([point, start_point, end_point])

        # Calculate vectors between the points
        vec_1 = np.asarray(point) - np.asarray(start_point)
        vec_2 = np.asarray(end_point) - np.asarray(start_point)

        # Calculate the scalar projections of vector1 onto vector2
        scalar_projection = sum(v1 * v2 for v1, v2 in zip(vec_1, vec_2)) / sum(v ** 2 for v in vec_2)

        # Check if the scalar projection is between 0 and 1 (inclusive)
        is_between: bool = 0 <= scalar_projection <= 1
        return is_between

    @staticmethod
    def is_non_str_sequence(seq: object) -> bool:
        """
        check whether object is a sequence but also not a string

        Returns:
            - bool: whether the sequence but also not a string
        """
        return isinstance(seq, Sequence) and not isinstance(seq, str)

    @staticmethod
    def chain_sequence(sequences: Sequence[Sequence[Any]]) -> Generator[Sequence[Any], Sequence[Any], None]:
        """
        Chains sequences together

        Args:
           - sequences (Sequence[Sequence[Any]]): sequences to chain

        Returns:
            - Generator[Sequence[Any], Sequence[Any], None]: generator for chaining sequences

        """
        for seq in sequences:
            yield from seq

    @staticmethod
    def merge(a: Dict[Any, Any], b: Dict[Any, Any], path: Union[List[str], Any] = None) -> Dict[Any, Any]:
        """
        merges dictionary b into dictionary a. if existing keywords conflict it assumes
        they are concatenated in a list

        Args:
            - a (Dict[Any,Any]): first dictionary
            - b (Dict[Any,Any]): second dictionary
            - path (List[str]): object to help navigate the deeper layers of the dictionary. \
                Initially this has to be None

        Returns:
            - a (Dict[Any,Any]): updated dictionary with the additional dictionary `b`
        """
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    Utils.merge(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass  # same leaf value
                elif any([not Utils.is_non_str_sequence(val) for val in (a[key], b[key])]):
                    # if none of them is a sequence and are found at the same key, then something went wrong.
                    # this should not be merged silently.
                    raise ValueError(f"Conflict of merging keys at {'->'.join(path + [str(key)])}. Two non sequence "
                                     f"values have been found.")
                else:
                    a[key] = list(Utils.chain_sequence([a[key], b[key]]))
            else:
                a[key] = b[key]
        return a

    @staticmethod
    def get_unique_objects(input_sequence: Sequence[Any]) -> List[Any]:
        """
        Get the unique objects, i.e., the objects that share the same memory location.

        Args:
            - input_sequence (Sequence[Any]): full list of possible duplicate objects

        Returns:
            - List[Any]: list of unique objects
        """
        return list({id(obj): obj for obj in input_sequence}.values())

    @staticmethod
    def is_line_edge_in_body(edge_element: 'Element', body_element: 'Element') -> bool:
        """
        Check if the edge element is a subset of the body element in the same order.

        Args:
            - edge_element (:class:`stem.mesh.Element`): 1D edge element
            - body_element (:class:`stem.mesh.Element`): 1D or 2D body element

        Raises:
            - ValueError: when the edge element is not a 1D element.
            - ValueError: when the body element is not a 1D or 2D element.
            - ValueError: when the element order of the edge element is different from the body element.
            - ValueError: when not all nodes of the edge element are part of the body element.

        Returns:
            - bool: True if the edge element is a subset of the body element in the same order, False otherwise.

        """

        # element info such as order, number of edges, element types etc.
        edge_el_info = Utils.get_element_info(edge_element.element_type)
        body_el_info = Utils.get_element_info(body_element.element_type)

        if edge_el_info["ndim"] != 1:
            raise ValueError("Edge element should be a 1D element.")
        if body_el_info["ndim"] != 1 and body_el_info["ndim"] != 2:
            raise ValueError("Body element should be a 1D or 2D element.")

        # if elements have different integration order, raise an error
        if edge_el_info["order"] != body_el_info["order"]:
            raise ValueError(
                f"Mismatch between edge element order ({edge_el_info['order']}) and body "
                f"element order ({body_el_info['order']})."
            )

        if not set(edge_element.node_ids).issubset(set(body_element.node_ids)):
            raise ValueError("All nodes of the edge element should be part of the body element.")


        # only vertices have to be checked, the rest of the nodes follows
        edge_vertices_ids = edge_element.node_ids[:edge_el_info["n_vertices"]]
        body_vertices_ids = body_element.node_ids[:body_el_info["n_vertices"]]

        # add first vertex to the end of the list, to make a closed loop for 2D elements
        if body_el_info["ndim"] == 2:
            body_vertices_ids.append(body_vertices_ids[0])

        n_edge_vertices = len(edge_vertices_ids)
        n_body_vertices = len(body_vertices_ids)

        # check if order of nodes in the edge element follows the body element.
        if n_body_vertices == n_edge_vertices:
            return body_vertices_ids == edge_vertices_ids
        elif n_body_vertices > n_edge_vertices:
            for ix in range(n_body_vertices - n_edge_vertices + 1):

                # check if edge vertices are part of body vertices in the same ordering
                if body_vertices_ids[ix:ix + n_edge_vertices] == edge_vertices_ids:
                    return True

        # if no match is found, return False
        return False

    @staticmethod
    def get_volume_line_edges(element):

        element_edges_dict = {"TETRAHEDRON_4N": [[0, 1],
                                                 [1, 2],
                                                 [2, 0],
                                                 [0, 3],
                                                 [1, 3],
                                                 [2, 3]],
                              "TETRAHEDRON_10N": [[0, 1, 4],
                                                  [1, 2, 5],
                                                  [2, 0, 6],
                                                  [0, 3, 7],
                                                  [1, 3, 8],
                                                  [2, 3, 9]],
                              "HEXAHEDRON_8N": [[0, 1],
                                                [1, 2],
                                                [2, 3],
                                                [3, 0],
                                                [4, 5],
                                                [5, 6],
                                                [6, 7],
                                                [7, 4],
                                                [0, 4],
                                                [1, 5],
                                                [2, 6],
                                                [3, 7]],
                                "HEXAHEDRON_20N": [[0, 1, 8],
                                                   [1, 2, 9],
                                                   [2, 3, 10],
                                                   [3, 0, 11],
                                                   [4, 5, 16],
                                                   [5, 6, 17],
                                                   [6, 7, 18],
                                                   [7, 4, 19],
                                                   [0, 4, 12],
                                                   [1, 5, 13],
                                                   [2, 6, 14],
                                                   [3, 7, 15]]}

        node_ids= np.array(element.node_ids)[element_edges_dict[element.element_type]]

        return node_ids





    @staticmethod
    def get_element_info(gmsh_element_type: str) -> Dict[str, Any]:
        """
        Returns the element info for a certain gmsh element type. The element info contains the number of dimensions,
        the order, the number of vertices and the reversed order of the connectivities.

        Args:
            - gmsh_element_type (str): gmsh element type

        Returns:
            - Dict[str, Any]: element info
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
                                                  "reversed_order": [1, 0, 3, 2]},
                                "QUADRANGLE_8N": {"ndim": 2,
                                                  "order": 2,
                                                  "n_vertices": 4,
                                                  "reversed_order": [1, 0, 3, 2, 4, 7, 6, 5]},
                                "TETRAHEDRON_4N": {"ndim": 3,
                                                   "order": 1,
                                                   "n_vertices": 4,
                                                   "reversed_order": [1, 0, 2, 3]},
                                "TETRAHEDRON_10N": {"ndim": 3,
                                                    "order": 2,
                                                    "n_vertices": 4,
                                                    "reversed_order": [1, 0, 2, 3, 4, 6, 5, 9, 8, 7]},
                                "HEXAHEDRON_8N": {"ndim": 3,
                                                  "order": 1,
                                                  "n_vertices": 8,
                                                  "reversed_order": [1, 0, 3, 2, 5, 4, 7, 6]},
                                "HEXAHEDRON_20N": {"ndim": 3,
                                                   "order": 2,
                                                   "n_vertices": 8,
                                                   "reversed_order": [1, 0, 3, 2,
                                                                      5, 4, 7, 6,
                                                                      8, 9, 10, 11,
                                                                      12, 13, 14, 15,
                                                                      16, 17, 18, 19]},
                                }

        # find element order
        if gmsh_element_type not in element_mapping_dict.keys():
            raise NotImplementedError(f"No reversed order defined for the element type: {gmsh_element_type}")

        return element_mapping_dict[gmsh_element_type]

    @staticmethod
    def flip_node_order(element_info: Dict[str, Any], elements: Sequence['Element']):
        """
        Flips the node order of the elements

        Args:
            - element_info (Dict[str, Any]): element info
            - elements (List[:class:`stem.mesh.Element`]): list of elements

        """

        # retrieve element ids and connectivities
        ids = [element.id for element in elements]
        element_connectivies = np.array([element.node_ids for element in elements])

        # flip the elements connectivities
        element_connectivies = element_connectivies[:, element_info["reversed_order"]]

        # update the elements connectivities
        for i, (id, element_connectivity) in enumerate(zip(ids, element_connectivies)):
            elements[i].node_ids = list(element_connectivity)

    @staticmethod
    def is_volume_edge_defined_outwards(edge_element: 'Element', body_element: 'Element',
                                        nodes: Dict[int, Sequence[float]]) -> Optional[bool]:
        """
        Checks if the normal vector of the edge element is pointing outwards of the body element.

        Args:
            - edge_element (:class:`stem.mesh.Element`): 2D edge surface element
            - body_element (:class:`stem.mesh.Element`): 3D body volume element
            - nodes (Dict[int, Sequence[float]]): dictionary of node ids and coordinates

        Raises:
            - ValueError: when the edge element is not a 2D element.
            - ValueError: when the body element is not a 3D element.
            - ValueError: when not all nodes of the edge element are part of the body element.

        Returns:
            - Optional[bool]: True if the normal vector of the edge element is pointing outwards of the body element,
                False otherwise.

        """

        # element info such as order, number of edges, element types etc.
        edge_el_info = Utils.get_element_info(edge_element.element_type)
        body_el_info = Utils.get_element_info(body_element.element_type)

        if edge_el_info["ndim"] != 2:
            raise ValueError("Edge element should be a 2D element.")

        if body_el_info["ndim"] != 3:
            raise ValueError("Body element should be a 3D element.")

        if not set(edge_element.node_ids).issubset(set(body_element.node_ids)):
            raise ValueError("All nodes of the edge element should be part of the body element.")

        # calculate normal vector of edge element
        coordinates_edge = np.array([nodes[node_id] for node_id in edge_element.node_ids[:edge_el_info["n_vertices"]]])

        normal_vector_edge = np.cross(coordinates_edge[1, :] - coordinates_edge[0, :],
                                      coordinates_edge[2, :] - coordinates_edge[0, :])

        # calculate centroid of neighbouring body element
        body_vertices_ids = body_element.node_ids[:body_el_info["n_vertices"]]
        coordinates_body_element = np.array([nodes[node_id] for node_id in body_vertices_ids])
        centroid_volume = np.mean(coordinates_body_element, axis=0)

        # calculate centroid of edge element
        centroid_edge = np.mean(coordinates_edge, axis=0)

        # calculate vector inwards of body element
        body_inward_vector = centroid_volume - centroid_edge

        # check if normal vector of edge element is pointing outwards of body element
        is_outwards: bool = np.dot(normal_vector_edge, body_inward_vector) < 0

        return is_outwards

