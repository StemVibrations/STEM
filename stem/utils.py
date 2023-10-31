from typing import Sequence, Dict, Any, List, Union, Optional, Generator, TYPE_CHECKING

import networkx as nx
import numpy as np
import numpy.typing as npt

from stem.globals import ELEMENT_DATA

if TYPE_CHECKING:
    from stem.mesh import Element
    from stem.geometry import Geometry


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
    def get_element_edges(element: 'Element') -> npt.NDArray[np.int64]:
        """
        Gets the nodal connectivities of the line edges of elements

        Args:
            - element (:class:`stem.mesh.Element`): element object

        Returns:
            - npt.NDArray[np.int64]: nodal connectivities of the line edges of the element

        """

        # get nodal connectivities of the line edges from the local element edges dictionary
        node_ids: npt.NDArray[np.int64] = np.array(element.node_ids, dtype=int)[
            ELEMENT_DATA[element.element_type]["edges"]]

        return node_ids

    @staticmethod
    def flip_node_order(elements: Sequence['Element']):
        """
        Flips the node order of the elements, where all elements should be of the same type.

        Args:
            - elements (List[:class:`stem.mesh.Element`]): list of elements

        Raises:
            - ValueError: when the elements are not of the same type.

        """

        # return of no elements are provided
        if len(elements) == 0:
            return

        # check if all elements are of the same type and get the element type
        element_types = set([element.element_type for element in elements])
        if len(element_types) > 1:
            raise ValueError("All elements should be of the same type.")
        element_type = list(element_types)[0]

        # retrieve element ids and connectivities
        ids = [element.id for element in elements]
        element_connectivies = np.array([element.node_ids for element in elements])

        # flip the elements connectivities
        element_connectivies = element_connectivies[:, ELEMENT_DATA[element_type]["reversed_order"]]

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
        edge_el_info = ELEMENT_DATA[edge_element.element_type]
        body_el_info = ELEMENT_DATA[body_element.element_type]

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

    @staticmethod
    def check_lines_geometry_are_path(geometry: Optional['Geometry']) -> None:

        """Checks if lines are connected forming a path without:

            a) disconnected lines,   b) branching out paths
                o---o       o---o              o
                |                              |
                o                         o----o----o
                                               |
                                               o

        Args:
            - geometry: geometry to be checked

        Raises:
            - ValueError: when geometry is not provided (is None).
            - ValueError: when geometry has no lines.
            - ValueError: when lines are disconnected.
            - ValueError: when lines are branching off.
            - ValueError: when lines are creating loops (passing on the same node twice).

        """

        if geometry is None:
            raise ValueError("No geometry has been provided.")

        if geometry.lines is None or len(geometry.lines) == 0:
            raise ValueError("The geometry doesn't contain lines to check.")

        if len(geometry.lines) > 1:

            lines = {_id: line.point_ids for _id, line in geometry.lines.items()}
            points = list(set([n for v in lines.values() for n in v]))

            graph = nx.Graph()

            for n in points:
                graph.add_node(n)

            for l1, nodes1 in lines.items():
                graph.add_edge(nodes1[0], nodes1[1])

            # Run DBSCAN clustering on the graph,
            # Count the number of clusters and identify branching points and loops
            clustered = list(nx.connected_components(graph))
            branching_points = [node for node, degree in graph.degree() if degree > 2]
            loops = list(nx.simple_cycles(graph))

            num_clusters = len(clustered)
            num_loops = len(loops)
            num_bp = len(branching_points)

            if num_clusters > 1:
                raise ValueError(f"Number of disconnected paths is >1: {num_clusters-1} discontinuities found in "
                                 f"the path!")

            if num_loops > 0:
                raise ValueError(f"Found {num_loops} loop(s) in the path.")

            if num_bp > 0:
                raise ValueError(f"Path is branching, should be on a line."
                                 f"{num_bp} branching point(s) have been found in the path!")

    @staticmethod
    def is_point_aligned_and_between_any_of_points(coordinates: Sequence[Sequence[Sequence[float]]],
                                                   origin: Sequence[float]):
        """Checks that any of the points provides in a list of pairs of coordinates

        Args:
            - coordinates (Sequence[Sequence[Sequence[float]]]): Pair-wise sets of coordinates representing the line
                on which the origin should lie.
            - origin (Sequence[float]): the coordinates of the point to be checked for alignment.

        Raises:
            - ValueError: when point is not aligned with one of the lines (pair-wise sets of coordinates).

        """

        for ix in range(len(coordinates)):
            # check origin is collinear to the edges of the line
            collinear_check = Utils.is_collinear(
                point=origin, start_point=coordinates[ix][0], end_point=coordinates[ix][1]
            )
            # check origin is between the edges of the line (edges included)
            is_between_check = Utils.is_point_between_points(
                point=origin, start_point=coordinates[ix][0], end_point=coordinates[ix][1]
            )
            # check if point complies
            is_on_line = collinear_check and is_between_check
            # exit at the first success of the test (point in the line)
            if is_on_line:
                return

        # none of the lines contain the origin, then raise an error
        raise ValueError(f"Origin is not in any of the lines given as trajectory of the moving load.")
