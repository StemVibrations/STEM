from pathlib import Path
from typing import Sequence, Dict, Any, List, Union, Optional, Generator, TYPE_CHECKING

import numpy as np
import numpy.typing as npty
from scipy.spatial import cKDTree

from stem.globals import ELEMENT_DATA

if TYPE_CHECKING:
    from stem.mesh import Element, Mesh
    from stem.geometry import Geometry

NUMBER_TYPES = (int, float, np.int64, np.float64)
"""
TypeAlias:
    - NUMBER_TYPES: Tuple[int, float, np.int64, np.float64]
"""


class Utils:
    """
    Class containing utility methods.

    """

    @staticmethod
    def check_ndim_nnodes_combinations(n_dim: int, n_nodes_element: Optional[int],
                                       available_combinations: Dict[int, List[Any]], class_name: str):
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
                f"supported. {n_nodes_element} nodes were provided.")

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
    def check_dimensions(points: Sequence[Sequence[float]]) -> None:
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
    def is_collinear(point: Sequence[float],
                     start_point: Sequence[float],
                     end_point: Sequence[float],
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
    def is_point_between_points(point: Sequence[float], start_point: Sequence[float], end_point: Sequence[float]) \
            -> bool:
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
        scalar_projection = sum(v1 * v2 for v1, v2 in zip(vec_1, vec_2)) / sum(v**2 for v in vec_2)

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
    def get_element_edges(element: 'Element') -> npty.NDArray[np.int64]:
        """
        Gets the nodal connectivities of the line edges of elements

        Args:
            - element (:class:`stem.mesh.Element`): element object

        Returns:
            - npty.NDArray[np.int64]: nodal connectivities of the line edges of the element

        """

        # get nodal connectivities of the line edges from the local element edges dictionary
        node_ids: npty.NDArray[np.int64] = np.array(element.node_ids,
                                                    dtype=int)[ELEMENT_DATA[element.element_type]["edges"]]

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
                                        nodes: Dict[int, Sequence[float]]) -> bool:
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
            - bool: True if the normal vector of the edge element is pointing outwards of the body element,
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
    def create_sigmoid_tiny_expr(start_time: float, dt_slope: float, initial_value: float, final_value: float,
                                 is_half_function: bool) -> str:
        """
        Creates a tiny expression with variable time for a sigmoid function. For more information on tiny expressions,
        see: https://github.com/codeplea/tinyexpr

        Args:
            - start_time (float): start time of the sigmoid function
            - dt_slope (float): delta time on where the slope is present in the sigmoid function
            - initial_value (float): initial value of the sigmoid function
            - final_value (float): final value of the sigmoid function
            - is_half_function (bool): whether to return half the sigmoid function or the full sigmoid function

        Returns:
            - str: tiny expression of the sigmoid time function
        """

        # only return half the sigmoid function, where the slope always contains the same sign
        if is_half_function:

            # calculate beta
            beta = 6 / dt_slope

            return (f"((1 / (1 + e^(-{beta} * (t -  {start_time}))) - 0.5)) * "
                    f"({final_value} - {initial_value}) * 2 + {initial_value}")

        # return full sigmoid function
        else:
            # calculate beta
            beta = 12 / dt_slope
            return (f"(1 / (1 + e^(-{beta} * (t - {dt_slope} / 2 - {start_time})))) * ({final_value} - {initial_value})"
                    f"+ {initial_value}")

    @staticmethod
    def create_box_tiny_expr(transition_parameter: float,
                             start_peak: float,
                             end_peak: float,
                             peak_value: float,
                             base_value: float,
                             variable: str = "x") -> str:
        """
        Creates a tiny expression for a hyperbolic approximation of the box function. For more information on tiny
        expressions, see: https://github.com/codeplea/tinyexpr

        Args:
            - transition_parameter (float): parameter to control the transition of the box function, \
              the higher the value, the steeper the transition
            - start_peak (float): start of the peak of the box function
            - end_peak (float): end of the peak of the box function
            - peak_value (float): value of the peak of the box function
            - base_value (float): value of the base of the box function
            - variable (str): variable within the box function tinyexpr, default is "x", other options are "y", "z", "t"

        Raises:
            - ValueError: when start peak is larger or equal to end peak
            - ValueError: when variable is not "x", "y", "z" or "t"

        Returns:
            - str: tiny expression of the box function

        """

        if start_peak >= end_peak:
            raise ValueError("Start peak should be smaller than end peak.")

        if variable not in ["x", "y", "z", "t"]:
            raise ValueError("Variable should be either 'x', 'y', 'z' or 't'.")

        length_peak = end_peak - start_peak
        centre_peak = (start_peak + end_peak) / 2

        tiny_expr = (
            f"(1 / 2 + 1 / 2 * tanh({transition_parameter} * ({variable} - ({centre_peak - length_peak / 2}) ))"
            f" - (1 / 2 + 1 / 2 * tanh({transition_parameter} * ({variable} - ({centre_peak + length_peak / 2}) ))))"
            f" * ({peak_value - base_value}) + {base_value}")

        return tiny_expr

    @staticmethod
    def check_lines_geometry_are_path(geometry: Optional['Geometry']) -> bool:
        """

        Checks if lines are connected forming a path without:
          a) disconnected lines,
          b) branching out paths::

              a) Disconnected lines:
                  o---o
                  |
                  o

              b) Branching out paths:
                  o---o
                       |
                  o----o----o
                       |
                       o

        Args:
            - geometry (:class:`stem.geometry.Geometry`): geometry to be checked.

        Raises:
            - ValueError: when geometry is not provided (is None).
            - ValueError: when geometry has no lines.

        Returns:
            - bool: whether the lines are connected along the path

        """

        if geometry is None:
            raise ValueError("No geometry has been provided.")

        if geometry.lines is None or len(geometry.lines) == 0:
            raise ValueError("The geometry doesn't contain lines to check.")

        # if 2 or more lines check for branching points/loops and discontinuities
        if len(geometry.lines) > 1:

            # find which lines are connected to which point
            lines_to_point: Dict[int, List[int]] = {point_id: [] for point_id in geometry.points.keys()}
            for line_id, line in geometry.lines.items():
                for point_id in line.point_ids:
                    lines_to_point[point_id].append(line_id)

            # check if the lines are connected without branches
            for line_ids in lines_to_point.values():

                # if more than 2 lines are connected to the point a branching point or loop is found
                if len(line_ids) > 2:
                    return False

            # if no branching point are found than the check of connectivity holds when
            if len(lines_to_point) != (len(geometry.lines) + 1):
                # lines are not connected.
                return False

        # All lines are connected without branches
        return True

    @staticmethod
    def is_point_aligned_and_between_any_of_points(coordinates: Sequence[Sequence[Sequence[float]]],
                                                   origin: Sequence[float]) -> bool:
        """
        Checks that the point (origin) provided aligns with at least one of the lines, expressed as
        list of pairs of coordinates representing the edges of the line.

        Args:
            - coordinates (Sequence[Sequence[Sequence[float]]]): Pair-wise sets of coordinates representing the line \
                on which the origin should lie.
            - origin (Sequence[float]): the coordinates of the point to be checked for alignment.

        Returns:
            - bool: whether the considered point in at least one of the given lines (i.e. within the sequence of \
                pair-wise points).

        """

        for ix in range(len(coordinates)):
            # check origin is collinear to the edges of the line
            collinear_check = Utils.is_collinear(point=origin,
                                                 start_point=coordinates[ix][0],
                                                 end_point=coordinates[ix][1])
            # check origin is between the edges of the line (edges included)
            is_between_check = Utils.is_point_between_points(point=origin,
                                                             start_point=coordinates[ix][0],
                                                             end_point=coordinates[ix][1])
            # check if point complies
            is_on_line = collinear_check and is_between_check
            # exit at the first success of the test (point in the line) and return True
            if is_on_line:
                return True

        # none of the lines contain the origin, return False
        return False

    @staticmethod
    def replace_extensions(filename: str, new_extension: str) -> str:
        """
        Adjust the extension of a file. Can remove multiple extensions (e.g. .tar.gz.tmp) with a new extension (e.g.
        json). If no extensions are given, the new extension is added directly.

        Args:
            - filename (str): name or path to the filename for which the extension needs to be changed
            - new_extension (str): the new extension for the file

        Returns:
            - str: name or path to the filename with the desired extension

        """
        path_obj = Path(filename)
        extensions = "".join(path_obj.suffixes)
        if len(extensions) == 0:
            return str(path_obj.with_suffix(new_extension))
        else:
            return str(path_obj).replace(extensions, new_extension)

    @staticmethod
    def find_node_ids_close_to_geometry_nodes(mesh: 'Mesh', geometry: 'Geometry', eps: float = 1e-6) \
            -> npty.NDArray[np.uint64]:
        """
        Searches the nodes in the mesh close to the point of a given geometry.

        Args:
            - mesh (:class:`stem.mesh.Mesh`): mesh object for which the node ids are required to be computed.
            - geometry (:class:`stem.geometry.Geometry`): geometry containing the points of interest.
            - eps (float): tolerance for searching close nodes.

        Returns:
            - npty.NDArray[np.uint64]: list of ids of the nodes close to the geometry points

        """
        # retrieve ids and coordinates of the nodes
        node_ids = list(mesh.nodes.keys())
        coordinates = np.stack([node.coordinates for node in mesh.nodes.values()])

        # compute pairwise distances between the geometry nodes (actual outputs and subset of the mesh nodes) and the
        # mesh nodes
        output_coordinates = np.stack([np.array(point.coordinates) for point in geometry.points.values()], dtype=float)

        # set up the tree for fast querying
        tree = cKDTree(coordinates)

        # find the ids of the nodes in the model that are close to the specified coordinates.
        _, close_indices = tree.query(output_coordinates, k=1, distance_upper_bound=eps)

        close_node_ids: npty.NDArray[np.uint64] = np.array(node_ids, dtype=np.uint64)[close_indices]

        return close_node_ids

    @staticmethod
    def find_first_three_non_collinear_points(points: Sequence[Sequence[float]],
                                              a_tol: float = 1e-06) -> Optional[Sequence[Sequence[float]]]:
        """
        Find the first 3 non-collinear points in sequence of points. If all are collinear, the function returns `None`.

        Args:
            - points (Sequence[Sequence[float]]): points from which the non-collinear points should be searched for.
            - a_tol (float): absolute tolerance to check collinearity (default 1e-6)

        Raises:
            -  ValueError: if less than three points are provided.

        Returns:
            - Optional[List[Sequence[float]]]: list of the first three points that are not collinear. If all are
            collinear, None is returned.

        """
        if len(points) < 3:
            raise ValueError("Less than 3 points are provided.")

        # select the first 2 points in the sequence
        p1 = points[0]
        p2 = points[1]

        for p_candidate in points[2:]:
            # the first point that is not collinear with the first 2, is returned altogether with p1 and p2
            if not Utils.is_collinear(p_candidate, p1, p2, a_tol=a_tol):
                return [p1, p2, p_candidate]
        # all are collinear, None is returned
        return None

    @staticmethod
    def is_point_coplanar_to_polygon(point: Sequence[float],
                                     polygon_points: Sequence[Sequence[float]],
                                     a_tol: float = 1e-06) -> bool:
        """
        Checks whether a point is coplanar to a list of points defining a polygon

        Args:
            - point (Sequence[float]): point to be checked.
            - polygon_points (Sequence[Sequence[float]]): points belonging to the polygon.
            - a_tol (float): absolute tolerance to check planarity (default 1e-6)

        Raises:
            -  ValueError: if the polygon itself is not planar.
            -  ValueError: if all the points in the polygon are collinear.

        Returns:
            - bool: whether the point is coplanar with the polygon.

        """

        # check that polygon is coplanar
        if not Utils.is_polygon_planar(polygon_points=polygon_points, a_tol=a_tol):
            raise ValueError("Points in the polygon are not co-planar.")

        # Choose three non-collinear points from the polygon

        non_collinear_points = Utils.find_first_three_non_collinear_points(points=polygon_points, a_tol=a_tol)

        if non_collinear_points is None:
            raise ValueError("All the points in the polygon are collinear.")

        # Convert points to a NumPy array for easier manipulation
        p1, p2, p3 = np.array(non_collinear_points)

        # Calculate vectors from p1 to p2 and p1 to p3
        v1 = p2 - p1
        v2 = p3 - p1

        # Calculate the normal vector of the plane formed by v1 and v2
        normal = np.cross(v1, v2)

        # Transform point in numpy array
        point_array = np.array(point)

        # Calculate the vector from p1 to the current point
        vector_to_point = point_array - p1

        # Calculate the dot product of normal and vector_to_point
        dot_product = np.dot(normal, vector_to_point)

        # If the dot product is not close to 0 (within a small tolerance),
        # the points are not coplanar
        if not np.isclose(dot_product, 0, atol=a_tol):
            return False
        return True

    @staticmethod
    def is_polygon_planar(polygon_points: Sequence[Sequence[float]], a_tol: float = 1e-06) -> bool:
        """
        Checks whether a polygon is planar, i.e. all its point lie on the same plane.

        Args:
            - polygon_points (Sequence[Sequence[float]]): points belonging to the polygon.
            - a_tol (float): absolute tolerance to check planarity (default 1e-6)

        Raises:
            -  ValueError: if less than three points are provided.
            -  ValueError: if all the points in the polygon are collinear.

        Returns:
            - bool: whether the polygon is planar.

        """
        if len(polygon_points) < 3:
            raise ValueError("Less than 3 points are given, the shape is not a polygon.")

        # get the first 3 non-collinear points in polygon
        non_collinear_points = Utils.find_first_three_non_collinear_points(points=polygon_points, a_tol=a_tol)

        if non_collinear_points is None:
            raise ValueError("All the points in the polygon are collinear.")

        # 3 non-collinear points form always a unique plane
        if len(polygon_points) == 3:
            return True

        # Convert points to a NumPy array for easier manipulation
        non_collinear_points_array = np.array(non_collinear_points)

        # Choose the first three non-collinear points
        p1, p2, p3 = non_collinear_points_array[:3]

        # Calculate vectors from p1 to p2 and p1 to p3
        v1 = p2 - p1
        v2 = p3 - p1

        # Calculate the normal vector of the plane formed by v1 and v2
        normal = np.cross(v1, v2)

        # Check if all other points lie on the plane
        for point in polygon_points:

            point_array = np.array(point)
            # Calculate the vector from p1 to the current point
            vector_to_point = point_array - p1

            # Calculate the dot product of normal and vector_to_point
            dot_product = np.dot(normal, vector_to_point)

            # If the dot product is not close to 0 (within a small tolerance),
            # the points are not coplanar
            if not np.isclose(dot_product, 0, atol=a_tol):
                return False

        # If the dot product is 0, the point is on the plane
        return True

    @staticmethod
    def validate_coordinates(coordinates: Union[Sequence[Sequence[float]], npty.NDArray[np.float64]]):
        """
        Validates the coordinates in input.

        Args:
            - coordinates (Sequence[Sequence[float]]): The coordinates of the load.

        Raises:
            - ValueError: if coordinates is not a sequence real numbers.
            - ValueError: if coordinates is not convertible to a 2D array (i.e. a sequence of sequences)
            - ValueError: if the number of elements (number of coordinates) is not 3.

        """

        # if is not an array, make it array!
        if not isinstance(coordinates, np.ndarray):
            coordinates = np.array(coordinates, dtype=np.float64)

        if len(coordinates.shape) != 2:
            raise ValueError("Coordinates are not a sequence of a sequence or a 2D array.")

        if coordinates.shape[1] != 3:
            raise ValueError(f"Coordinates should be 3D but {coordinates.shape[1]} coordinates were given.")

        # check if coordinates are real numbers
        for coordinate in coordinates:
            for i in coordinate:
                if not isinstance(i, NUMBER_TYPES) or np.isnan(i) or np.isinf(i):
                    raise ValueError(f"Coordinates should be a sequence of sequence of real numbers, "
                                     f"but {i} was given.")
