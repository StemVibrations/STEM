from typing import Sequence, List

import numpy as np


def check_dimensions(points:Sequence[Sequence[float]]):
    """

    Check if points have the same dimensions (2D or 3D).

    Args:
        - points: (Sequence[Sequence[float]]): points to be tested

    Raises:
        - ValueError: when the points have different dimensions.
        - ValueError: when the dimension is not either 2 or 3D.
    """

    lengths = [len(point) for point in points]
    if len(np.unique(lengths)) != 1:
        raise ValueError("Mismatch in dimension of given points!")

    if any([ll not in [2,3] for ll in lengths]):
        raise ValueError("Dimension of the points should be 2D or 3D.")


def is_collinear(point:Sequence[float], start_point:Sequence[float], end_point:Sequence[float], a_tol:float=1e-06):
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
    check_dimensions([point, start_point, end_point])

    vec_1 = np.asarray(point) - np.asarray(start_point)
    vec_2 = np.asarray(end_point) - np.asarray(start_point)

    # cross product of the two vector
    cross_product = np.cross(vec_1, vec_2)
    # It should be smaller than tolerance for points to be aligned
    return np.sum(np.abs(cross_product)) < a_tol


def is_point_between_points(point:Sequence[float], start_point:Sequence[float], end_point:Sequence[float]):
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
    check_dimensions([point, start_point, end_point])

    # Calculate vectors between the points
    vec_1 = np.asarray(point) - np.asarray(start_point)
    vec_2 = np.asarray(end_point) - np.asarray(start_point)

    # Calculate the scalar projections of vector1 onto vector2
    scalar_projection = sum(v1 * v2 for v1, v2 in zip(vec_1, vec_2)) / sum(v ** 2 for v in vec_2)

    # Check if the scalar projection is between 0 and 1 (inclusive)
    return 0 <= scalar_projection <= 1


def is_non_string_sequence(obj:object):
    """
    Check if object is a sequence but not a string

    Args:
        obj (object): object to be tested

    Returns:
        bool: whether the object is a sequence but not a string
    """

    if isinstance(obj, str):
        return False
    return isinstance(obj, Sequence)