from typing import Sequence

import numpy as np


def is_collinear(point:Sequence[float], start_point:Sequence[float], end_point:Sequence[float]):
    """
    Check if point is aligned with the other two on a line. Points must have the same dimension (2D or 3D)

    Args:
        point (Sequence[float]): point to be tested
        start_point (Sequence[float]): first point on the line
        end_point (Sequence[float]): second point on the line

    Returns:
        bool: whether the point is aligned or not
    """

    vec_1 = np.asarray(point) - np.asarray(start_point)
    vec_2 = np.asarray(end_point) - np.asarray(start_point)

    cross_product = np.cross(vec_1, vec_2)
    return np.sum(np.abs(cross_product)) < 1e-06


def is_point_between_points(point:Sequence[float], start_point:Sequence[float], end_point:Sequence[float]):
    """
    Check if point is between the other two. Points must have the same dimension (2D or 3D).

    Args:
        point (Sequence[float]): point to be tested
        start_point (Sequence[float]): first extreme on the line
        end_point (Sequence[float]): second extreme on the line

    Returns:
        bool: whether the point is between the other two or not
    """

    # Calculate vectors between the points
    vec_1 = np.asarray(point) - np.asarray(start_point)
    vec_2 = np.asarray(end_point) - np.asarray(start_point)

    # Calculate the scalar projections of vector1 onto vector2
    scalar_projection = sum(v1 * v2 for v1, v2 in zip(vec_1, vec_2)) / sum(v ** 2 for v in vec_2)

    # Check if the scalar projection is between 0 and 1 (inclusive)
    return 0 <= scalar_projection <= 1