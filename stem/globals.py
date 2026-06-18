"""
This module contains global variables used in the stem package.
"""

from typing import Dict, Any
from dataclasses import dataclass

VERTICAL_AXIS = 1  # [0, 1, 2] = [x, y, z]
OUT_OF_PLANE_AXIS_2D = 2  # [0, 1, 2] = [x, y, z]


# mutable global variables by users
@dataclass
class GlobalSettings:
    """
    Class containing global settings for the stem package. These settings can be modified by the user to adjust
    the behavior of the package.

    Attributes:
        - gravity_value (float): The gravitational acceleration value in m/s². Default is -9.81 m/s².
        - time_step_precision (float): The precision for time step calculations in seconds. Default is 1e-08 s.
        - geometry_precision (float): The precision for geometry calculations in meters. Default is 1e-08 m.
    """

    gravity_value: float = -9.81  # [m/s2]
    time_step_precision: float = 1e-08  # s
    geometry_precision: float = 1e-08  # m


# yapf: disable
#: Element data for supported element types in STEM. The data contains the following information: \
#: - ndim (int): number of dimensions
#: - element_order (int): element order
#: - n_vertices (int): number of vertices
#: - reversed_connectivity_order (List[int]): reversed connectivity order of the element
#: - n_nodes_facets (int): number of nodes on the facets (edges for 1D and 2D elements, faces for 3D elements)
#: - edges (List[List[int]]): edges of the element, line edges of each element
#: - facets (List[List[int]]): facets of the element, faces for 3D elements, line edges for 2D and 1D elements
#: - gmsh_to_kratos_order (List[int]): mapping from Gmsh node ordering to Kratos node ordering (only if required)
ELEMENT_DATA: Dict[str, Any] = {"POINT_1N": {"ndim": 0,
                                             "element_order": 1,
                                             "n_vertices": 1,
                                             "reversed_connectivity_order": [0],
                                             "n_nodes_facets": 0,
                                             "edges": [],
                                             "facets": []},
                                "LINE_2N": {"ndim": 1,
                                            "element_order": 1,
                                            "n_vertices": 2,
                                            "reversed_connectivity_order": [1, 0],
                                            "n_nodes_facets": 2,
                                            "edges": [[0, 1]],
                                            "facets": [[0, 1]]},
                                "LINE_3N": {"ndim": 1,
                                            "element_order": 2,
                                            "n_vertices": 2,
                                            "reversed_connectivity_order": [1, 0, 2],
                                            "n_nodes_facets": 3,
                                            "edges": [[0, 1, 2]],
                                            "facets": [[0, 1, 2]]},
                                "TRIANGLE_3N": {"ndim": 2,
                                                "element_order": 1,
                                                "n_vertices": 3,
                                                "reversed_connectivity_order": [2, 1, 0],
                                                "n_nodes_facets": 2,
                                                "edges": [[1, 2], [2, 0], [0, 1]],
                                                "facets": [[1, 2], [2, 0], [0, 1]]},
                                "TRIANGLE_6N": {"ndim": 2,
                                                "element_order": 2,
                                                "n_vertices": 3,
                                                "reversed_connectivity_order": [2, 1, 0, 4, 3, 5],
                                                "n_nodes_facets": 3,
                                                "edges": [[1, 2, 3], [1, 2, 4], [2, 0, 5]],
                                                "facets": [[1, 2, 3], [1, 2, 4], [2, 0, 5]]},
                                "QUADRANGLE_4N": {"ndim": 2,
                                                  "element_order": 1,
                                                  "n_vertices": 4,
                                                  "reversed_connectivity_order": [1, 0, 3, 2],
                                                  "n_nodes_facets": 2,
                                                  "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                                                  "facets": [[0, 1], [1, 2], [2, 3], [3, 0]]},
                                "QUADRANGLE_8N": {"ndim": 2,
                                                  "element_order": 2,
                                                  "n_vertices": 4,
                                                  "reversed_connectivity_order": [1, 0, 3, 2, 4, 7, 6, 5],
                                                  "n_nodes_facets": 3,
                                                  "edges": [[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]],
                                                  "facets": [[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]]},
                                "TETRAHEDRON_4N": {"ndim": 3,
                                                   "element_order": 1,
                                                   "n_vertices": 4,
                                                   "reversed_connectivity_order": [1, 0, 2, 3],
                                                   "n_nodes_facets": 3,
                                                   "edges": [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
                                                   "facets": [[0, 2, 1], [0, 3, 2], [0, 1, 3], [2, 3, 1]]},
                                "TETRAHEDRON_10N": {"ndim": 3,
                                                    "element_order": 2,
                                                    "n_vertices": 4,
                                                    "reversed_connectivity_order": [1, 0, 2, 3, 4, 6, 5, 8, 7, 9],
                                                    "n_nodes_facets": 6,
                                                    "edges": [[0, 1, 4], [1, 2, 5], [2, 0, 6],
                                                              [0, 3, 7], [1, 3, 8], [2, 3, 9]],
                                                    "facets": [[0, 2, 1, 6, 5, 4], [0, 3, 2, 7, 9, 6],
                                                               [0, 1, 3, 4, 8, 7], [2, 3, 1, 9, 8, 5]],
                                                    "gmsh_to_kratos_order": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]},
                                "HEXAHEDRON_8N": {"ndim": 3,
                                                  "element_order": 1,
                                                  "n_vertices": 8,
                                                  "reversed_connectivity_order": [1, 0, 3, 2, 5, 4, 7, 6],
                                                  "n_nodes_facets": 4,
                                                  "edges": [[0, 1], [1, 2], [2, 3], [3, 0],
                                                            [4, 5], [5, 6], [6, 7], [7, 4],
                                                            [0, 4], [1, 5], [2, 6], [3, 7]],
                                                  "facets": [[3, 2, 1, 0], [0, 1, 5, 4], [2, 6, 5, 1],
                                                             [7, 6, 2, 3], [7, 3, 0, 4], [4, 5, 6, 7]]},
                                "HEXAHEDRON_20N": {"ndim": 3,
                                                   "element_order": 2,
                                                   "n_vertices": 8,
                                                   "reversed_connectivity_order": [1, 0, 3, 2,
                                                                                   5, 4, 7, 6,
                                                                                   8, 11, 10, 9,
                                                                                   13, 12, 15, 14,
                                                                                   16, 19, 18, 17],
                                                   "n_nodes_facets": 8,
                                                   "edges": [[0, 1, 8], [1, 2, 9], [2, 3, 10], [3, 0, 11],

                                                             [4, 5, 16], [5, 6, 17], [6, 7, 18], [7, 4, 19],
                                                             [0, 4, 12], [1, 5, 13], [2, 6, 14], [3, 7, 15]],
                                                   "facets": [[3, 2, 1, 0, 10, 9, 8, 11], [0, 1, 5, 4, 8, 13, 16, 12],
                                                              [2, 6, 5, 1, 14, 17, 13, 9],
                                                              [7, 6, 2, 3, 18, 14, 10, 15],
                                                              [7, 3, 0, 4, 15, 11, 12, 19],
                                                              [4, 5, 6, 7, 16, 17, 18, 19]],
                                                   "gmsh_to_kratos_order": [0, 1, 2, 3,
                                                                            4, 5, 6, 7,
                                                                            8, 11, 13, 9,
                                                                            10, 12, 14, 15,
                                                                            16, 18, 19, 17]},
                                }

# yapf: enable
