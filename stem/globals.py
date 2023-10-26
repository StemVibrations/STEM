"""
This module contains global variables used in the stem package.
"""

from typing import Dict,  Any

GRAVITY_VALUE = -9.81  # [m/s2]
VERTICAL_AXIS = 1  # [0, 1, 2] = [x, y, z]
OUT_OF_PLANE_AXIS_2D = 2  # [0, 1, 2] = [x, y, z]

#: Element data for supported element types in STEM. The data contains the following information: \
#: - ndim (int): number of dimensions
#: - order (int): element order
#: - n_vertices (int): number of vertices
#: - reversed_order (List[int]): reversed connectivity order of the element
#: - edges (List[List[int]]): edges of the element, line edges of each element
ELEMENT_DATA: Dict[str, Any] = {"POINT_1N": {"ndim": 0,
                                             "order": 1,
                                             "n_vertices": 1,
                                             "reversed_order": [0],
                                             "edges": []},
                                "LINE_2N": {"ndim": 1,
                                            "order": 1,
                                            "n_vertices": 2,
                                            "reversed_order": [1, 0],
                                            "edges": [[0, 1]]},
                                "LINE_3N": {"ndim": 1,
                                            "order": 2,
                                            "n_vertices": 2,
                                            "reversed_order": [1, 0, 2],
                                            "edges": [[0, 1, 2]]},
                                "TRIANGLE_3N": {"ndim": 2,
                                                "order": 1,
                                                "n_vertices": 3,
                                                "reversed_order": [2, 1, 0],
                                                "edges": [[1, 2], [2, 0], [0, 1]]},
                                "TRIANGLE_6N": {"ndim": 2,
                                                "order": 2,
                                                "n_vertices": 3,
                                                "reversed_order": [2, 1, 0, 5, 4, 3],
                                                "edges": [[1, 2, 3], [1, 2, 4], [2, 0, 5]]},
                                "QUADRANGLE_4N": {"ndim": 2,
                                                  "order": 1,
                                                  "n_vertices": 4,
                                                  "reversed_order": [1, 0, 3, 2],
                                                  "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]},
                                "QUADRANGLE_8N": {"ndim": 2,
                                                  "order": 2,
                                                  "n_vertices": 4,
                                                  "reversed_order": [1, 0, 3, 2, 4, 7, 6, 5],
                                                  "edges": [[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]]},
                                "TETRAHEDRON_4N": {"ndim": 3,
                                                   "order": 1,
                                                   "n_vertices": 4,
                                                   "reversed_order": [1, 0, 2, 3],
                                                   "edges": [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]},
                                "TETRAHEDRON_10N": {"ndim": 3,
                                                    "order": 2,
                                                    "n_vertices": 4,
                                                    "reversed_order": [1, 0, 2, 3, 4, 6, 5, 9, 8, 7],
                                                    "edges": [[0, 1, 4], [1, 2, 5], [2, 0, 6],
                                                              [0, 3, 7], [1, 3, 8], [2, 3, 9]]},
                                "HEXAHEDRON_8N": {"ndim": 3,
                                                  "order": 1,
                                                  "n_vertices": 8,
                                                  "reversed_order": [1, 0, 3, 2, 5, 4, 7, 6],
                                                  "edges": [[0, 1], [1, 2], [2, 3], [3, 0],
                                                            [4, 5], [5, 6], [6, 7], [7, 4],
                                                            [0, 4], [1, 5], [2, 6], [3, 7]]},
                                # todo make sure gmsh and kratos hexahedron order is the same, currently not the case
                                # "HEXAHEDRON_20N": {"ndim": 3,
                                #                    "order": 2,
                                #                    "n_vertices": 8,
                                #                    "reversed_order": [1, 0, 3, 2,
                                #                                       5, 4, 7, 6,
                                #                                       8, 9, 10, 11,
                                #                                       12, 13, 14, 15,
                                #                                       16, 17, 18, 19],
                                #                    "edges": [[0, 1, 8], [1, 2, 9], [2, 3, 10], [3, 0, 11],
                                #                              [4, 5, 16], [5, 6, 17], [6, 7, 18], [7, 4, 19],
                                #                              [0, 4, 12], [1, 5, 13], [2, 6, 14], [3, 7, 15]]}
                                }
