import pytest

from stem.globals import ELEMENT_DATA


class TestGlobalsStem:

    def test_elements_names(self):
        """
        Test if all keys are correct
        """
        key_list = ["POINT_1N", "LINE_2N", "LINE_3N", "TRIANGLE_3N", "TRIANGLE_6N", "QUADRANGLE_4N", "QUADRANGLE_8N",
                    "TETRAHEDRON_4N", "TETRAHEDRON_10N", "HEXAHEDRON_8N"]

        assert all(key in ELEMENT_DATA for key in key_list)


    def test_point(self):
        """
        Test the ELEMENT DATA order for Point
        """
        assert ELEMENT_DATA["POINT_1N"]["ndim"] == 0
        assert ELEMENT_DATA["POINT_1N"]["order"] == 1
        assert ELEMENT_DATA["POINT_1N"]["n_vertices"] == 1
        assert ELEMENT_DATA["POINT_1N"]["reversed_order"] == [0]
        assert not ELEMENT_DATA["POINT_1N"]["edges"]

    def test_line2n(self):
        """
        Test the ELEMENT DATA order for Point
        """
        assert ELEMENT_DATA["LINE_2N"]["ndim"] == 1
        assert ELEMENT_DATA["LINE_2N"]["order"] == 1
        assert ELEMENT_DATA["LINE_2N"]["n_vertices"] == 2
        assert ELEMENT_DATA["LINE_2N"]["reversed_order"] == [1, 0]
        assert ELEMENT_DATA["LINE_2N"]["edges"] == [[0, 1]]

    def test_line3n(self):
        """
        Test the ELEMENT DATA order for Line2
        """
        assert ELEMENT_DATA["LINE_3N"]["ndim"] == 1
        assert ELEMENT_DATA["LINE_3N"]["order"] == 2
        assert ELEMENT_DATA["LINE_3N"]["n_vertices"] == 2
        assert ELEMENT_DATA["LINE_3N"]["reversed_order"] == [1, 0, 2]
        assert ELEMENT_DATA["LINE_3N"]["edges"] == [[0, 1, 2]]

    def test_tri3n(self):
        """
        Test the ELEMENT DATA order for Tri3
        """
        assert ELEMENT_DATA["TRIANGLE_3N"]["ndim"] == 2
        assert ELEMENT_DATA["TRIANGLE_3N"]["order"] == 1
        assert ELEMENT_DATA["TRIANGLE_3N"]["n_vertices"] == 3
        assert ELEMENT_DATA["TRIANGLE_3N"]["reversed_order"] == [2, 1, 0]
        assert ELEMENT_DATA["TRIANGLE_3N"]["edges"] == [[1, 2], [2, 0], [0, 1]]

    def test_tri6n(self):
        """
        Test the ELEMENT DATA order for Tri6
        """
        assert ELEMENT_DATA["TRIANGLE_6N"]["ndim"] == 2
        assert ELEMENT_DATA["TRIANGLE_6N"]["order"] == 2
        assert ELEMENT_DATA["TRIANGLE_6N"]["n_vertices"] == 3
        assert ELEMENT_DATA["TRIANGLE_6N"]["reversed_order"] == [2, 1, 0, 5, 4, 3]
        assert ELEMENT_DATA["TRIANGLE_6N"]["edges"] == [[1, 2, 3], [1, 2, 4], [2, 0, 5]]

    def test_quad4n(self):
        """
        Test the ELEMENT DATA order for Qaud4
        """
        assert ELEMENT_DATA["QUADRANGLE_4N"]["ndim"] == 2
        assert ELEMENT_DATA["QUADRANGLE_4N"]["order"] == 1
        assert ELEMENT_DATA["QUADRANGLE_4N"]["n_vertices"] == 4
        assert ELEMENT_DATA["QUADRANGLE_4N"]["reversed_order"] == [1, 0, 3, 2]
        assert ELEMENT_DATA["QUADRANGLE_4N"]["edges"] == [[0, 1], [1, 2], [2, 3], [3, 0]]

    def test_quad8n(self):
        """
        Test the ELEMENT DATA order for Quad8
        """
        assert ELEMENT_DATA["QUADRANGLE_8N"]["ndim"] == 2
        assert ELEMENT_DATA["QUADRANGLE_8N"]["order"] == 2
        assert ELEMENT_DATA["QUADRANGLE_8N"]["n_vertices"] == 4
        assert ELEMENT_DATA["QUADRANGLE_8N"]["reversed_order"] == [1, 0, 3, 2, 4, 7, 6, 5]
        assert ELEMENT_DATA["QUADRANGLE_8N"]["edges"] == [[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]]

    def test_tetra4n(self):
        """
        Test the ELEMENT DATA order for Tetra4
        """
        assert ELEMENT_DATA["TETRAHEDRON_4N"]["ndim"] == 3
        assert ELEMENT_DATA["TETRAHEDRON_4N"]["order"] == 1
        assert ELEMENT_DATA["TETRAHEDRON_4N"]["n_vertices"] == 4
        assert ELEMENT_DATA["TETRAHEDRON_4N"]["reversed_order"] == [1, 0, 2, 3]
        assert ELEMENT_DATA["TETRAHEDRON_4N"]["edges"] == [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]

    def test_tetra10n(self):
        """
        Test the ELEMENT DATA order for Tetra10
        """
        assert ELEMENT_DATA["TETRAHEDRON_10N"]["ndim"] == 3
        assert ELEMENT_DATA["TETRAHEDRON_10N"]["order"] == 2
        assert ELEMENT_DATA["TETRAHEDRON_10N"]["n_vertices"] == 4
        assert ELEMENT_DATA["TETRAHEDRON_10N"]["reversed_order"] == [1, 0, 2, 3, 4, 6, 5, 9, 8, 7]
        assert ELEMENT_DATA["TETRAHEDRON_10N"]["edges"] == [[0, 1, 4], [1, 2, 5], [2, 0, 6],
                                                              [0, 3, 7], [1, 3, 8], [2, 3, 9]]

    def test_hexa8n(self):
        """
        Test the ELEMENT DATA order for Hexa8
        """
        assert ELEMENT_DATA["HEXAHEDRON_8N"]["ndim"] == 3
        assert ELEMENT_DATA["HEXAHEDRON_8N"]["order"] == 1
        assert ELEMENT_DATA["HEXAHEDRON_8N"]["n_vertices"] == 8
        assert ELEMENT_DATA["HEXAHEDRON_8N"]["reversed_order"] == [1, 0, 3, 2, 5, 4, 7, 6]
        assert ELEMENT_DATA["HEXAHEDRON_8N"]["edges"] == [[0, 1], [1, 2], [2, 3], [3, 0],
                                                            [4, 5], [5, 6], [6, 7], [7, 4],
                                                            [0, 4], [1, 5], [2, 6], [3, 7]]
