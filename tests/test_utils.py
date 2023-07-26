
from stem.utils import *


class TestUtilsStem:

    def test_collinearity_2d(self):
        """
        Check collinearity between 3 points in 2D
        """
        p1 = np.array([0, 0])
        p2 = np.array([-2, -1])

        p_test_1 = np.array([2, 1])
        p_test_2 = np.array([-5, 1])

        assert is_collinear(point=p_test_1, start_point=p1, end_point=p2)
        assert not is_collinear(point=p_test_2, start_point=p1, end_point=p2)

    def test_collinearity_3d(self):
        """
        Check collinearity between 3 points in 3D
        """

        p1 = np.array([0, 0, 0])
        p2 = np.array([-2, -2, 2])

        p_test_1 = np.array([2, 2, -2])
        p_test_2 = np.array([2, -2, 2])

        assert is_collinear(point=p_test_1, start_point=p1, end_point=p2)
        assert not is_collinear(point=p_test_2, start_point=p1, end_point=p2)

    def test_is_in_between_2d(self):
        """
        Check if point is in between other 2 points in 2D
        """

        p1 = np.array([0, 0])
        p2 = np.array([-2, -2])

        p_test_1 = np.array([2, 2])
        p_test_2 = np.array([-1, -1])

        assert not is_point_between_points(point=p_test_1, start_point=p1, end_point=p2)
        assert is_point_between_points(point=p_test_2, start_point=p1, end_point=p2)

    def test_is_in_between_3d(self):
        """
        Check if point is in between other 2 points in 3D
        """

        p1 = np.array([0, 0, 0])
        p2 = np.array([-2, -2, 2])

        p_test_1 = np.array([2, 2, -2])
        p_test_2 = np.array([-1, -1, 1])

        assert not is_point_between_points(point=p_test_1, start_point=p1, end_point=p2)
        assert is_point_between_points(point=p_test_2, start_point=p1, end_point=p2)