import pytest

from stem.water_boundaries import *

class TestWaterBoundaries:

    def test_raise_errors_for_water_boundaries(self):

        pytest.raises(ValueError, PhreaticMultiLineBoundary, x_coordinates=[0, 1, 2], y_coordinates=[0, 1, 2, 3])

        pytest.raises(ValueError, PhreaticMultiLineBoundary, x_coordinates=[0, 1, 2, 3, 4], y_coordinates=[0, 1, 2, 3])

        pytest.raises(ValueError,
                      PhreaticMultiLineBoundary,
                      x_coordinates=[0, 1, 2, 3],
                      y_coordinates=[0, 1, 2, 3],
                      z_coordinates=[0, 1, 2, 3, 4])