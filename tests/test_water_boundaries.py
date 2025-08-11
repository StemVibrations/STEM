import pytest

from stem.water_boundaries import *


class TestWaterBoundaries:

    def test_raise_errors_for_water_boundaries(self):

        pytest.raises(ValueError,
                      PhreaticMultiLineBoundary,
                      x_coordinates=[0, 1, 2],
                      y_coordinates=[0, 1, 2, 3],
                      surfaces_assigment=["surface_1", "surface_2", "surface_3"],
                      is_fixed=True,
                      gravity_direction=1,
                      out_of_plane_direction=2,
                      specific_weight=9.81,
                      water_pressure=1000)

        pytest.raises(ValueError,
                      PhreaticMultiLineBoundary,
                      x_coordinates=[0, 1, 2, 3, 4],
                      y_coordinates=[0, 1, 2, 3],
                      surfaces_assigment=["surface_1", "surface_2", "surface_3"],
                      is_fixed=True,
                      gravity_direction=1,
                      out_of_plane_direction=2,
                      specific_weight=9.81,
                      water_pressure=1000
                      )

        pytest.raises(ValueError,
                      PhreaticMultiLineBoundary,
                      x_coordinates=[0, 1, 2, 3],
                      y_coordinates=[0, 1, 2, 3],
                      z_coordinates=[0, 1, 2, 3, 4],
                      surfaces_assigment=["surface_1", "surface_2", "surface_3"],
                      is_fixed=True,
                      gravity_direction=1,
                      out_of_plane_direction=2,
                      specific_weight=9.81,
                      water_pressure=1000
                      )