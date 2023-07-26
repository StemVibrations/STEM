import pprint
from typing import Dict, Any

import numpy.testing as npt
import pytest

from stem.geometry import Geometry


class TestUtils:

    @staticmethod
    def assert_dictionary_almost_equal(expected: Dict[Any, Any], actual: Dict[Any, Any]):
        """
        Checks whether two dictionaries are equal.

        Args:
            expected: Expected dictionary.
            actual: Actual dictionary.

        """

        for k, v in expected.items():

            assert k in actual

            if isinstance(v, dict):
                TestUtils.assert_dictionary_almost_equal(v, actual[k])
            elif isinstance(v, str):
                assert v == actual[k]
            elif isinstance(v, list):
                assert len(v) == len(actual[k])
                for v_i, actual_i in zip(v, actual[k]):
                    if isinstance(v_i, dict):
                        TestUtils.assert_dictionary_almost_equal(v_i, actual_i)
                    elif isinstance(v_i, str):
                        assert v_i == actual_i
                    else:
                        npt.assert_allclose(v_i, actual_i)

            else:
                npt.assert_allclose(v, actual[k])

    @staticmethod
    def assert_almost_equal_geometries(expected_geometry: Geometry, actual_geometry:Geometry):
        """
        Checks whether two Geometries are (almost) equal.

        Args:
            expected_geometry (:class:`stem.geometry.Geometry`): expected geometry of the model
            actual_geometry (:class:`stem.geometry.Geometry`): actual geometry of the model

        Returns:

        """
        # check if points are added correctly
        for generated_point, expected_point in zip(actual_geometry.points, expected_geometry.points):
            assert generated_point.id == expected_point.id
            assert pytest.approx(generated_point.coordinates) == expected_point.coordinates

        # check if lines are added correctly
        for generated_line, expected_line in zip(actual_geometry.lines, expected_geometry.lines):
            assert generated_line.id == expected_line.id
            assert generated_line.point_ids == expected_line.point_ids

        # check if surfaces are added correctly
        for generated_surface, expected_surface in zip(actual_geometry.surfaces, expected_geometry.surfaces):
            assert generated_surface.id == expected_surface.id
            assert generated_surface.line_ids == expected_surface.line_ids

        # check if volumes are added correctly
        for generated_volume, expected_volume in zip(actual_geometry.volumes, expected_geometry.volumes):
            assert generated_volume.id == expected_volume.id
            assert generated_volume.surface_ids == expected_volume.surface_ids
