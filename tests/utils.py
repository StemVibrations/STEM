from typing import Dict, Any

import numpy.testing as npt
from stem.geometry import Geometry
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw

class TestUtils:

    @staticmethod
    def create_default_soil_material(ndim: int):
        """
        Creates a default soil material.

        Args:
            - ndim (int): number of dimensions of the model

        Returns:
            - :class:`stem.soil_material.SoilMaterial`: default soil material

        """
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        return soil_material

    @staticmethod
    def assert_dictionary_almost_equal(expected: Dict[Any, Any], actual: Dict[Any, Any]):
        """
        Checks whether two dictionaries are equal.

        Args:
            - expected: Expected dictionary.
            - actual: Actual dictionary.

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
            - expected_geometry (:class:`stem.geometry.Geometry`): expected geometry of the model
            - actual_geometry (:class:`stem.geometry.Geometry`): actual geometry of the model

        Returns:

        """
        # check if points are added correctly
        for (generated_point_id, generated_point), (expected_point_id, expected_point) in \
                zip(actual_geometry.points.items(), expected_geometry.points.items()):
            assert generated_point_id == expected_point_id
            assert generated_point.id == expected_point.id
            npt.assert_allclose(generated_point.coordinates, expected_point.coordinates)

        # check if lines are added correctly
        for (generated_lines_id, generated_line), (expected_line_id, expected_line) in \
                zip(actual_geometry.lines.items(), expected_geometry.lines.items()):
            assert generated_lines_id == expected_line_id
            assert generated_line.id == expected_line.id
            npt.assert_equal(generated_line.point_ids, expected_line.point_ids)

        # check if surfaces are added correctly
        for (generated_surface_id, generated_surface), (expected_surface_id, expected_surface) in \
                zip(actual_geometry.surfaces.items(), expected_geometry.surfaces.items()):
            assert generated_surface_id == expected_surface_id
            assert generated_surface.id == expected_surface.id
            npt.assert_equal(generated_surface.line_ids, expected_surface.line_ids)

        # check if volumes are added correctly
        for (generated_volume_id, generated_volume), (expected_volume_id, expected_volume) in \
                zip(actual_geometry.volumes.items(), expected_geometry.volumes.items()):
            assert generated_volume_id == expected_volume_id
            assert generated_volume.id == expected_volume.id
            npt.assert_equal(generated_volume.surface_ids, expected_volume.surface_ids)
