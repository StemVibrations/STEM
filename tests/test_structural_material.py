import pytest

from stem.structural_material import *


class TestStructuralMaterial:

    def test_raise_errors_for_material_parameters(self):
        """
        Tests that errors are raised when the structural material parameters are not valid.

        """

        # create 3d euler beam without I22, which is not allowed
        pytest.raises(ValueError, EulerBeam, ndim=3, DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2, CROSS_AREA=1.0,
                      I33=1, TORSIONAL_INERTIA=1)

        # create 3d euler beam without torsional inertia, which is not allowed
        pytest.raises(ValueError, EulerBeam, ndim=3, DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2, CROSS_AREA=1.0,
                      I33=1, I22=1)

    def test_get_property_in_structural_material(self):
        """
        Check that properties are correctly returned in soil material properties
        """
        beam_material = StructuralMaterial(
            "euler_beam",
            EulerBeam(ndim=3, DENSITY=1.0, YOUNG_MODULUS=1e5, POISSON_RATIO=0.2, CROSS_AREA=1.0, I33=1, I22=0.1,
                      TORSIONAL_INERTIA=1)
        )

        assert beam_material.get_property_in_material("YOUNG_MODULUS") == 1e5
        assert beam_material.get_property_in_material("POISSON_RATIO") == 0.2
        assert beam_material.get_property_in_material("CROSS_AREA") == 1.0

        msg = "Property YOUNGS_MODULUS is not one of the parameters of the structural material"
        with pytest.raises(ValueError, match=msg):
            beam_material.get_property_in_material("YOUNGS_MODULUS")
