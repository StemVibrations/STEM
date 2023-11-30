import pytest

from stem.soil_material import *


class TestSoilMaterial:

    def test_raise_errors_for_material_parameters(self):
        """
        Tests that errors are raised when the soil material parameters are not valid.

        """

        # create 3d two phase soil without zz permeability, which is not allowed
        pytest.raises(ValueError, TwoPhaseSoil, ndim=3, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15, PERMEABILITY_XY=1, PERMEABILITY_ZX=2,
                      PERMEABILITY_YZ=3)

        # create 3d two phase soil without yz permeability, which is not allowed
        pytest.raises(ValueError, TwoPhaseSoil, ndim=3, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15, PERMEABILITY_ZX=1, PERMEABILITY_ZZ=2,
                      PERMEABILITY_YZ=None)

        # create 3d two phase soil without zx permeability, which is not allowed
        pytest.raises(ValueError, TwoPhaseSoil, ndim=3, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15, PERMEABILITY_XY=1, PERMEABILITY_ZX=None,
                      PERMEABILITY_ZZ=3)

    def test_get_property_in_soil_material(self):
        """
        Check that properties are correctly returned in soil material properties
        """
        ndim = 2
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        assert soil_material.get_property_in_material("YOUNG_MODULUS") == 100e6
        assert soil_material.get_property_in_material("POISSON_RATIO") == 0.3
        assert soil_material.get_property_in_material("POROSITY") == 0.3

        msg = "Property YOUNGS_MODULUS is not one of the parameters of the soil material"
        with pytest.raises(ValueError, match=msg):
            soil_material.get_property_in_material("YOUNGS_MODULUS")
