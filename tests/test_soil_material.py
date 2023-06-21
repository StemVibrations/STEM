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




