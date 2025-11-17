import pytest

from stem.soil_material import *


class TestSoilMaterial:

    def test_raise_errors_for_material_parameters(self):
        """
        Tests that errors are raised when the soil material parameters are not valid.

        """

        # create 3d two phase soil without zz permeability, which is not allowed
        pytest.raises(ValueError,
                      TwoPhaseSoil,
                      ndim=3,
                      DENSITY_SOLID=2650,
                      POROSITY=0.3,
                      BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15,
                      PERMEABILITY_YY=1E-15,
                      PERMEABILITY_XY=1,
                      PERMEABILITY_ZX=2,
                      PERMEABILITY_YZ=3)

        # create 3d two phase soil without yz permeability, which is not allowed
        pytest.raises(ValueError,
                      TwoPhaseSoil,
                      ndim=3,
                      DENSITY_SOLID=2650,
                      POROSITY=0.3,
                      BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15,
                      PERMEABILITY_YY=1E-15,
                      PERMEABILITY_ZX=1,
                      PERMEABILITY_ZZ=2,
                      PERMEABILITY_YZ=None)

        # create 3d two phase soil without zx permeability, which is not allowed
        pytest.raises(ValueError,
                      TwoPhaseSoil,
                      ndim=3,
                      DENSITY_SOLID=2650,
                      POROSITY=0.3,
                      BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15,
                      PERMEABILITY_YY=1E-15,
                      PERMEABILITY_XY=1,
                      PERMEABILITY_ZX=None,
                      PERMEABILITY_ZZ=3)

    def test_get_property_in_soil_material(self):
        """
        Check that properties are correctly returned in soil material properties
        """
        ndim = 2
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil",
                                     soil_formulation=soil_formulation,
                                     constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        assert soil_material.get_property_in_material("YOUNG_MODULUS") == 100e6
        assert soil_material.get_property_in_material("POISSON_RATIO") == 0.3
        assert soil_material.get_property_in_material("POROSITY") == 0.3

        msg = "Property YOUNGS_MODULUS is not one of the parameters of the soil material"
        with pytest.raises(ValueError, match=msg):
            soil_material.get_property_in_material("YOUNGS_MODULUS")

    def test_to_dict_serializes_soil_material(self):
        """
        Check that the soil material is correctly serialized to a dictionary
        """

        soil_formulation = OnePhaseSoil(ndim=3, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        fluid_properties = FluidProperties(DENSITY_FLUID=1000, DYNAMIC_VISCOSITY=1.3e-3, BULK_MODULUS_FLUID=2e9)
        soil_material = SoilMaterial(
            name="soil",
            soil_formulation=soil_formulation,
            constitutive_law=constitutive_law,
            retention_parameters=retention_parameters,
            fluid_properties=fluid_properties,
        )
        serialized = soil_material.to_dict()
        assert serialized == {
            "name": "soil",
            "soil_formulation": {
                "ndim": 3,
                "IS_DRAINED": True,
                "DENSITY_SOLID": 2650,
                "POROSITY": 0.3,
                "BULK_MODULUS_SOLID": 50e9,
                "BIOT_COEFFICIENT": None,
                "RAYLEIGH_M": None,
                "RAYLEIGH_K": None,
                "type": "OnePhaseSoil",
            },
            "constitutive_law": {
                "YOUNG_MODULUS": 100e6,
                "POISSON_RATIO": 0.3,
                "type": "LinearElasticSoil",
            },
            "retention_parameters": {
                "SATURATED_SATURATION": 1.0,
                "RESIDUAL_SATURATION": 1e-10,
                "type": "SaturatedBelowPhreaticLevelLaw",
            },
            "fluid_properties": {
                "DENSITY_FLUID": 1000,
                "DYNAMIC_VISCOSITY": 1.3e-3,
                "BULK_MODULUS_FLUID": 2e9,
                "type": "FluidProperties",
            },
        }
