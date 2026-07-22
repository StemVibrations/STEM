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
                      BULK_MODULUS_SOLID=1e9,
                      PERMEABILITY_XX=1e-15,
                      PERMEABILITY_YY=1e-15,
                      PERMEABILITY_XY=1,
                      PERMEABILITY_ZX=2,
                      PERMEABILITY_YZ=3)

        # create 3d two phase soil without yz permeability, which is not allowed
        pytest.raises(ValueError,
                      TwoPhaseSoil,
                      ndim=3,
                      DENSITY_SOLID=2650,
                      POROSITY=0.3,
                      BULK_MODULUS_SOLID=1e9,
                      PERMEABILITY_XX=1e-15,
                      PERMEABILITY_YY=1e-15,
                      PERMEABILITY_ZX=1,
                      PERMEABILITY_ZZ=2,
                      PERMEABILITY_YZ=None)

        # create 3d two phase soil without zx permeability, which is not allowed
        pytest.raises(ValueError,
                      TwoPhaseSoil,
                      ndim=3,
                      DENSITY_SOLID=2650,
                      POROSITY=0.3,
                      BULK_MODULUS_SOLID=1e9,
                      PERMEABILITY_XX=1e-15,
                      PERMEABILITY_YY=1e-15,
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

    @pytest.mark.parametrize("ndim, n_nodes, expected_element_name", [
        (2, 4, "Geo_ULineInterfacePlaneStrainElement2Plus2N"),
        (3, 8, "Geo_USurfaceInterfaceElement4Plus4N"),
        (3, 6, "Geo_USurfaceInterfaceElement3Plus3N"),
    ])
    def test_get_interface_element_name(self, ndim, n_nodes, expected_element_name):
        """
        Test the get_element_name method of the Interface class for 2D 4N elements.
        """
        analysis_type = AnalysisType.MECHANICAL

        # Valid case
        element_name = InterfaceMaterial.get_element_name(ndim, n_nodes, analysis_type)
        assert element_name == expected_element_name

    def test_get_interface_element_name_unavailable_analysis_type_for_soil_material(self):
        """
        Test the get_element_name method of the Interface class for unavailable analysis types.
        """
        ndim = 2
        n_nodes_element = 4

        # Invalid analysis type
        invalid_analysis_type = AnalysisType.GROUNDWATER_FLOW
        with pytest.raises(
                ValueError,
                match=f"Analysis type {invalid_analysis_type} is not implemented yet for interface material."):
            InterfaceMaterial.get_element_name(ndim, n_nodes_element, invalid_analysis_type)

    def test_get_element_name_invalid(self):
        """
        Test the get_element_name method of the Interface class for invalid cases.
        """
        ndim = 2
        n_nodes_element = 4

        # Invalid analysis type
        invalid_analysis_type = "INVALID_TYPE"
        with pytest.raises(
                ValueError,
                match=f"Analysis type {invalid_analysis_type} is not implemented yet for interface material."):
            InterfaceMaterial.get_element_name(ndim, n_nodes_element, invalid_analysis_type)

    def test_get_property_in_interface_material(self):
        """
        Test the get_property_in_material method of the InterfaceMaterial class.
        """
        ndim = 2
        DENSITY_SOLID = 2700
        POROSITY = 0.3
        YOUNG_MODULUS = 50e6
        POISSON_RATIO = 0.3
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
        soil_formulation_one_phase = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        # Define interface material parameters
        interface_material_parameters = InterfaceMaterial(name="test_interface_material_one_phase_linear_elastic",
                                                          constitutive_law=constitutive_law,
                                                          soil_formulation=soil_formulation_one_phase,
                                                          retention_parameters=retention_parameters)
        # get the property in the material
        assert (interface_material_parameters.get_property_in_material("YOUNG_MODULUS") == YOUNG_MODULUS)
        assert (interface_material_parameters.get_property_in_material("POISSON_RATIO") == POISSON_RATIO)
        assert (interface_material_parameters.get_property_in_material("DENSITY_SOLID") == DENSITY_SOLID)
        assert (interface_material_parameters.get_property_in_material("POROSITY") == POROSITY)

    def test_get_property_in_interface_material_property_not_in_material(self):
        """
        Test the get_property_in_material method of the InterfaceMaterial class when the property is not in the material.
        """
        ndim = 2
        DENSITY_SOLID = 2700
        POROSITY = 0.3
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=0.3)
        soil_formulation_one_phase = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        # Define interface material parameters
        interface_material_parameters = InterfaceMaterial(name="test_interface_material_one_phase_linear_elastic",
                                                          constitutive_law=constitutive_law,
                                                          soil_formulation=soil_formulation_one_phase,
                                                          retention_parameters=retention_parameters)
        # get the property in the material
        with pytest.raises(ValueError,
                           match="Property INVALID_PROPERTY is not one of the parameters of the interface material"):
            interface_material_parameters.get_property_in_material("INVALID_PROPERTY")
