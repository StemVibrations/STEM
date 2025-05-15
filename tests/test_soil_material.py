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

    def test_get_element_name_2d_4N(self):
        """
        Test the get_element_name method of the Interface class for 2D 4N elements.
        """
        ndim = 2
        n_nodes_element = 4
        analysis_type = AnalysisType.MECHANICAL

        # Valid case
        element_name = Interface.get_element_name(ndim, n_nodes_element, analysis_type)
        assert element_name == "UPwSmallStrainInterfaceElement2D4N"
    
    def test_get_element_name_3d_8N(self):
        """
        Test the get_element_name method of the Interface class for 3D 8N elements.

        """
        analysis_type = AnalysisType.MECHANICAL
        # Higher order element
        ndim = 3
        n_nodes_element = 8
        element_name = Interface.get_element_name(ndim, n_nodes_element, analysis_type)
        assert element_name == "UPwSmallStrainInterfaceElement3D8N"
    
    def test_get_element_name_3d_6N(self):
        """
        Test the get_element_name method of the Interface class for 3D 6N elements.

        """
        analysis_type = AnalysisType.MECHANICAL
        # Lower order element
        ndim = 3
        n_nodes_element = 6
        element_name = Interface.get_element_name(ndim, n_nodes_element, analysis_type)
        assert element_name == "UPwSmallStrainInterfaceElement3D6N"
    
    def test_get_element_name_unaivailable_analysis_type_for_soil_material(self):
        """
        Test the get_element_name method of the Interface class for unavailable analysis types.
        """
        ndim = 2
        n_nodes_element = 4

        # Invalid analysis type
        invalid_analysis_type = AnalysisType.GROUNDWATER_FLOW
        with pytest.raises(ValueError, match=f"Analysis type {invalid_analysis_type} is not implemented yet for soil material."):
            Interface.get_element_name(ndim, n_nodes_element, invalid_analysis_type)
    
    def test_get_element_name_invalid(self):
        """
        Test the get_element_name method of the Interface class for invalid cases.
        """
        ndim = 2
        n_nodes_element = 4

        # Invalid analysis type
        invalid_analysis_type = "INVALID_TYPE"
        with pytest.raises(ValueError, match=f"Analysis type {invalid_analysis_type} is not implemented yet for soil material."):
            Interface.get_element_name(ndim, n_nodes_element, invalid_analysis_type)

    def test_get_property_in_material(self):
        """
        Test the get_property_in_material method of the SoilMaterial class.
        """
        ndim =2     # Linear elastic drained soil with a Density of 2700, a Young's modulus of 50e6,
        # a Poisson ratio of 0.3 & a Porosity of 0.3 is specified.
        DENSITY_SOLID = 2700
        POROSITY = 0.3
        YOUNG_MODULUS = 50e6
        POISSON_RATIO = 0.3
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
        soil_formulation_one_phase = OnePhaseSoilInterface(ndim,
                                          IS_DRAINED=True,
                                          DENSITY_SOLID=DENSITY_SOLID,
                                          POROSITY=POROSITY,
                                          MINIMUM_JOINT_WIDTH=0.001)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        # Define interface material parameters
        interface_material_parameters = Interface(
            name="test_interface_material_one_phase_linear_elastic",
            constitutive_law=constitutive_law,
            soil_formulation=soil_formulation_one_phase,
            retention_parameters=retention_parameters,
        ) 
        # get the property in the material
        assert interface_material_parameters.get_property_in_material("YOUNG_MODULUS") == YOUNG_MODULUS
        assert interface_material_parameters.get_property_in_material("POISSON_RATIO") == POISSON_RATIO
        assert interface_material_parameters.get_property_in_material("DENSITY_SOLID") == DENSITY_SOLID
        assert interface_material_parameters.get_property_in_material("POROSITY") == POROSITY
        assert interface_material_parameters.get_property_in_material("MINIMUM_JOINT_WIDTH") == 0.001


    def test_get_property_in_material_property_not_in_material(self):
        """
        Test the get_property_in_material method of the SoilMaterial class when the property is not in the material.
        """
        ndim = 2
        DENSITY_SOLID = 2700
        POROSITY = 0.3
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=0.3)
        soil_formulation_one_phase = OnePhaseSoilInterface(ndim,
                                          IS_DRAINED=True,
                                          DENSITY_SOLID=DENSITY_SOLID,
                                          POROSITY=POROSITY,
                                          MINIMUM_JOINT_WIDTH=0.001)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        # Define interface material parameters
        interface_material_parameters = Interface(
            name="test_interface_material_one_phase_linear_elastic",
            constitutive_law=constitutive_law,
            soil_formulation=soil_formulation_one_phase,
            retention_parameters=retention_parameters,
        ) 
        # get the property in the material
        with pytest.raises(ValueError, match="Property INVALID_PROPERTY is not one of the parameters of the soil material"):
            interface_material_parameters.get_property_in_material("INVALID_PROPERTY")