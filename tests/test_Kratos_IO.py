import json

from stem.kratos_IO import KratosIO
from stem.material import (SmallStrainUmat3DLaw,
                           SmallStrainUdsm3DLaw,
                           BeamLaw,
                           SpringDamperLaw,
                           NodalConcentratedLaw,
                           Material)

from tests.utils import TestUtils

class TestKratosIO:

    def test_write_soil_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a UMAT and a UDSM material.

        """

        # Define material parameters
        umat_material_parameters = SmallStrainUmat3DLaw(DENSITY_SOLID=1.0, UMAT_PARAMETERS=[1,5.6,False],
                                                        UMAT_NAME="test_name")
        udsm_material_parameters = SmallStrainUdsm3DLaw(DENSITY_WATER=1.0, UDSM_PARAMETERS=[1,5.6,False],
                                                        UDSM_NAME="test_name")

        umat_material = Material("test_umat_material", material_parameters=umat_material_parameters)
        udsm_material = Material("test_udsm_material", material_parameters=udsm_material_parameters)

        all_materials = [umat_material, udsm_material]

        # write json file
        kratos_io = KratosIO()
        kratos_io.write_material_parameters_json(all_materials, "test_write_MaterialParameters.json")

        # read generated json file and expected json file

        written_material_parameters_json = json.load(open("test_write_MaterialParameters.json"))
        expected_material_parameters_json = json.load(open("tests/test_data/expected_material_parameters.json"))

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)

    def test_write_structural_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a beam material, a spring damper
        material and a nodal concentrated material.

        """

        # Define material parameters
        beam_material_parameters = BeamLaw(DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2)
        spring_damper_material_parameters = SpringDamperLaw(NODAL_DISPLACEMENT_STIFFNESS=1.0,
                                                            NODAL_DAMPING_COEFFICIENT=0.2)
        nodal_concentrated_material_parameters = NodalConcentratedLaw( NODAL_MASS=1.0, NODAL_DAMPING_COEFFICIENT=0.2)

        beam_material = Material("test_beam_material", material_parameters=beam_material_parameters)
        spring_damper_material = Material("test_spring_damper_material",
                                          material_parameters=spring_damper_material_parameters)
        nodal_concentrated_material = Material("test_nodal_concentrated_material",
                                               material_parameters=nodal_concentrated_material_parameters)

        all_materials = [beam_material, spring_damper_material, nodal_concentrated_material]

        # write json file
        kratos_io = KratosIO()
        kratos_io.write_material_parameters_json(all_materials, "test_write_structural_MaterialParameters.json")

        # read generated json file and expected json file

        written_material_parameters_json = json.load(open("test_write_structural_MaterialParameters.json"))
        expected_material_parameters_json = json.load(open("tests/test_data/expected_structural_material_parameters.json"))

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)
