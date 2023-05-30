import json

from stem.kratos_IO import KratosIO
from stem.material import SmallStrainUmat3DLaw, SmallStrainUdsm3DLaw, Material

from tests.utils import TestUtils

class TestKratosIO:

    def test_write_material_to_json(self):
        """
        Test writing a material to json
        Returns:

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
        expected_material_parameters_json = json.load(open("test_data/expected_material_parameters.json"))

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)

