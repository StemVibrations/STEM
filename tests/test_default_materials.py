import json

from stem.IO.kratos_material_io import KratosMaterialIO
from stem.default_materials import DefaultMaterial

from tests.utils import TestUtils


class TestKratosMaterialIO:
    def test_default_structural_materials(self):
        """
        Test writing a material list to json. In this test, the material list contains a beam material, a spring damper
        material and a nodal concentrated material.

        """
        ndim = 3
        default_materials = [
            DefaultMaterial.Rail_46E3_3D.value,
            DefaultMaterial.Rail_54E1_3D.value,
            DefaultMaterial.Rail_60E1_3D.value
        ]

        all_materials = {m_obj.name:m_obj for m_obj in default_materials}

        # write json file
        material_io = KratosMaterialIO(ndim=ndim, domain="PorousDomain")
        # TODO: when model part are implemented, generate file through kratos_io

        test_dict = {"properties": []}
        for ix, (part_name, material_parameters) in enumerate(all_materials.items()):
            test_dict["properties"].append(
                material_io.create_material_dict(
                    part_name=part_name,
                    material=material_parameters,
                    material_id=ix + 1,
                )
            )

        expected_material_parameters_json = json.load(
            open("tests/test_data/expected_default_materials.json")
        )

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(
            test_dict, expected_material_parameters_json
        )
