import json

from stem.kratos_IO import KratosIO
from stem.material import (Material, SmallStrainUmatLaw,
SmallStrainUdsmLaw,
LinearElasticSoil,
EulerBeam2D,
EulerBeam3D,
ElasticSpringDamper,
NodalConcentrated,
                           DrainedSoil, UndrainedSoil, TwoPhaseSoil2D, TwoPhaseSoil3D)
from stem.retention_law import SaturatedBelowPhreaticLevelLaw, VanGenuchtenLaw

from tests.utils import TestUtils

class TestKratosIO:

    def test_write_soil_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a UMAT and a UDSM material.

        """

        soil_type = DrainedSoil(DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E16)

        # Define material parameters
        umat_material_parameters = SmallStrainUmatLaw(UMAT_PARAMETERS=[1, 5.6, False], UMAT_NAME="test_name",
                                                      IS_FORTRAN_UMAT=False, STATE_VARIABLES=[], SOIL_TYPE=soil_type,
                                                      RETENTION_PARAMETERS=SaturatedBelowPhreaticLevelLaw())



        udsm_soil_type = UndrainedSoil(DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E16, BIOT_COEFFICIENT=0.5)
        udsm_retention_parameters = VanGenuchtenLaw(VAN_GENUCHTEN_AIR_ENTRY_PRESSURE=1, VAN_GENUCHTEN_GN=0.2,
                                                    VAN_GENUCHTEN_GL=0.5)

        udsm_material_parameters = SmallStrainUdsmLaw(SOIL_TYPE=udsm_soil_type,
                                                      RETENTION_PARAMETERS=udsm_retention_parameters,
                                                      UDSM_PARAMETERS=[1,5.6,False], UDSM_NUMBER=2,
                                                      UDSM_NAME="test_name_UDSM", IS_FORTRAN_UDSM=True)

        umat_material = Material(id=0, name="test_umat_material", material_parameters=umat_material_parameters)
        udsm_material = Material(id=1, name="test_udsm_material", material_parameters=udsm_material_parameters)

        all_materials = [umat_material, udsm_material]

        # write json file
        kratos_io = KratosIO(ndim=3)
        kratos_io.write_material_parameters_json(all_materials, "test_write_MaterialParameters.json")

        # read generated json file and expected json file
        written_material_parameters_json = json.load(open("test_write_MaterialParameters.json"))
        expected_material_parameters_json = json.load(open("tests/test_data/expected_material_parameters.json"))

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)

    # def test_write_structural_material_to_json(self):
    #     """
    #     Test writing a material list to json. In this test, the material list contains a beam material, a spring damper
    #     material and a nodal concentrated material.
    #
    #     """
    #
    #     # Define material parameters
    #     beam_material_parameters = EulerBeam2D(DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2)
    #     spring_damper_material_parameters = SpringDamperLaw(NODAL_DISPLACEMENT_STIFFNESS=1.0,
    #                                                         NODAL_DAMPING_COEFFICIENT=0.2)
    #     nodal_concentrated_material_parameters = NodalConcentratedLaw( NODAL_MASS=1.0, NODAL_DAMPING_COEFFICIENT=0.2)
    #
    #     beam_material = Material("test_beam_material", material_parameters=beam_material_parameters)
    #     spring_damper_material = Material("test_spring_damper_material",
    #                                       material_parameters=spring_damper_material_parameters)
    #     nodal_concentrated_material = Material("test_nodal_concentrated_material",
    #                                            material_parameters=nodal_concentrated_material_parameters)
    #
    #     all_materials = [beam_material, spring_damper_material, nodal_concentrated_material]
    #
    #     # write json file
    #     kratos_io = KratosIO()
    #     kratos_io.write_material_parameters_json(all_materials, "test_write_structural_MaterialParameters.json")
    #
    #     # read generated json file and expected json file
    #
    #     written_material_parameters_json = json.load(open("test_write_structural_MaterialParameters.json"))
    #     expected_material_parameters_json = json.load(open("tests/test_data/expected_structural_material_parameters.json"))
    #
    #     # compare json files using custom dictionary comparison
    #     TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)
