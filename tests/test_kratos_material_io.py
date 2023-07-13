import json

from stem.IO.kratos_material_io import KratosMaterialIO
from stem.soil_material import *
from stem.structural_material import *

from tests.utils import TestUtils


class TestKratosMaterialIO:
    def test_write_soil_material_dict(self):
        """
        Test writing a material list to json. In this test, the material list contains a UMAT and a UDSM material.

        """
        ndim = 3

        # create drained soil
        umat_formulation = OnePhaseSoil(
            ndim=ndim,
            IS_DRAINED=True,
            DENSITY_SOLID=2650,
            POROSITY=0.3,
            BULK_MODULUS_SOLID=1e16,
        )

        # Define umat constitutive law parameters
        umat_constitutive_parameters = SmallStrainUmatLaw(
            UMAT_PARAMETERS=[1, 5.6, False],
            UMAT_NAME="test_name",
            IS_FORTRAN_UMAT=False,
            STATE_VARIABLES=[],
        )

        umat_retention_parameters = SaturatedBelowPhreaticLevelLaw()

        # Create undrained soil
        udsm_formulation = OnePhaseSoil(
            ndim=ndim,
            IS_DRAINED=False,
            DENSITY_SOLID=2650,
            POROSITY=0.3,
            BULK_MODULUS_SOLID=1e16,
            BIOT_COEFFICIENT=0.5,
        )

        # Define retention law parameters
        udsm_retention_parameters = VanGenuchtenLaw(
            VAN_GENUCHTEN_AIR_ENTRY_PRESSURE=1,
            VAN_GENUCHTEN_GN=0.2,
            VAN_GENUCHTEN_GL=0.5,
        )

        # Define udsm constitutive law parameters
        udsm_constitutive_parameters = SmallStrainUdsmLaw(
            UDSM_PARAMETERS=[1, 5.6, False],
            UDSM_NUMBER=2,
            UDSM_NAME="test_name_UDSM",
            IS_FORTRAN_UDSM=True,
        )

        # Create two phase soil
        two_phase_formulation_2D = TwoPhaseSoil(
            ndim=2,
            DENSITY_SOLID=2650,
            POROSITY=0.3,
            BULK_MODULUS_SOLID=1e9,
            PERMEABILITY_XX=1e-15,
            PERMEABILITY_YY=1e-15,
            PERMEABILITY_XY=2,
        )

        two_phase_formulation_3D = TwoPhaseSoil(
            ndim=3,
            DENSITY_SOLID=2650,
            POROSITY=0.3,
            PERMEABILITY_XX=1e-15,
            PERMEABILITY_YY=1e-15,
            PERMEABILITY_ZZ=1e-15,
            PERMEABILITY_XY=1,
            PERMEABILITY_ZX=2,
            PERMEABILITY_YZ=3,
        )

        # Define two phase constitutive law parameters
        two_phase_constitutive_parameters = LinearElasticSoil(
            YOUNG_MODULUS=1e9, POISSON_RATIO=0.3
        )
        two_phase_retention_parameters = SaturatedBelowPhreaticLevelLaw()

        # Create materials
        umat_material = SoilMaterial(
            name="test_umat_material",
            soil_formulation=umat_formulation,
            constitutive_law=umat_constitutive_parameters,
            retention_parameters=umat_retention_parameters,
        )

        udsm_material = SoilMaterial(
            name="test_udsm_material",
            soil_formulation=udsm_formulation,
            constitutive_law=udsm_constitutive_parameters,
            retention_parameters=udsm_retention_parameters,
        )

        two_phase_material_2D = SoilMaterial(
            name="test_two_phase_material_2D",
            soil_formulation=two_phase_formulation_2D,
            constitutive_law=two_phase_constitutive_parameters,
            retention_parameters=two_phase_retention_parameters,
        )

        two_phase_material_3D = SoilMaterial(
            name="test_two_phase_material_3D",
            soil_formulation=two_phase_formulation_3D,
            constitutive_law=two_phase_constitutive_parameters,
            retention_parameters=two_phase_retention_parameters,
        )

        all_materials = {
            "test_umat_material": umat_material,
            "test_udsm_material": udsm_material,
            "test_two_phase_material_2D": two_phase_material_2D,
            "test_two_phase_material_3D": two_phase_material_3D,
        }

        # write json file
        material_io = KratosMaterialIO(ndim=3, domain="PorousDomain")
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
            open("tests/test_data/expected_material_parameters.json")
        )

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(
            expected_material_parameters_json, test_dict
        )

    def test_write_structural_material_dict(self):
        """
        Test writing a material list to json. In this test, the material list contains a beam material, a spring damper
        material and a nodal concentrated material.

        """
        ndim = 2
        # define euler beam parameters
        beam_material_parameters = EulerBeam(
            ndim=ndim,
            DENSITY=1.0,
            YOUNG_MODULUS=1.0,
            POISSON_RATIO=0.2,
            CROSS_AREA=1.0,
            I33=1,
        )

        # define spring damper parameters
        spring_damper_material_parameters = ElasticSpringDamper(
            NODAL_DISPLACEMENT_STIFFNESS=[1.0, 2, 3],
            NODAL_DAMPING_COEFFICIENT=[0, 0.2, 3],
            NODAL_ROTATIONAL_STIFFNESS=[2.0, 4, 5],
            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 9],
        )

        # define nodal concentrated parameters
        nodal_concentrated_material_parameters = NodalConcentrated(
            NODAL_MASS=1.0,
            NODAL_DAMPING_COEFFICIENT=[1, 2, 0.2],
            NODAL_DISPLACEMENT_STIFFNESS=[1, 2, 3],
        )

        # Create structural materials
        beam_material = StructuralMaterial(
            name="test_beam_material", material_parameters=beam_material_parameters
        )
        spring_damper_material = StructuralMaterial(
            name="test_spring_damper_material",
            material_parameters=spring_damper_material_parameters,
        )
        nodal_concentrated_material = StructuralMaterial(
            name="test_nodal_concentrated_material",
            material_parameters=nodal_concentrated_material_parameters,
        )

        all_materials = {
            "test_beam_material": beam_material,
            "test_spring_damper_material": spring_damper_material,
            "test_nodal_concentrated_material": nodal_concentrated_material,
        }

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
            open("tests/test_data/expected_structural_material_parameters.json")
        )

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(
            expected_material_parameters_json, test_dict
        )
