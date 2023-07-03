import json

from stem.IO.kratos_io import KratosIO
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.mesh import Node
from stem.model import Model
from stem.model_part import BodyModelPart
from stem.soil_material import *
from stem.structural_material import *

from tests.utils import TestUtils


class TestKratosMaterialIO:
    def test_write_soil_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a UMAT and a UDSM material.

        """
        ndim = 3

        # dummy node
        node = Node(id=1, coordinates=(0, 0, 0))

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

        # assign them to body model parts and create body model parts
        bmp_umat = BodyModelPart(
            name="test_umat_material", nodes=[node], parameters=umat_material
        )
        bmp_udsm = BodyModelPart(
            name="test_udsm_material", nodes=[node], parameters=udsm_material
        )
        bmp_two_phase_2D = BodyModelPart(
            name="test_two_phase_material_2D",
            nodes=[node],
            parameters=two_phase_material_2D,
        )
        bmp_two_phase_3D = BodyModelPart(
            name="test_two_phase_material_3D",
            nodes=[node],
            parameters=two_phase_material_3D,
        )

        body_model_parts = [bmp_umat, bmp_udsm, bmp_two_phase_2D, bmp_two_phase_3D]

        # write json file
        kratos_io = KratosIO(
            ndim=3, model=Model(ndim=ndim, body_model_parts=body_model_parts)
        )
        kratos_io.write_material_parameters_json("test_write_MaterialParameters.json")

        # read generated json file and expected json file
        written_material_parameters_json = json.load(
            open("test_write_MaterialParameters.json")
        )
        expected_material_parameters_json = json.load(
            open("tests/test_data/expected_material_parameters.json")
        )

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(
            written_material_parameters_json, expected_material_parameters_json
        )

    def test_write_structural_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a beam material, a spring damper
        material and a nodal concentrated material.

        """

        # dummy node
        node = Node(id=1, coordinates=(0, 0, 0))

        # define ndim
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
            name="test_beam_material",
            material_parameters=beam_material_parameters,
        )
        spring_damper_material = StructuralMaterial(
            name="test_spring_damper_material",
            material_parameters=spring_damper_material_parameters,
        )
        nodal_concentrated_material = StructuralMaterial(
            name="test_nodal_concentrated_material",
            material_parameters=nodal_concentrated_material_parameters,
        )

        # assign them to body model parts and create body model parts
        bmp_beam = BodyModelPart(
            name="test_beam_material", nodes=[node], parameters=beam_material
        )
        bmp_spring_damper = BodyModelPart(
            name="test_spring_damper_material",
            nodes=[node],
            parameters=spring_damper_material,
        )
        bmp_nodal_concentrated = BodyModelPart(
            name="test_nodal_concentrated_material",
            nodes=[node],
            parameters=nodal_concentrated_material,
        )

        body_model_parts = [bmp_beam, bmp_spring_damper, bmp_nodal_concentrated]

        # write json file
        kratos_io = KratosIO(
            ndim=2, model=Model(ndim=2, body_model_parts=body_model_parts)
        )
        kratos_io.write_material_parameters_json(
            "test_write_structural_MaterialParameters.json"
        )

        # read generated json file and expected json file
        written_material_parameters_json = json.load(
            open("test_write_structural_MaterialParameters.json")
        )
        expected_material_parameters_json = json.load(
            open("tests/test_data/expected_structural_material_parameters.json")
        )

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(
            written_material_parameters_json, expected_material_parameters_json
        )
