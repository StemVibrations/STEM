import json

import pytest

from stem.kratos_IO import KratosIO
from stem.soil_material import *
from stem.structural_material import *

from stem.load import PointLoad, MovingLoad, Load
from tests.utils import TestUtils


class TestKratosIO:

    def test_write_soil_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a UMAT and a UDSM material.

        """
        ndim = 3

        # create drained soil
        umat_formulation = OnePhaseSoil(ndim=ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3,
                                        BULK_MODULUS_SOLID=1E16)

        # Define umat constitutive law parameters
        umat_constitutive_parameters = SmallStrainUmatLaw(UMAT_PARAMETERS=[1, 5.6, False], UMAT_NAME="test_name",
                                                          IS_FORTRAN_UMAT=False, STATE_VARIABLES=[])

        umat_retention_parameters = SaturatedBelowPhreaticLevelLaw()


        # Create undrained soil
        udsm_formulation = OnePhaseSoil(ndim=ndim, IS_DRAINED=False, DENSITY_SOLID=2650, POROSITY=0.3,
                                        BULK_MODULUS_SOLID=1E16, BIOT_COEFFICIENT=0.5)

        # Define retention law parameters
        udsm_retention_parameters = VanGenuchtenLaw(VAN_GENUCHTEN_AIR_ENTRY_PRESSURE=1, VAN_GENUCHTEN_GN=0.2,
                                                    VAN_GENUCHTEN_GL=0.5)

        # Define udsm constitutive law parameters
        udsm_constitutive_parameters = SmallStrainUdsmLaw(
                                                      UDSM_PARAMETERS=[1, 5.6, False], UDSM_NUMBER=2,
                                                      UDSM_NAME="test_name_UDSM", IS_FORTRAN_UDSM=True)

        # Create two phase soil
        two_phase_formulation_2D = TwoPhaseSoil(ndim=2, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                                             PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15 , PERMEABILITY_XY=2)

        two_phase_formulation_3D = TwoPhaseSoil(ndim=3, DENSITY_SOLID=2650, POROSITY=0.3,
                                                PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15 ,
                                                PERMEABILITY_ZZ=1E-15, PERMEABILITY_XY=1, PERMEABILITY_ZX=2,
                                                PERMEABILITY_YZ=3)

        # Define two phase constitutive law parameters
        two_phase_constitutive_parameters = LinearElasticSoil(YOUNG_MODULUS=1E9, POISSON_RATIO=0.3)
        two_phase_retention_parameters = SaturatedBelowPhreaticLevelLaw()


        # Create materials
        umat_material = SoilMaterial(name="test_umat_material", soil_formulation=umat_formulation,
                                     constitutive_law=umat_constitutive_parameters,
                                     retention_parameters=umat_retention_parameters)

        udsm_material = SoilMaterial( name="test_udsm_material", soil_formulation=udsm_formulation,
                                      constitutive_law=udsm_constitutive_parameters,
                                      retention_parameters=udsm_retention_parameters)

        two_phase_material_2D = SoilMaterial(name="test_two_phase_material_2D", soil_formulation=two_phase_formulation_2D,
                                                constitutive_law=two_phase_constitutive_parameters,
                                                retention_parameters=two_phase_retention_parameters)

        two_phase_material_3D = SoilMaterial(name="test_two_phase_material_3D", soil_formulation=two_phase_formulation_3D,
                                                constitutive_law=two_phase_constitutive_parameters,
                                                retention_parameters=two_phase_retention_parameters)


        all_materials = [umat_material, udsm_material, two_phase_material_2D, two_phase_material_3D]

        # write json file
        kratos_io = KratosIO(ndim=3)
        kratos_io.write_material_parameters_json(all_materials, "test_write_MaterialParameters.json")

        # read generated json file and expected json file
        written_material_parameters_json = json.load(open("test_write_MaterialParameters.json"))
        expected_material_parameters_json = json.load(open("tests/test_data/expected_material_parameters.json"))

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)

    def test_raise_errors_for_material_parameters(self):
        """
        Tests that errors are raised when the soil material parameters are not valid.

        """

        # create 3d two phase soil without zz permeability, which is not allowed
        pytest.raises(ValueError, TwoPhaseSoil, ndim=3, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15 , PERMEABILITY_XY=1, PERMEABILITY_ZX=2,
                      PERMEABILITY_YZ=3)

        # create 3d two phase soil without yz permeability, which is not allowed
        pytest.raises(ValueError, TwoPhaseSoil, ndim=3, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15, PERMEABILITY_ZX=1, PERMEABILITY_ZZ=2,
                      PERMEABILITY_YZ=None)

        # create 3d two phase soil without zx permeability, which is not allowed
        pytest.raises(ValueError, TwoPhaseSoil, ndim=3, DENSITY_SOLID=2650, POROSITY=0.3, BULK_MODULUS_SOLID=1E9,
                      PERMEABILITY_XX=1E-15, PERMEABILITY_YY=1E-15 , PERMEABILITY_XY=1, PERMEABILITY_ZX=None,
                      PERMEABILITY_ZZ=3)

        # create 3d euler beam without I22, which is not allowed
        pytest.raises(ValueError, EulerBeam, ndim=3, DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2, CROSS_AREA=1.0,
                      I33=1, TORSIONAL_INERTIA=1)

        # create 3d euler beam without torsional inertia, which is not allowed
        pytest.raises(ValueError, EulerBeam, ndim=3, DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2, CROSS_AREA=1.0,
                      I33=1, I22=1)



    def test_write_structural_material_to_json(self):
        """
        Test writing a material list to json. In this test, the material list contains a beam material, a spring damper
        material and a nodal concentrated material.

        """

        # define ndim
        ndim=2

        # define euler beam parameters
        beam_material_parameters = EulerBeam(ndim=ndim, DENSITY=1.0, YOUNG_MODULUS=1.0, POISSON_RATIO=0.2, CROSS_AREA=1.0, I33=1)

        # define spring damper parameters
        spring_damper_material_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1.0, 2, 3],
                                                                NODAL_DAMPING_COEFFICIENT=[0, 0.2, 3],
                                                                NODAL_ROTATIONAL_STIFFNESS=[2.0, 4, 5],
                                                                NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 9])

        # define nodal concentrated parameters
        nodal_concentrated_material_parameters = NodalConcentrated( NODAL_MASS=1.0, NODAL_DAMPING_COEFFICIENT=[1,2,0.2],
                                                                    NODAL_DISPLACEMENT_STIFFNESS=[1, 2, 3])

        # Create structural materials
        beam_material = StructuralMaterial(name="test_beam_material", material_parameters=beam_material_parameters)
        spring_damper_material = StructuralMaterial(name="test_spring_damper_material",
                                          material_parameters=spring_damper_material_parameters)
        nodal_concentrated_material = StructuralMaterial(name="test_nodal_concentrated_material",
                                               material_parameters=nodal_concentrated_material_parameters)

        all_materials = [beam_material, spring_damper_material, nodal_concentrated_material]

        # write json file
        kratos_io = KratosIO(ndim=ndim)
        kratos_io.write_material_parameters_json(all_materials, "test_write_structural_MaterialParameters.json")

        # read generated json file and expected json file
        written_material_parameters_json = json.load(open("test_write_structural_MaterialParameters.json"))
        expected_material_parameters_json = json.load(open("tests/test_data/expected_structural_material_parameters.json"))

        # compare json files using custom dictionary comparison
        TestUtils.assert_dictionary_almost_equal(written_material_parameters_json, expected_material_parameters_json)

    def test_create_load_process_dictionary(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file
        """
        # define load(s) parameters
        # point load
        point_load_parameters = PointLoad(
            active=[True, False, True], value=[1000, 0, 0]
        )

        # moving (point) load
        moving_point_load_parameters = MovingLoad(
            origin=[0.0, 1.0, 2.0],
            load=[0.0, -10.0, 0.0],
            direction=[1.0, 0.0, -1.0],
            velocity=5.0,
            offset=3.0
        )

        # create Load objects and store in the list
        point_load = Load(name="test_name", load_parameters=point_load_parameters)

        moving_point_load = Load(
            name="test_name_moving", load_parameters=moving_point_load_parameters
        )

        all_loads = [point_load, moving_point_load]

        # write dictionary for the load(s)
        kratos_io = KratosIO(ndim=2)
        test_dictionary = kratos_io.create_loads_process_dictionary(all_loads)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )
