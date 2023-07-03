import json

from stem.IO.kratos_io import KratosIO
from stem.boundary import *
from stem.model import Model
from stem.model_part import ModelPart
from tests.utils import TestUtils


class TestKratosBoundariesIO:
    def test_create_boundary_condition_dictionaries(self):
        """
        Test the creation of the boundary condition dictionaries for the
        ProjectParameters.json file
        """
        # define constraints

        # Displacements
        mp_fix_displacements = ModelPart(
            name="test_displacement_constraint",
            parameters=DisplacementConstraint(
                active=[True, True, False],
                is_fixed=[True, True, False],
                value=[0.0, 0.0, 0.0],
            ),
        )

        # Rotations
        mp_fix_rotations = ModelPart(
            name="test_rotation_constraint",
            parameters=RotationConstraint(
                active=[False, False, True],
                is_fixed=[False, False, True],
                value=[0.0, 0.0, 0.0],
            ),
        )
        # Absorbing boundaries
        mp_absorbing_boundaries = ModelPart(
            name="abs",
            parameters=AbsorbingBoundary(
                absorbing_factors=[1.0, 1.0], virtual_thickness=1000.0
            ),
        )

        # collect model parts together
        model_parts = [mp_fix_displacements, mp_fix_rotations, mp_absorbing_boundaries]

        # write dictionary for the boundary condition(s)
        kratos_io = KratosIO(ndim=2, model=Model(ndim=2, model_parts=model_parts))
        # no inputs write no json file!
        test_boundaries_dict = kratos_io.write_project_parameters_json()

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_boundary_conditions_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_boundaries_dict, expected_load_parameters_json
        )
