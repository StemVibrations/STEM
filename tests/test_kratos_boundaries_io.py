import json

import pytest

from stem.IO.kratos_boundaries_io import KratosBoundariesIO
from stem.boundary import *
from tests.utils import TestUtils


class TestKratosBoundariesIO:

    def test_create_boundary_condition_dictionaries(self):
        """
        Test the creation of the boundary condition dictionaries for the
        ProjectParameters.json file
        """
        # define constraints

        # Displacements
        fix_displacements_parameters = DisplacementConstraint(
            active=[True, True, False],
            is_fixed=[True, True, False],
            value=[0.0, 0.0, 0.0],
        )

        # Rotations
        fix_rotations_parameters = RotationConstraint(
            active=[False, False, True],
            is_fixed=[False, False, True],
            value=[0.0, 0.0, 0.0],
        )

        # Boundary conditions
        absorbing_boundaries_parameters = AbsorbingBoundary(
            absorbing_factors=[1.0, 1.0], virtual_thickness=1000.0
        )

        # create Load objects and store in the list
        displacement_boundary_condition = Boundary(
            part_name="test_displacement_constraint",
            boundary_parameters=fix_displacements_parameters,
        )
        rotation_boundary_condition = Boundary(
            part_name="test_rotation_constraint",
            boundary_parameters=fix_rotations_parameters,
        )

        absorbing_boundary = Boundary(
            part_name="abs",
            boundary_parameters=absorbing_boundaries_parameters,
        )
        all_outputs = [
            displacement_boundary_condition,
            rotation_boundary_condition,
            absorbing_boundary
        ]

        # write dictionary for the output(s)
        kratos_io = KratosBoundariesIO(domain="PorousDomain")
        (
            test_constraint_dictionary,
            test_absorbing_bound_list
        ) = kratos_io.create_dictionaries_for_boundaries(all_outputs)

        # nest the json into the process dictionary, as it should!
        test_constraint_dictionary["loads_process_list"] = test_absorbing_bound_list
        test_dictionary = {"processes": test_constraint_dictionary}
        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_boundary_conditions_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )