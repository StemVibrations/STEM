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

        # Absorbing boundaries
        absorbing_boundaries_parameters = AbsorbingBoundary(
            absorbing_factors=[1.0, 1.0], virtual_thickness=1000.0
        )

        # collect the part names and parameters into a dictionary
        # TODO: change later when model part is implemented
        all_boundary_parameters = {
            "test_displacement_constraint":fix_displacements_parameters,
            "test_rotation_constraint":fix_rotations_parameters,
            "test_absorbing_boundaries":absorbing_boundaries_parameters
        }

        # initialize process dictionary
        test_dictionary: Dict[str, Any] = {
            "processes": {"constraints_process_list": [], "loads_process_list": []}
        }

        # write dictionary for the boundary(/ies)
        boundaries_io = KratosBoundariesIO(domain="PorousDomain")
        # TODO: when model part are implemented, generate file through kratos_io

        for part_name, part_parameters in all_boundary_parameters.items():
            _parameters = boundaries_io.create_boundary_condition_dict(
                part_name=part_name, parameters=part_parameters
            )
            _key = "loads_process_list"
            if part_parameters.is_constraint:
                _key = "constraints_process_list"
            test_dictionary["processes"][_key].append(_parameters)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_boundary_conditions_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            expected_load_parameters_json, test_dictionary
        )