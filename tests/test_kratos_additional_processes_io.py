import json

import pytest

from stem.IO.kratos_additional_processes_io import KratosAdditionalProcessesIO
from stem.additional_processes import *
from tests.utils import TestUtils


class TestKratosBoundariesIO:

    def test_create_boundary_condition_dictionaries(self):
        """
        Test the creation of the boundary condition dictionaries for the
        ProjectParameters.json file
        """
        # define constraints

        # Absorbing boundaries
        excavation_parameters = Excavation(deactivate_soil_part=True)

        # collect the part names and parameters into a dictionary
        # TODO: change later when model part is implemented
        all_parameters = {
            "test_excavation": excavation_parameters,
        }

        # initialize process dictionary
        test_dictionary: Dict[str, Any] = {
            "processes": {"constraints_process_list": [], "loads_process_list": []}
        }

        # write dictionary for the boundary(/ies)
        add_processes_io = KratosAdditionalProcessesIO(domain="PorousDomain")
        # TODO: when model part are implemented, generate file through kratos_io

        for part_name, part_parameters in all_parameters.items():
            _parameters = add_processes_io.create_additional_processes_dict(
                part_name=part_name, parameters=part_parameters
            )
            _key = "loads_process_list"
            if isinstance(part_parameters, Excavation):
                _key = "constraints_process_list"
            test_dictionary["processes"][_key].append(_parameters)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_additional_processes_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            expected_load_parameters_json, test_dictionary
        )