import json
from typing import Dict, Any

import pytest

from random_fields.generate_field import ModelName

from stem.IO.kratos_additional_processes_io import KratosAdditionalProcessesIO
from stem.additional_processes import *
from stem.load import PointLoad
from tests.utils import TestUtils


class TestKratosAdditionalProcessesIO:

    def test_create_additional_processes_dictionaries(self):
        """
        Test the creation of the additional processes dictionaries for the
        ProjectParameters.json file
        """

        # Excavation process
        excavation_parameters = Excavation(deactivate_body_model_part=True)

        # Define random field generator

        random_field_generator = RandomFields(n_dim=3, mean=10, variance=2,
                                              model_name=ModelName.Gaussian,
                                              v_scale_fluctuation=5,
                                              anisotropy=[0.5, 0.5], angle=[0, 0], seed=42, v_dim=1)

        # define the field parameters
        field_parameters_json = ParameterFieldParameters(
            variable_name="YOUNG_MODULUS",
            function_type="json_file",
            function="test_random_field_json",
            field_generator=random_field_generator
        )

        field_parameters_python = ParameterFieldParameters(
            variable_name="YOUNG_MODULUS",
            function_type="python",
            function="test_random_field_python",
            field_generator=random_field_generator
        )

        field_parameters_input = ParameterFieldParameters(
            variable_name="YOUNG_MODULUS",
            function_type="input",
            function="20000*x + 30000*y",
            field_generator=random_field_generator
        )
        # collect the part names and parameters into a dictionary
        # TODO: change later when model part is implemented
        all_parameters = {
            "test_excavation": excavation_parameters,
            "test_random_field_json": field_parameters_json,
            "test_random_field_python": field_parameters_python,
            "test_random_field_input": field_parameters_input
        }

        # initialize process dictionary
        test_dictionary: Dict[str, Any] = {
            "processes": {"constraints_process_list": []}
        }

        # write dictionary for the boundary(/ies)
        add_processes_io = KratosAdditionalProcessesIO(domain="PorousDomain")
        # TODO: when model part are implemented, generate file through kratos_io

        for part_name, part_parameters in all_parameters.items():
            _parameters = add_processes_io.create_additional_processes_dict(
                part_name=part_name, parameters=part_parameters
            )
            test_dictionary["processes"]["constraints_process_list"].append(_parameters)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_additional_processes_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )

    def test_raise_errors_additional_processes_io(self):
        """Test that the additional_processes_io raises correctly the errors
        """


        # Define random field generator
        random_field_generator = RandomFields(n_dim=3, mean=10, variance=2,
                                              model_name=ModelName.Gaussian,
                                              v_scale_fluctuation=5,
                                              anisotropy=[0.5, 0.5], angle=[0, 0], seed=42, v_dim=1)

        # define the field parameters
        field_parameters_json = ParameterFieldParameters(
            variable_name="YOUNG_MODULUS",
            function_type="json_file",
            function="test_random_field_json",
            field_generator=random_field_generator
        )

        field_parameters_json.function_type = "csv"
        add_processes_io = KratosAdditionalProcessesIO(domain="PorousDomain")

        # Function type is not allowed
        with pytest.raises(ValueError):
            _parameters = add_processes_io.create_additional_processes_dict(
                part_name="test", parameters=field_parameters_json
            )

        load_parameters = PointLoad(value=[0, 1, 0], active=[True, True, True])

        # Wrong parameters type (load)
        with pytest.raises(NotImplementedError):
            _parameters = add_processes_io.create_additional_processes_dict(
                part_name="test", parameters=load_parameters
            )
