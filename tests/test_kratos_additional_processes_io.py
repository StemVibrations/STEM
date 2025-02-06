import json
from typing import Dict, Any
import re

import pytest

from stem.IO.kratos_additional_processes_io import KratosAdditionalProcessesIO
from stem.additional_processes import Excavation, RandomFieldGenerator, ParameterFieldParameters, HingeParameters
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

        # Define the field generator
        random_field_generator = RandomFieldGenerator(cov=0.1,
                                                      model_name="Gaussian",
                                                      v_scale_fluctuation=5,
                                                      anisotropy=[0.5, 0.5],
                                                      angle=[0, 0],
                                                      seed=42)

        # add field via json file
        field_parameters_json = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                         function_type="json_file",
                                                         field_file_names=["json_file.json"],
                                                         field_generator=random_field_generator)

        # add field via tiny expression
        field_parameters_input = ParameterFieldParameters(
            property_names=["YOUNG_MODULUS"],
            function_type="input",
            tiny_expr_function="20000*x + 30000*y",
        )
        # collect the part names and parameters into a dictionary
        # TODO: change later when model part is implemented
        all_parameters = {
            "test_excavation": excavation_parameters,
            "test_random_field_json": field_parameters_json,
            "test_random_field_input": field_parameters_input
        }

        # initialize process dictionary
        test_dictionary: Dict[str, Any] = {"processes": {"constraints_process_list": []}}

        # write dictionary for the boundary(/ies)
        add_processes_io = KratosAdditionalProcessesIO(domain="PorousDomain")

        for part_name, part_parameters in all_parameters.items():
            _parameters = add_processes_io.create_additional_processes_dict(part_name=part_name,
                                                                            parameters=part_parameters)
            test_dictionary["processes"]["constraints_process_list"].extend(_parameters)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(open("tests/test_data/expected_additional_processes_parameters.json"))

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(test_dictionary, expected_load_parameters_json)

    def test_raise_errors_additional_processes_io(self):
        """
        Test that the additional_processes_io raises correctly the errors

        """

        # Define the field generator
        random_field_generator = RandomFieldGenerator(cov=0.1,
                                                      model_name="Gaussian",
                                                      v_scale_fluctuation=5,
                                                      anisotropy=[0.5, 0.5],
                                                      angle=[0, 0],
                                                      seed=42)

        # define the field parameters
        field_parameters_json = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                         function_type="json_file",
                                                         field_file_names=["test_random_field_json"],
                                                         field_generator=random_field_generator)

        field_parameters_json.function_type = "csv"
        add_processes_io = KratosAdditionalProcessesIO(domain="PorousDomain")

        # Function type is not allowed
        with pytest.raises(ValueError):
            _parameters = add_processes_io.create_additional_processes_dict(part_name="test",
                                                                            parameters=field_parameters_json)

        load_parameters = PointLoad(value=[0, 1, 0], active=[True, True, True])

        # Wrong parameters type (load)
        with pytest.raises(NotImplementedError):
            _parameters = add_processes_io.create_additional_processes_dict(part_name="test",
                                                                            parameters=load_parameters)

        # Test Raise empty field_file_names with json function type
        field_parameters_json.function_type = "json_file"
        field_parameters_json.field_file_names = [""]
        message = "`field_file_names` should be provided when `json_file` function type is selected."
        with pytest.raises(ValueError, match=re.escape(message)):
            add_processes_io.create_additional_processes_dict(part_name="test", parameters=field_parameters_json)

    def test_create_hinge_dict(self):
        """
        Test the creation of the hinge dictionary for the ProjectParameters.json file

        """

        # Define the hinge parameters
        hinge_parameters = HingeParameters(ROTATIONAL_STIFFNESS_AXIS_2=14, ROTATIONAL_STIFFNESS_AXIS_3=13)

        # write dictionary for the boundary(/ies)
        additional_processes_io = KratosAdditionalProcessesIO(domain="PorousDomain")

        # create the hinge dictionary
        generated_dicts = additional_processes_io.create_additional_processes_dict(part_name="hinge",
                                                                                   parameters=hinge_parameters)

        expected_dicts = [{
            "python_module": "assign_scalar_variable_to_nodes_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignScalarVariableToNodesProcess",
            "Parameters": {
                "model_part_name": "PorousDomain.hinge",
                "variable_name": "ROTATIONAL_STIFFNESS_AXIS_2",
                "value": 14
            }
        }, {
            "python_module": "assign_scalar_variable_to_nodes_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignScalarVariableToNodesProcess",
            "Parameters": {
                "model_part_name": "PorousDomain.hinge",
                "variable_name": "ROTATIONAL_STIFFNESS_AXIS_3",
                "value": 13
            }
        }]

        # assert the objects to be equal
        assert len(generated_dicts) == len(expected_dicts) == 2
        for generated_dict, expected_dict in zip(generated_dicts, expected_dicts):
            TestUtils.assert_dictionary_almost_equal(generated_dict, expected_dict)
