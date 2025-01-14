import numpy as np
import numpy.testing as npt
import pytest

from stem.additional_processes import ParameterFieldParameters
from stem.field_generator import RandomFieldGenerator


class TestAdditionalProcesses:

    def test_random_field_generator(self):
        """
        Tests that random field is generated correctly.

        """
        # Create a valid random field generator
        random_field_generator = RandomFieldGenerator(cov=0.1,
                                                      model_name="Gaussian",
                                                      v_scale_fluctuation=1,
                                                      anisotropy=[0.5],
                                                      angle=[0],
                                                      seed=42,
                                                      mean_value=10)

        field_parameters_object = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                           function_type="json_file",
                                                           field_file_names=["test_random_field_json"],
                                                           field_generator=random_field_generator)

        # Generate random field
        field_parameters_object.field_generator.generate(coordinates=np.array([(0, 1, 0), (0, 5, 0)]))
        actual_parameters = field_parameters_object.field_generator.generated_fields[0]
        expected_parameters = [10.765957, 10.345959]

        # assert if the generated field is correct
        npt.assert_allclose(actual_parameters, expected_parameters)

    def test_random_field_generator_expected_errors(self):
        """
        Tests if the parameter field raises errors correctly.
        """

        # Raise error if random field generator is None for json_type function
        msg = ("`field_generator` parameter is a required when `json_file` field parameter is "
               "selected for `function_type`.")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                     function_type="json_file",
                                     field_file_names=["test_random_field_json"],
                                     field_generator=None)

        # Raise error if values are asked but generator is None (from python or input function type)
        msg = ("ParameterField Error:`function_type` is not understood: python."
               r"Should be one of \['json_file', 'input'\].")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                     function_type="python",
                                     field_file_names=None,
                                     field_generator=None)

        # Raise error if values are asked but generator is None (from python or input function type)
        msg = ("`tiny_expr_function` parameter is a required when `input` field parameter is "
               "selected for `function_type`.")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(property_names=["YOUNG_MODULUS"], function_type="input")

        # Raise error if field_file_names is not the same length as property_names
        msg = "`field_file_names` should have the same length as `property_names`."
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(property_names=["YOUNG_MODULUS", "POISSON_RATIO"],
                                     function_type="json_file",
                                     field_generator=RandomFieldGenerator(cov=0.1,
                                                                          model_name="Gaussian",
                                                                          v_scale_fluctuation=1,
                                                                          anisotropy=[0.5],
                                                                          angle=[0],
                                                                          seed=42,
                                                                          mean_value=10),
                                     field_file_names=["test_random_field_json"])

        msg = ("Only one property name can be provided for the field generator class "
               "'RandomFieldGenerator'.")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(property_names=["YOUNG_MODULUS", "POISSON_RATIO"],
                                     function_type="json_file",
                                     field_file_names=["json_file1.json", "json_file2.json"],
                                     field_generator=RandomFieldGenerator(cov=0.1,
                                                                          model_name="Gaussian",
                                                                          v_scale_fluctuation=1,
                                                                          anisotropy=[0.5],
                                                                          angle=[0],
                                                                          seed=42,
                                                                          mean_value=10))
