import numpy.testing as npt
import pytest

from stem.additional_processes import ParameterFieldParameters


class TestAdditionalProcesses:

    def test_correct_field_parameter_creation(self):
        """
        Tests that field parameter is created correctly and raises correctly errors.
        """

        # Raise error if random field generator is None for json_type function
        msg = ("`field_generator` parameter is a required when `json_file` field parameter is "
               "selected for `function_type`.")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(
                property_name="YOUNG_MODULUS",
                function_type="json_file",
                field_file_name="test_random_field_json",
                field_generator=None
            )

        # Raise error if values are asked but generator is None (from python or input function type)
        msg = ("ParameterField Error:`function_type` is not understood: python."
               r"Should be one of \['json_file', 'input'\].")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(
                property_name="YOUNG_MODULUS",
                function_type="python",
                field_file_name=None,
                field_generator=None
            )

        # Raise error if values are asked but generator is None (from python or input function type)
        msg = ("`tiny_expr_function` parameter is a required when `input` field parameter is "
               "selected for `function_type`.")
        with pytest.raises(ValueError, match=msg):
            ParameterFieldParameters(
                property_name="YOUNG_MODULUS",
                function_type="input"
            )

