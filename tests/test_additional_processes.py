import pytest
from random_fields.generate_field import RandomFields, ModelName

from stem.additional_processes import ParameterFieldParameters
from stem.field_generator import RandomFieldGenerator


class TestAdditionalProcesses:

    def test_correct_field_parameter_creation(self):
        """
        Tests that field parameter is created correctly and raises correctly errors.
        """

        # Raise error if random field generator is None for json_type function
        msg = (f"Field generator object is required to produce the json file parameters!")
        with pytest.raises(ValueError, match=msg):
            field_parameters_json = ParameterFieldParameters(
                variable_name="YOUNG_MODULUS",
                function_type="json_file",
                function="test_random_field_json",
                field_generator=None
            )

        # Raise error if values are asked but generator is None (from python or input function type)
        msg = (f"ParameterField Error:\n`function_type` is not understood: python.\n"
               f"Should be one of ['json_file', 'input'].")
        with pytest.raises(ValueError):
            field_parameters_python = ParameterFieldParameters(
                variable_name="YOUNG_MODULUS",
                function_type="python",
                function="test_random_field_python.py",
                field_generator=None
            )

        # Raise error if random field values have not been initialised
        random_field_generator = RandomFieldGenerator(
            n_dim=3, mean=10, variance=2, model_name="Gaussian",
            v_scale_fluctuation=5, anisotropy=[0.5, 0.5], angle=[0, 0], seed=42, v_dim=1
        )
        field_parameters_json = ParameterFieldParameters(
            variable_name="YOUNG_MODULUS",
            function_type="json_file",
            function="test_random_field_json",
            field_generator=random_field_generator
        )

        # values are not generated yet!
        msg = "Values for field parameters are not generated yet."
        with pytest.raises(ValueError, match=msg):
            values = field_parameters_json.field_generator.values

        field_parameters_json.field_generator.generate([(0, 0, 0), (1, 1, 0)])


