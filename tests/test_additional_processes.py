import pytest
from random_fields.generate_field import RandomFields, ModelName

from stem.additional_processes import ParameterFieldParameters


class TestAdditionalProcesses:

    def test_correct_field_parameter_creation(self):
        """
        Tests that field parameter is created correctly and raises correctly errors.
        """

        # Raise error if random field generator is None for json_type function

        with pytest.raises(ValueError):

            field_parameters_json = ParameterFieldParameters(
                variable_name="YOUNG_MODULUS",
                function_type="json_file",
                function="test_random_field_json",
                field_generator=None
            )

        # Raise error if random field values have not been initialised
        random_field_generator = RandomFields(n_dim=3, mean=10, variance=2,
                                              model_name=ModelName.Gaussian,
                                              v_scale_fluctuation=5,
                                              anisotropy=[0.5, 0.5], angle=[0, 0], seed=42, v_dim=1)
        field_parameters_json = ParameterFieldParameters(
            variable_name="YOUNG_MODULUS",
            function_type="json_file",
            function="test_random_field_json",
            field_generator=random_field_generator
        )
        with pytest.raises(ValueError):
            values = field_parameters_json.values
