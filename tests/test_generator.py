import numpy.testing as npt
import pytest

from stem.field_generator import RandomFieldGenerator


class TestGenerators:

    def test_random_field_generator(self):
        """
        Tests that the random field generator initialises correctly and generates the expected parameters.
        """

        # Raise error when model for random field is not understood.
        msg = (r"Model name: `Gauss` was provided but not understood or implemented yet. "
               r"Available models are: \['Gaussian', 'Exponential', 'Matern', 'Linear'\]")
        with pytest.raises(ValueError, match=msg):
            # Raise error if random field values have not been initialised
            RandomFieldGenerator(cov=0.1,
                                 model_name="Gauss",
                                 v_scale_fluctuation=5,
                                 anisotropy=[0.5, 0.5],
                                 angle=[0, 0],
                                 seed=42)

        # Raise error when anisotropy has more than 2 elements.
        msg = ("Anisotropy has to be a float, or a list of either 1 or 2 floats.")
        with pytest.raises(ValueError, match=msg):
            RandomFieldGenerator(cov=0.1,
                                 model_name="Gaussian",
                                 v_scale_fluctuation=5,
                                 anisotropy=[0.5, 0.5, 20],
                                 angle=[0, 0],
                                 seed=42)

        # Raise error when anisotropy has more than 2 elements.
        msg = ("Angle has to be a float, or a list of either 1 or 2 floats")
        with pytest.raises(ValueError, match=msg):
            RandomFieldGenerator(cov=0.1,
                                 model_name="Gaussian",
                                 v_scale_fluctuation=5,
                                 anisotropy=[0.5, 0.5],
                                 angle=[0, 0, 1],
                                 seed=42)

        # Raise error if random field values have not been initialised
        random_field_generator = RandomFieldGenerator(cov=0.1,
                                                      model_name="Gaussian",
                                                      v_scale_fluctuation=5,
                                                      anisotropy=[0.5, 0.5],
                                                      angle=[0, 0],
                                                      seed=42)
        msg = "Field is not generated yet."
        with pytest.raises(ValueError, match=msg):
            values = random_field_generator.generated_fields[0]

        msg = "The mean value of the random field is not set yet. Error."
        with pytest.raises(ValueError, match=msg):
            random_field_generator.generate([(0, 0, 0), (10, 1, 0)])

        random_field_generator.mean_value = 10
        random_field_generator.generate([(0, 0, 0), (10, 1, 0)])
        values = random_field_generator.generated_fields[0]
        npt.assert_allclose(values, [12.379497, 10.593184])
