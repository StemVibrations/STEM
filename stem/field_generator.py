from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union

import numpy as np
import numpy.typing as npty
from random_fields.generate_field import RandomFields, ModelName

from stem.globals import VERTICAL_AXIS

AVAILABLE_RANDOM_FIELD_MODEL_NAMES = ["Gaussian", "Exponential", "Matern", "Linear"]


class FieldGeneratorABC(ABC):
    """
    Abstract class to generate fields as function of points coordinates (x, y and z).
    The function should implement a `generate` method to initialise the field and the `values`
    property to retrieve the generated field.

    """

    @abstractmethod
    def generate(self, coordinates: npty.NDArray[np.float64]):
        """
        Abstract method to generate the fields for the required coordinates.
        It has to set the generated_field attribute.

        Args:
            - coordinates (numpy.typing.NDArray[np.float64]): Sequence of points where the field needs to be generated.

        Raises:
            - Exception: abstract class of generate is called

        """
        raise Exception("abstract class of generate is called")

    @property
    @abstractmethod
    def generated_field(self) -> Optional[List[Any]]:
        """
        Abstract property of the generated field.

        Raises:
            - ValueError: if field is not generated using the `generate()` method

        Returns:
            - Optional[list[Any]]: the list of generated values for the field.

        """

        raise Exception("abstract class of generated_field is called")


class RandomFieldGenerator(FieldGeneratorABC):
    """
    Class to generate random fields for a material property in the model as funtion of the coordinates
    of the centroid of the elements (x, y and z).

    Inheritance:
        - :class:`FieldGeneratorABC`: abstract class to generate fields as function of points coordinates (x, y and z).

    Attributes:
        - __generated_field (Optional[List[float]]): The generated field values. Defaults to None.
        - model_name (str): Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
        - n_dim (int): number of dimensions of the model (2 or 3).
        - cov (float): The coefficient of variation of the random field.
        - v_scale_fluctuation (float): The vertical scale of fluctuation of the random field.
        - anisotropy (list): The anisotropy of the random field in the other directions (per dimension).
        - angle (list): The angle of the random field (per dimension).
        - mean_value (Optional[float]): mean value of the random field. Defaults to None. \
            In that case it should be set otherwise before running the generate method.
        - seed (int): The seed number for the random number generator.

    """

    def __init__(self, model_name: str,
                 n_dim: int,
                 cov: float,
                 v_scale_fluctuation: float,
                 anisotropy: Union[float, List[float]],
                 angle: Union[float, List[float]],
                 mean_value: Optional[Union[int, float]] = None,
                 seed: int = 14):
        """
        Initialise a random generator field. The mean value is optional because it can be set at another moment.
        In that case it should be set before running the generate method.

        Anisotropy and angle can be given as scalar, 1-D and 2-D lists. In case the model is 3D but a 1-D or scalar
        is provided, it is assumed the same angle and anisotropy along both horizontal direction.

        Args:
            - model_name (str): Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
            - n_dim (int): number of dimensions of the model (2 or 3).
            - cov (float): The coefficient of variation of the random field.
            - v_scale_fluctuation (float): The vertical scale of fluctuation of the random field.
            - anisotropy (list): The anisotropy of the random field in the other directions (per dimension).
            - angle (list): The angle of the random field (per dimension).
            - mean_value (Optional[float]): mean value of the random field. Defaults to None. \
                In that case it should be set otherwise before running the generate method.
            - seed (int): The seed number for the random number generator.

        Raises:
            - ValueError: if the model dimensions is not 2 or 3.
            - ValueError: if the model_name is not a valid or implemented model.

        """
        # validate the number of dimensions of the model
        if n_dim not in [2, 3]:
            raise ValueError(f"Number of dimension {n_dim} specified, but should be one of either 2 or 3.")

        # check that random field model is one of the implemented
        if model_name not in AVAILABLE_RANDOM_FIELD_MODEL_NAMES:
            raise ValueError(f"Model name: `{model_name}` was provided but not understood or implemented yet. "
                             f"Available models are: {AVAILABLE_RANDOM_FIELD_MODEL_NAMES}")

        # if anisotropy or angle are float, convert to list
        if isinstance(anisotropy, float):
            anisotropy = [anisotropy]
        if isinstance(angle, float):
            angle = [angle]

        # if angle or anisotropy are 1-D list but model is 3-D duplicate them
        if n_dim == 3:
            anisotropy = anisotropy if len(anisotropy) == 2 else [anisotropy[0], anisotropy[0]]
            angle = angle if len(angle) == 2 else [angle[0], angle[0]]

        self.__generated_field: Optional[List[float]] = None
        self.model_name = model_name
        self.n_dim = n_dim
        self.cov = cov
        self.v_scale_fluctuation = v_scale_fluctuation
        self.anisotropy = anisotropy
        self.angle = angle
        self.mean_value = mean_value
        self.seed = seed

    @property
    def generated_field(self) -> Optional[List[Any]]:
        """
        Returns the value of the generated field.

        Raises:
            - ValueError: if field is not generated using the `generate()` method

        Returns:
            - Optional[list[Any]]: the list of generated values for the field.

        """

        if self.__generated_field is None:
            raise ValueError("Field is not generated yet.")

        return self.__generated_field

    def generate(self, coordinates: npty.NDArray[np.float64]):
        """
        Generate the random field parameters at the coordinates specified.
        The generated values are stored in `generated_field` attribute.

        Args:
            - coordinates (numpy.typing.NDArray[np.float64]): Sequence of points where the random field needs to be
            generated.

        Raises:
            - ValueError: if the mean value of the random field is undefined.

        """

        if self.mean_value is None:
            raise ValueError("The mean value of the random field is not set yet. Error.")

        variance = (self.cov * self.mean_value) ** 2

        rf_generator = RandomFields(
            n_dim=self.n_dim, mean=self.mean_value, variance=variance,
            model_name=ModelName[self.model_name],
            v_scale_fluctuation=self.v_scale_fluctuation,
            anisotropy=self.anisotropy,
            angle=self.angle,
            seed=self.seed,
            v_dim=VERTICAL_AXIS
        )
        coordinates_for_rf = np.array(coordinates)
        rf_generator.generate(coordinates_for_rf)
        self.__generated_field = list(rf_generator.random_field)[0].tolist()
