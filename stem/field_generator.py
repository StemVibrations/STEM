from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union

import numpy as np
import numpy.typing as npty
from random_fields.generate_field import RandomFields, ModelName
from random_fields.geostatistical_cpt_interpretation import ElasticityFieldsFromCpt, RandomFieldProperties

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
    def generated_fields(self) -> Optional[List[List[Any]]]:
        """
        Abstract property of the generated field.

        Raises:
            - ValueError: if field is not generated using the `generate()` method

        Returns:
            - Optional[List[List[Any]]]: the list of generated values for the fields.

        """

        raise Exception("abstract class of generated_fields is called")


class RandomFieldGenerator(FieldGeneratorABC):
    """
    Class to generate random fields for a material property in the model as function of the coordinates
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

    def __init__(self,
                 model_name: str,
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
        Because the models in STEM always have coordinates in three dimensions (x, y, and z), random fields always have
        a dimension (n_dim) equal to 3.

        Args:
            - model_name (str): Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
            - cov (float): The coefficient of variation of the random field.
            - v_scale_fluctuation (float): The vertical scale of fluctuation of the random field.
            - anisotropy (Union[float, List[float]]): The anisotropy of the random field in the other directions \
                (per dimension). Either a float, or a list of float containing 1 or 2 elements is accepted.
            - angle (Union[float, List[float]]): The angle of the random field (per dimension). \
                Either a float, or a list of float containing 1 or 2 elements is accepted.
            - mean_value (Optional[float]): mean value of the random field. Defaults to None. \
                In that case it should be set otherwise before running the generate method.
            - seed (int): The seed number for the random number generator.

        Raises:
            - ValueError: if the model_name is not a valid or implemented model.
            - ValueError: if the `anisotropy` has more than 2 elements.
            - ValueError: if the `angle` has more than 2 elements.

        """

        # check that random field model is one of the implemented
        if model_name not in AVAILABLE_RANDOM_FIELD_MODEL_NAMES:
            raise ValueError(f"Model name: `{model_name}` was provided but not understood or implemented yet. "
                             f"Available models are: {AVAILABLE_RANDOM_FIELD_MODEL_NAMES}")

        # if anisotropy or angle are not a list make a list out of them
        if not isinstance(anisotropy, list):
            anisotropy = [anisotropy]
        if not isinstance(angle, list):
            angle = [angle]

        # validate the inputs for anisotropy and angle that control the 3d effects of the field
        if len(anisotropy) not in [1, 2]:
            raise ValueError("Anisotropy has to be a float, or a list of either 1 or 2 floats.")
        if len(angle) not in [1, 2]:
            raise ValueError("Angle has to be a float, or a list of either 1 or 2 floats.")

        # if angle or anisotropy are 1-D list duplicate them in the 3rd dimension.
        # for 2d models this will have no effect, for 3d models it will make a radial symmetry of the field.
        anisotropy = anisotropy if len(anisotropy) == 2 else [anisotropy[0], anisotropy[0]]
        angle = angle if len(angle) == 2 else [angle[0], angle[0]]

        self.__generated_field: Optional[List[float]] = None
        self.model_name = model_name
        self.__n_dim = 3  # stem coordinates are always 3 even for a 2D model wit the third one being irrelevant.
        self.cov = cov
        self.v_scale_fluctuation = v_scale_fluctuation
        self.anisotropy = anisotropy
        self.angle = angle
        self.mean_value = mean_value
        self.seed = seed

    @property
    def generated_fields(self) -> Optional[List[Any]]:
        """
        Returns the value of the generated field.

        Raises:
            - ValueError: if field is not generated using the `generate()` method

        Returns:
            - Optional[List[List[Any]]]: the list of generated values for the fields.

        """

        if self.__generated_field is None:
            raise ValueError("Field is not generated yet.")

        return [self.__generated_field]

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

        variance = (self.cov * self.mean_value)**2

        rf_generator = RandomFields(n_dim=self.__n_dim,
                                    mean=self.mean_value,
                                    variance=variance,
                                    model_name=ModelName[self.model_name],
                                    v_scale_fluctuation=self.v_scale_fluctuation,
                                    anisotropy=self.anisotropy,
                                    angle=self.angle,
                                    seed=self.seed,
                                    v_dim=VERTICAL_AXIS)
        coordinates_for_rf = np.array(coordinates)
        rf_generator.generate(coordinates_for_rf)
        self.__generated_field = list(rf_generator.random_field)[0].tolist()


class ElasticFieldsFromCptGenerator(FieldGeneratorABC):

    def __init__(self, cpt_folder, ref_coordinates, orientation_x_axis):

        self.cpt_folder = cpt_folder
        self.ref_coordinates = ref_coordinates
        self.orientation_x_axis = orientation_x_axis

        self.porosity: Optional[float] = None
        self.fluid_density: Optional[float] = None
        self.field_properties: List[str] = []
        self.max_conditioning_points: int = 2000
        self.__generated_fields: Optional[List[List[Any]]] = None

    @property
    def generated_fields(self) -> Optional[List[List[Any]]]:
        return self.__generated_fields

    def generate(self, coordinates: npty.NDArray[np.float64]):

        if self.porosity is None:
            raise ValueError("Porosity is not set.")
        if self.fluid_density is None:
            raise ValueError("Fluid density is not set.")
        if len(self.field_properties) == 0:
            raise ValueError("Field properties are not set.")
        if any([field not in RandomFieldProperties.__members__ for field in self.field_properties]):
            raise ValueError(f"Field properties should be one or both of {list(RandomFieldProperties.__members__)}")

        # get enums from the field properties
        field_properties = [RandomFieldProperties[field] for field in self.field_properties]

        elastic_field_generator = ElasticityFieldsFromCpt(cpt_file_folder=self.cpt_folder,
                                                          porosity=self.porosity,
                                                          water_density=self.fluid_density,
                                                          x_ref=self.ref_coordinates[0],
                                                          y_ref=self.ref_coordinates[2],
                                                          orientation_x_axis=self.orientation_x_axis,
                                                          return_property=field_properties,
                                                          max_conditioning_points=self.max_conditioning_points,
                                                          based_on_midpoint=False)

        # calibrate the geostatistical model
        elastic_field_generator.calibrate_geostat_model()

        # generate the fields on the coordinates
        elastic_field_generator.generate(coordinates)

        self.__generated_fields = [field.tolist() for field in elastic_field_generator.generated_field]
