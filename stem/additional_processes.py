from dataclasses import dataclass
from abc import ABC
from typing import Optional

from random_fields.generate_field import RandomFields

_field_input_types = ["pyhon", "json_file", "input"]


@dataclass
class AdditionalProcessesParametersABC(ABC):
    """
    Abstract base class to describe the parameters required for additional processes (e.g. excavations and random
    fields)
    """


@dataclass
class Excavation(AdditionalProcessesParametersABC):
    """
    Class containing the parameters for an excavation process

    Inheritance:
        - :class:`AdditionalProcessesParametersABC`

    Attributes:
        - deactivate_body_model_part (bool): Deactivate or not the body model part
    """

    deactivate_body_model_part: bool

@dataclass
class ParameterFieldParameters(AdditionalProcessesParametersABC):
    """
    More info here: https://github.com/KratosMultiphysics/Kratos/blob/master/applications/GeoMechanicsApplication/python_scripts/set_parameter_field_process.py

    For the changing a parameter field, 3 options are available for the `function_type` parameter:
        -  `json_file`: an additional json file should be provided that contains a `values` field.
            The number length of the values must match with the number of elements of the part to be updated.

            Parameters required: `dataset_file_name`, name of the json file containing the values, including json
                extension.
            Dummy parameters: `function` and `dataset`.

        -  `python`: an additional python script should be provided file should be provided that provides a manual
            update of the parameters.
            Parameters required: `function`, name of the function without .py extension.
            Dummy parameters: `dataset`


            This type is currently not supported because requires to be linked directly to
            Kratos. Below an example template:
            ___________________________________________________________________________________________________________

            from KratosMultiphysics.GeoMechanicsApplication.user_defined_scripts.user_defined_parameter_field_base \
                import ParameterFieldBase


            class ParameterField(ParameterFieldBase):
                '''Base class of a user defined parameter field'''

                def generate_field(self):
                    '''Creates custom parameter field'''

                    super(ParameterField, self).generate_field()

                    input_dict = self.get_input()
                    output_dict = self.get_output()

                    # add custom run functionalities here

                    new_values = []
                    for value, coord in zip(input_dict['values'], input_dict['coordinates']):
                        new_value = value * 2 * coord[0] + value * 3 * coord[1]
                    new_values.append(new_value)

                    output_dict["values"] = new_values
            ___________________________________________________________________________________________________________

        -   `input`: In this case, the function is explicitly defined as function of coordinates
            Parameters required: `function`, the explicit function.
                e.g. `20000*x + 30000*y`
            Dummy parameters: `dataset`


    Attributes:
        - variable_name (str): the name of the variable that needs to be changed (e.g. YOUNG_MODULUS)
        - function_type (str): the type of function to be provided. It can be either `json_file`, `python` or `input`,
            as described in the description.

        - function (str): this depends on function_type
            o `python` , this is the function of the python function (without .py extension)
            o `json_file`, this is the json file containing the new values of the parameter (with .json extension)
            o `input`, is a string with dependency of the parameter on coordinates (e.g. `x + y**2`)

    """

    variable_name: str
    function_type: str
    function: str
    rf_generator: RandomFields

    def __post_init__(self):
        """
        Validation of inputs
        """
        if self.function_type.lower() not in _field_input_types:
            raise ValueError(f"ParameterField Error:\n"
                             f"`function_type` is not understood: {self.function_type}.\n"
                             f"Should be one of {_field_input_types}.")

        if self.function_type.lower() == "python" and ".py" in self.function:
            self.function = self.function.split('.')[0]

        if self.function_type.lower() == "json_file" and ".json" not in self.function:
            self.function = self.function_type+'.json'

    @property
    def values(self):

        if self.rf_generator.random_field is None:
            raise ValueError("Values for field parameters are not generated yet.")

        return list(self.rf_generator.random_field)[0].tolist()


class AdditionalProcess:
    """
    Class containing the information required to perform additional processes like excavations or add random fields.
    Attributes:
        - process_parameters (:class:`OutputParametersABC`): class containing the process parameters
        - part_name (Optional[str]): name of the body model part to apply the process.
    """

    def __init__(
            self,
            process_parameters: AdditionalProcessesParametersABC,
            part_name: str
    ):
        """
        Constructor of the additional process class
        """

        self.process_parameters = process_parameters
        self.part_name = part_name
