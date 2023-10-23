from dataclasses import dataclass
from abc import ABC
from typing import Optional, List, Any

from random_fields.generate_field import RandomFields

_field_input_types = ["python", "json_file", "input"]


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

        - field_generator (Optional[RandomFields]): the field generator to produce the values in the json file.
            Currently only random fields is supported but will be in the future implemented as custom functions that
            take in input X, Y, Z coordinates. Not required for `python` and `input` function types.
    """

    variable_name: str
    function_type: str
    function: str
    field_generator: Optional[RandomFields]

    def __post_init__(self):
        """
        Validation of inputs
        """
        self.function_type = self.function_type.lower()

        if self.function_type not in _field_input_types:
            raise ValueError(f"ParameterField Error:\n"
                             f"`function_type` is not understood: {self.function_type}.\n"
                             f"Should be one of {_field_input_types}.")

        if self.function_type == "python" and ".py" in self.function:
            self.function = self.function.split('.')[0]

        if self.function_type == "json_file" and ".json" not in self.function:
            self.function = self.function_type+'.json'

        if self.field_generator is None and self.function_type == "json_file":
            raise ValueError("Field generator object is required to produce the json file"
                             " parameters!")

    @property
    def values(self) -> Any:

        if self.field_generator is None:
            raise ValueError("No generator is assigned. Error.")

        if self.field_generator.random_field is None:
            raise ValueError("Values for field parameters are not generated yet.")

        return list(self.field_generator.random_field)[0].tolist()
