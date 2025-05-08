from dataclasses import dataclass
from abc import ABC
from typing import Optional, List

from stem.field_generator import FieldGeneratorABC, RandomFieldGenerator

FIELD_INPUT_TYPES = ["json_file", "input"]


@dataclass
class AdditionalProcessesParametersABC(ABC):
    """
    Abstract base class to describe the parameters required for additional processes (e.g. excavations and parameter
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
    r"""
    For the changing a parameter field, 3 options are available to se the parameter field:
        -   json: an additional json file should be provided that contains a `values` field.
            The number length of the values must match with the number of elements of the part to be updated.\
        -   input: In this case, the function is explicitly defined as function of coordinates (x, y and z) and \
            time (t).
        -   python: A python script needs to be provided for the purpose. This is currently not supported in STEM.\

    Attributes:
        - property_names (List[str]): the names of the (material) properties that needs to be changed \
            (e.g. [YOUNG_MODULUS])
        - function_type (str): the type of function to be provided. It can be either `json_file` or `input`,
            as provided in the function documentation.
        - field_file_names (Optional[List[str]]): Name for the json file where the field parameters will be stored. \
            This is optional for `json` function_type.
        - field_generator (Optional[:class:`stem.field_generator.FieldGeneratorABC`]): the field generator to produce \
            the values in the json file. Currently only random fields is supported but will be in the future \
            implemented as custom functions that take in input X, Y, Z coordinates. Not required for \
            `python` and `input` function types. This is optional for `json` function_type.

        - tiny_expr_function (Optional[str]): is a tiny expression string with dependency on coordinates (x, y, z) \
            and time (e.g. `x + y^2 + 2*cos(t)`). For more info check tinyexpr on GitHub. \
            This is optional for `input` function_type.

    """

    property_names: List[str]
    function_type: str
    field_file_names: Optional[List[str]] = None
    field_generator: Optional[FieldGeneratorABC] = None
    tiny_expr_function: Optional[str] = None

    def __post_init__(self):
        """
        Validation of inputs

        Raises:
            - ValueError: if the function type is not `input` or `json_file`.
            - ValueError: if the field_generator is not provided when function_type is `json_file`.`input`
            - ValueError: if the tiny_expr_function is not provided when function_type is `input`.
            - ValueError: if the length of the field_file_names is not equal to the length of the property_names.
            - ValueError: if the length of the property_names is not equal to 1 when field_generator:
                'RandomFieldGenerator' is used.

        """
        self.function_type = self.function_type.lower()

        if self.function_type not in FIELD_INPUT_TYPES:
            raise ValueError(f"ParameterField Error:"
                             f"`function_type` is not understood: {self.function_type}."
                             f"Should be one of {FIELD_INPUT_TYPES}.")

        if self.function_type == "json_file" and self.field_generator is None:
            raise ValueError("`field_generator` parameter is a required when `json_file` field parameter is "
                             "selected for `function_type`.")

        if self.function_type == "input" and self.tiny_expr_function is None:
            raise ValueError("`tiny_expr_function` parameter is a required when `input` field parameter is "
                             "selected for `function_type`.")

        if self.field_file_names is not None:
            if len(self.field_file_names) != len(self.property_names):
                raise ValueError("`field_file_names` should have the same length as `property_names`.")

        if isinstance(self.field_generator, RandomFieldGenerator):
            if len(self.property_names) != 1:
                raise ValueError("Only one property name can be provided for the field generator class "
                                 "'RandomFieldGenerator'.")


@dataclass
class HingeParameters(AdditionalProcessesParametersABC):
    """
    Class containing the parameters for a hinge process

    Inheritance:
        - :class:`AdditionalProcessesParametersABC`

    Attributes:
        - ROTATIONAL_STIFFNESS_AXIS_2 (float): Rotational stiffness ratio local axis 2
        - ROTATIONAL_STIFFNESS_AXIS_3 (float): Rotational stiffness ratio local axis 3
    """

    ROTATIONAL_STIFFNESS_AXIS_2: float
    ROTATIONAL_STIFFNESS_AXIS_3: float
