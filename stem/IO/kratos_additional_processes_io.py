from typing import Any, Dict, Union, List
from copy import deepcopy

from stem.additional_processes import *


class KratosAdditionalProcessesIO:
    """
    Class containing methods for additional Kratos processes

    Attributes:
        - domain (str): name of the Kratos domain

    """

    def __init__(self, domain: str):
        """
        Constructor of KratosBoundariesIO class

        Args:
            - domain (str): name of the Kratos domain

        """
        self.domain = domain

    def __create_excavation_dict(self, part_name: str, parameters: Excavation) -> Dict[str, Any]:
        """
        Creates a dictionary containing the parameters for the excavation process

        Args:
            - part_name (str): part name where the excavation is applied
            - parameters (:class:`stem.additional_processes.Excavation`): excavation parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the additional process parameters
        """

        # initialize boundary dictionary
        process_dict: Dict[str, Any] = {
            "python_module": "apply_excavation_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyExcavationProcess",
            "Parameters": {},
        }

        process_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        process_dict["Parameters"]["variable_name"] = "EXCAVATION"
        process_dict["Parameters"]["deactivate_soil_part"] = parameters.deactivate_body_model_part

        return process_dict

    def __create_parameter_field_dict(self, part_name: str,
                                      parameters: ParameterFieldParameters) -> List[Dict[str, Any]]:
        """
        Creates a dictionary containing the parameters for the parameter field process

        Args:
            - part_name (str): part name where the parameter field is applied
            - parameters (:class:`stem.additional_processes.ParameterFieldParameters`): parameter field parameters \
                object

        Raises:
            - ValueError: if `field_file_names` is not provided when `json_file` function type is selected

        Returns:
            - List[Dict[str, Any]]: list of dictionaries containing the parameter field process parameters
        """
        # add 1 process for each property
        processes = []
        for i, property_name in enumerate(parameters.property_names):
            # initialize parameter field process dictionary
            process_dict: Dict[str, Any] = {
                "python_module": "set_parameter_field_process",
                "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                "process_name": "SetParameterFieldProcess",
                "Parameters": {},
            }

            process_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
            process_dict["Parameters"]["variable_name"] = property_name
            process_dict["Parameters"]["func_type"] = parameters.function_type

            # initialise to dummy
            process_dict["Parameters"]["function"] = "dummy"
            process_dict["Parameters"]["dataset"] = "dummy"

            if parameters.function_type == "json_file":

                if parameters.field_file_names is None or parameters.field_file_names[i] == "":
                    raise ValueError(
                        "`field_file_names` should be provided when `json_file` function type is selected.")

                process_dict["Parameters"]["dataset_file_name"] = parameters.field_file_names[i]
            elif parameters.function_type == "input":
                process_dict["Parameters"]["function"] = parameters.tiny_expr_function
            else:
                raise ValueError(f"function type {parameters.function_type} not supported.")

            processes.append(process_dict)

        return processes

    def __create_hinge_dict(self, part_name: str, parameters: HingeParameters) -> List[Dict[str, Any]]:
        """
        Creates a dictionary containing the parameters for the hinge process

        Args:
            - part_name (str): part name where the hinge is applied
            - parameters (:class:`stem.additional_processes.HingeParameters`): hinge parameters object

        Returns:
            - List[Dict[str, Any]]: list of dictionaries containing the rotational stiffness ratio values for
            axis 2 and 3
        """

        process_dict_axis_2: Dict[str, Any] = {
            "python_module": "assign_scalar_variable_to_nodes_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignScalarVariableToNodesProcess",
            "Parameters": {},
        }
        process_dict_axis_2["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        process_dict_axis_2["Parameters"]["variable_name"] = "ROTATIONAL_STIFFNESS_AXIS_2"

        process_dict_axis_2["Parameters"]["value"] = parameters.ROTATIONAL_STIFFNESS_AXIS_2

        process_dict_axis_3 = deepcopy(process_dict_axis_2)
        process_dict_axis_3["Parameters"]["variable_name"] = "ROTATIONAL_STIFFNESS_AXIS_3"
        process_dict_axis_3["Parameters"]["value"] = parameters.ROTATIONAL_STIFFNESS_AXIS_3

        return [process_dict_axis_2, process_dict_axis_3]

    def create_additional_processes_dict(
            self, part_name: str, parameters: AdditionalProcessesParametersABC) -> Union[List[Dict[str, Any]], None]:
        """
        Creates a list of dictionaries containing the parameters for the additional processes

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.additional_processes.AdditionalProcessesParametersABC`): additional process \
                parameters object

        Returns:
            - List[Dict[str, Any]]: list of dictionaries containing the parameters for the additional process
        """

        # add additional processes dictionary
        if isinstance(parameters, Excavation):
            return [self.__create_excavation_dict(part_name, parameters)]
        elif isinstance(parameters, ParameterFieldParameters):
            return self.__create_parameter_field_dict(part_name, parameters)
        elif isinstance(parameters, HingeParameters):
            return self.__create_hinge_dict(part_name, parameters)
        else:
            raise NotImplementedError
