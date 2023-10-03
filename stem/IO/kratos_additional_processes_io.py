from typing import Any, Dict, Union

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

    def __create_excavation_dict(
        self, part_name: str, parameters: Excavation
    ) -> Dict[str, Any]:
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
            "process_name":  "ApplyExcavationProcess",
            "Parameters": {},
        }

        process_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        process_dict["Parameters"]["variable_name"] = "EXCAVATION"
        process_dict["Parameters"]["deactivate_soil_part"] = parameters.deactivate_body_model_part

        return process_dict

    def __create_parameter_field_dict(
        self, part_name: str, parameters: ParameterFieldParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the parameters for the random field process

        Args:
            - part_name (str): part name where the random field is applied
            - parameters (:class:`stem.additional_processes.Excavation`): random field parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the additional process parameters
        """

        # initialize boundary dictionary
        process_dict: Dict[str, Any] = {
            "python_module": "apply_excavation_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name":  "ApplyExcavationProcess",
            "Parameters": {},
        }

        process_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        process_dict["Parameters"]["variable_name"] = parameters.variable_name
        process_dict["Parameters"]["func_type"] = parameters.function_type

        # initialise to dummy
        process_dict["Parameters"]["function"] = "dummy"
        process_dict["Parameters"]["dataset"] = "dummy"

        if parameters.function_type == "json_file":
            process_dict["Parameters"]["dataset_file_name"] = parameters.function
        elif parameters.function_type == "python":
            process_dict["Parameters"]["function"] = parameters.function
        elif parameters.function_type == "input":
            process_dict["Parameters"]["function"] = parameters.function
        else:
            raise ValueError

        return process_dict

    def create_additional_processes_dict(
        self, part_name: str, parameters: AdditionalProcessesParametersABC
    ) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the boundary parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.additional_processes.AdditionalProcessesParametersABC`): additional process \
                parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the parameters for the additional process
        """

        # add boundary parameters to dictionary based on boundary type.

        if isinstance(parameters, Excavation):
            return self.__create_excavation_dict(part_name, parameters)
        elif isinstance(parameters, ParameterFieldParameters):
            return self.__create_parameter_field_dict(part_name, parameters)
        else:
            raise NotImplementedError
