from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

DEFAULT_GID_FLAGS = {
    "WriteDeformedMeshFlag": "WriteUndeformed",
    "WriteConditionsFlag": "WriteElementsOnly",
    "GiDPostMode": "GiD_PostBinary",
    "MultiFileFlag": "SingleFile",
}


@dataclass
class OutputParametersABC(ABC):
    """
    Abstract class for the definition of user output parameters (GiD, VTK, json).

    Attributes:
        -
    """

    pass

    @abstractmethod
    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Abstract method for assembling the output properties into a nested dictionary
        """
        pass


@dataclass
class GiDOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Attributes:
        gidpost_flags (Dict[str, Any]):
        file_label (str):
        output_control_type (str):
        output_interval (int):
        body_output (bool):
        node_output (bool):
        skin_output (bool):
        plane_output (List[str]):
        nodal_results (List[str]):
        gauss_point_results (List[str]):
        point_data_configuration (List[str]):
    """

    gidpost_flags: Dict[str, Any] = field(default_factory=lambda: DEFAULT_GID_FLAGS)
    file_label: str = "step"
    output_control_type: str = "step"
    output_interval: int = 1
    body_output: bool = True
    node_output: bool = False
    skin_output: bool = False
    plane_output: List[str] = field(default_factory=lambda: [])
    nodal_results: List[str] = field(default_factory=lambda: [])
    gauss_point_results: List[str] = field(default_factory=lambda: [])
    point_data_configuration: List[str] = field(default_factory=lambda: [])

    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Method for assembling the output properties for GiD format into a nested
        dictionary.

        Returns:
            Dict[str, Any]: dictionary of a list containing the output parameters
        """
        return {
            "postprocess_parameters": {
                "result_file_configuration": dict(
                    gidpost_flags=self.gidpost_flags,
                    file_label=self.file_label,
                    output_control_type=self.output_control_type,
                    output_interval=self.output_interval,
                    body_output=self.body_output,
                    node_output=self.node_output,
                    skin_output=self.skin_output,
                    plane_output=self.plane_output,
                    nodal_results=self.nodal_results,
                    gauss_point_results=self.gauss_point_results,
                ),
                "point_data_configuration": [],
            }
        }


@dataclass
class VtkOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Attributes:
        file_format (str): file format for VTK, either `binary` or `ascii` are allowed.
        output_precision (int):
        output_control_type (str): type of output control, either `step` or `time`.
        output_interval (float):
        output_sub_model_parts (bool):
        custom_name_prefix (str):
        custom_name_postfix (str):
        save_output_files_in_folder (bool):
        write_deformed_configuration (bool):
        write_ids (bool):
        nodal_solution_step_data_variables (List[str]):
        nodal_data_value_variables (List[str]):
        nodal_flags (List[str]):
        element_data_value_variables (List[str]):
        element_flags (List[str]):
        condition_data_value_variables (List[str]):
        condition_flags (List[str]):
        gauss_point_variables_extrapolated_to_nodes (List[str]):
        gauss_point_variables_in_elements (List[str]):
    """

    file_format: str = "binary"
    output_precision: int = 7
    output_control_type: str = "step"
    output_interval: float = 1.0
    output_sub_model_parts: bool = False
    custom_name_prefix: str = ""
    custom_name_postfix: str = ""
    save_output_files_in_folder: bool = True
    write_deformed_configuration: bool = False
    write_ids: bool = False
    nodal_solution_step_data_variables: List[str] = field(default_factory=lambda: [])
    nodal_data_value_variables: List[str] = field(default_factory=lambda: [])
    nodal_flags: List[str] = field(default_factory=lambda: [])
    element_data_value_variables: List[str] = field(default_factory=lambda: [])
    element_flags: List[str] = field(default_factory=lambda: [])
    condition_data_value_variables: List[str] = field(default_factory=lambda: [])
    condition_flags: List[str] = field(default_factory=lambda: [])
    gauss_point_variables_extrapolated_to_nodes: List[str] = field(
        default_factory=lambda: []
    )
    gauss_point_variables_in_elements: List[str] = field(default_factory=lambda: [])

    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Method for assembling the output properties for VTK format into a nested
        dictionary.

        Returns:
            Dict[str, Any]: dictionary of a list containing the output parameters
        """
        return self.__dict__


@dataclass
class JsonOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for JSON output

    Attributes:
        sub_model_part_name (str):
        time_frequency (float):
        output_variables (List[str]):
        gauss_points_output_variables (List[str]):
        check_for_flag (str):
        historical_value (bool):
        resultant_solution (bool):
        use_node_coordinates (bool):
    """

    sub_model_part_name: str = ""
    time_frequency: float = 1.0
    output_variables: List[str] = field(default_factory=lambda: [])
    gauss_points_output_variables: List[str] = field(default_factory=lambda: [])
    check_for_flag: str = ""
    historical_value: bool = True
    resultant_solution: bool = False
    use_node_coordinates: bool = False

    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Method for assembling the output properties for JSON format into a nested
        dictionary.

        Returns:
            Dict[str, Any]: dictionary of a list containing the output parameters
        """
        return self.__dict__


class OutputProcess:
    """
    Class containing output information for postprocessing

    Attributes:
        part_name (str): name of the model part
        output_name (str): name of the output file
        output_parameters (OutputParametersABC): class containing the output parameters
    """

    def __init__(
        self, output_name: str, part_name: str, output_parameters: OutputParametersABC
    ):
        """
        Constructor of the output process class

        Args:
            part_name (str): name of the model part
            output_name (str): name of the output file
            output_parameters (OutputParametersABC): class containing output parameters
        """

        self.part_name: str = part_name
        self.output_name: str = output_name
        self.output_parameters: OutputParametersABC = output_parameters
