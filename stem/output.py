import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

DEFAULT_GID_FLAGS = {
    "WriteDeformedMeshFlag": "WriteUndeformed",
    "WriteConditionsFlag": "WriteElementsOnly",
    "GiDPostMode": "GiD_PostBinary",
    "MultiFileFlag": "SingleFile",
}


class NodalOutput(Enum):
    DISPLACEMENT = 1
    DISPLACEMENT_X = 11
    DISPLACEMENT_Y = 12
    DISPLACEMENT_Z = 13
    TOTAL_DISPLACEMENT = 2
    TOTAL_DISPLACEMENT_X = 21
    TOTAL_DISPLACEMENT_Y = 22
    TOTAL_DISPLACEMENT_Z = 23
    WATER_PRESSURE = 3
    VOLUME_ACCELERATION = 4
    VOLUME_ACCELERATION_X = 41
    VOLUME_ACCELERATION_Y = 42
    VOLUME_ACCELERATION_Z = 43


TENSOR_OUTPUTS = [
    "GREEN_LAGRANGE_STRAIN",
    "ENGINEERING_STRAIN",
    "CAUCHY_STRESS",
    "TOTAL_STRESS",
]


def detect_tensor_outputs(requested_outputs: List[str]):
    """
    Detects whether gauss point outputs are requested and warns the user

    Args:
        requested_outputs (List[st]): list of requested outputs (gauss point)
    """
    if len(requested_outputs) > 0:
        detected_tensor_outputs = []
        for requested_output in requested_outputs:
            for tensor_output in TENSOR_OUTPUTS:
                if tensor_output in requested_output:
                    detected_tensor_outputs.append(tensor_output)
                    break
        if len(detected_tensor_outputs):
            _fmt_list = "".join([f" - {outpt} \n" for outpt in detected_tensor_outputs])
            _msg = (
                f"\n[WARNING] tensor output detected:\n{_fmt_list}"
                f"The outputs are incorrectly rendered for the selected output.\n"
                f"If vector is selected and ignored if tensor is selected."
            )
            print(_msg)


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

    @abstractmethod
    def validate(self):
        """
        Abstract method for validating user inputs
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
                "point_data_configuration": self.point_data_configuration,
            }
        }

    def validate(self):
        pass


@dataclass
class VtkOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Attributes:
        file_format (str): file format for VTK, either `binary` or `ascii` are allowed.
        output_precision (int):
        output_control_type (str): type of output control, either `step` or `time`.
        output_interval (float):
        nodal_solution_step_data_variables (List[str]):
        gauss_point_variables_in_elements (List[str]):
    """

    file_format: str = "binary"
    output_precision: int = 7
    output_control_type: str = "step"
    output_interval: float = 1.0
    nodal_solution_step_data_variables: List[str] = field(default_factory=lambda: [])
    gauss_point_variables_in_elements: List[str] = field(default_factory=lambda: [])

    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Method for assembling the output properties for VTK format into a nested
        dictionary.

        Returns:
            Dict[str, Any]: dictionary of a list containing the output parameters
        """
        return self.__dict__

    def validate(self):
        detect_tensor_outputs(requested_outputs=self.gauss_point_variables_in_elements)


@dataclass
class JsonOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for JSON output

    Attributes:
        time_frequency (float):
        output_variables (List[str]):
        gauss_points_output_variables (List[str]):
        sub_model_part_name (str):
    """

    time_frequency: float = 1.0
    output_variables: List[str] = field(default_factory=lambda: [])
    gauss_points_output_variables: List[str] = field(default_factory=lambda: [])
    sub_model_part_name: str = ""

    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Method for assembling the output properties for JSON format into a nested
        dictionary.

        Returns:
            Dict[str, Any]: dictionary of a list containing the output parameters
        """
        return self.__dict__

    def validate(self):
        detect_tensor_outputs(requested_outputs=self.gauss_points_output_variables)


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
