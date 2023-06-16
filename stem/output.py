import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
    VELOCITY = 2
    VELOCITY_X = 21
    VELOCITY_Y = 22
    VELOCITY_Z = 23
    ACCELERATION = 3
    ACCELERATION_X = 31
    ACCELERATION_Y = 32
    ACCELERATION_Z = 33
    TOTAL_DISPLACEMENT = 4
    TOTAL_DISPLACEMENT_X = 41
    TOTAL_DISPLACEMENT_Y = 42
    TOTAL_DISPLACEMENT_Z = 43
    WATER_PRESSURE = 5
    VOLUME_ACCELERATION = 6
    VOLUME_ACCELERATION_X = 61
    VOLUME_ACCELERATION_Y = 62
    VOLUME_ACCELERATION_Z = 63


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
                f"The outputs are incorrectly rendered for the selected output\n"
                f"if vector is selected and ignored if tensor is selected."
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

    @abstractmethod
    def is_output_process(self):
        """
        Abstract method for checking whether an output is in the output process list
        (True for GiD, TVK) or not (False for JSON) which is in `processes`
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

    nodal_results: List[str] = field(default_factory=lambda: [])
    gauss_point_results: List[str] = field(default_factory=lambda: [])

    gid_post_mode: str = "GiD_PostBinary"
    file_label: str = "step"
    output_control_type: str = "step"
    output_interval: int = 1
    body_output: bool = True
    node_output: bool = False
    skin_output: bool = False
    # optional
    plane_output: List[str] = field(default_factory=lambda: [])
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

    def is_output_process(self):
        return True


@dataclass
class VtkOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Attributes:
        nodal_results (List[str]):
        gauss_point_results (List[str]):
        file_format (str): file format for VTK, either `binary` or `ascii` are allowed.
        output_precision (int):
        output_control_type (str): type of output control, either `step` or `time`.
        output_interval (float):
    """
    # general inputs
    nodal_results: List[str] = field(default_factory=lambda: [])
    gauss_point_results: List[str] = field(default_factory=lambda: [])
    # VTK specif inputs
    file_format: str = "binary"
    output_precision: int = 7
    output_control_type: str = "step"
    output_interval: float = 1.0

    def assemble_parameters(self) -> Dict[str, Any]:
        """
        Method for assembling the output properties for VTK format into a nested
        dictionary.

        Returns:
            Dict[str, Any]: dictionary of a list containing the output parameters
        """
        return self.__dict__

    def validate(self):
        detect_tensor_outputs(requested_outputs=self.gauss_point_results)

    def is_output_process(self):
        return True


@dataclass
class JsonOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for JSON output

    Attributes:
        nodal_results (List[str]):
        gauss_point_results (List[str]):
        time_frequency (float):
        sub_model_part_name (str):
    """
    # general inputs
    nodal_results: List[str] = field(default_factory=lambda: [])
    gauss_point_results: List[str] = field(default_factory=lambda: [])
    # JSON specif inputs
    time_frequency: float = 1.0
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
        detect_tensor_outputs(requested_outputs=self.gauss_point_results)

    def is_output_process(self):
        return False


class OutputProcess:
    """
    Class containing output information for postprocessing

    Attributes:
        part_name (str): name of the model part
        output_path (str): output path for the output
            - for GID, it represents the name (or relative path) ot the bin file.
              For relative paths, the stem is referred as the name and the previous
              as the directory. They will be created if they do not exist.
              example1: `test1` results in the test1.bin file in the current folder.
              example2: `path1/path2/test2` results in the test2.bin file in the
                current_folder/path1/path2 where current_folder is the working
                directory.
              example3: `C:/Documents/yourproject/test3` results in the test2.bin
                test3.bin output file in the folder C:/Documents/yourproject.
            - for VTK it represents the path to folder to store the vtk outputs.
              example1: `test1` stores the file in the test1 directory in the project
                folder.
              example2: `path1/path2/test2` stores the file in the test1 directory in
                the project folder path1/path2/test2
            - for JSON, it represents the name (or path) to the file.
                If is a file (thus with extension .json), then it is saved with the
                given name. If extension differs from json it is renamed to json. If
                is a folder (no extension), the filename will have the same name of
                the part of interest `part_name`.json
        output_parameters (OutputParametersABC): class containing the output parameters
    """

    def __init__(
        self, output_path: str, part_name: str, output_parameters: OutputParametersABC
    ):
        """
        Constructor of the output process class

        Args:
            part_name (str): name of the model part
            output_path (str): output path for the output
            output_parameters (OutputParametersABC): class containing output parameters
        """

        self.part_name: str = part_name
        self.output_path: Path = Path(output_path)
        self.output_parameters: OutputParametersABC = output_parameters
