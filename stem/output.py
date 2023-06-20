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


class GaussPointOutput(Enum):
    VON_MISES_STRESS = 1
    FLUID_FLUX_VECTOR = 2
    HYDRAULIC_HEAD = 3
    GREEN_LAGRANGE_STRAIN_VECTOR = 41
    GREEN_LAGRANGE_STRAIN_TENSOR = 42
    ENGINEERING_STRAIN_VECTOR = 51
    ENGINEERING_STRAIN_TENSOR = 52
    CAUCHY_STRESS_VECTOR = 61
    CAUCHY_STRESS_TENSOR = 62
    TOTAL_STRESS_VECTOR = 71
    TOTAL_STRESS_TENSOR = 72


TENSOR_OUTPUTS = [
    "GREEN_LAGRANGE_STRAIN",
    "ENGINEERING_STRAIN",
    "CAUCHY_STRESS",
    "TOTAL_STRESS",
]
# Nodal results
nodal_results1 = [NodalOutput.DISPLACEMENT, NodalOutput.TOTAL_DISPLACEMENT]
nodal_results2 = [NodalOutput.WATER_PRESSURE, NodalOutput.VOLUME_ACCELERATION]
# gauss point results
gauss_point_results1 = [
    GaussPointOutput.VON_MISES_STRESS,
    GaussPointOutput.FLUID_FLUX_VECTOR,
    GaussPointOutput.HYDRAULIC_HEAD,
]
gauss_point_results2 = [
    GaussPointOutput.GREEN_LAGRANGE_STRAIN_TENSOR,
    GaussPointOutput.ENGINEERING_STRAIN_TENSOR,
    GaussPointOutput.CAUCHY_STRESS_TENSOR,
    GaussPointOutput.TOTAL_STRESS_TENSOR,
]


def detect_vector_in_tensor_outputs(requested_outputs: List[GaussPointOutput]):
    """
    Detects whether gauss point outputs are requested and warns the user

    Args:
        requested_outputs (List[st]): list of requested outputs (gauss point)
    """
    if len(requested_outputs) > 0:
        detected_tensor_outputs = []
        for requested_output in requested_outputs:
            for tensor_output in TENSOR_OUTPUTS:
                if (
                    tensor_output in requested_output.name
                    and "_VECTOR" in requested_output.name
                ):
                    detected_tensor_outputs.append(tensor_output)
                    break
        if len(detected_tensor_outputs):
            _fmt_list = "".join([f" - {outpt} \n" for outpt in detected_tensor_outputs])
            _msg = (
                f"\n[WARNING] Vector specified for tensor output:\n{_fmt_list}"
                f"In GiD, Such outputs are incorrectly rendered."
            )
            print(_msg)


# define out


def detect_tensor_outputs(requested_outputs: List[GaussPointOutput]):
    """
    Detects whether gauss point outputs are requested and warns the user

    Args:
        requested_outputs (List[st]): list of requested outputs (gauss point)
    """
    if len(requested_outputs) > 0:
        detected_tensor_outputs = []
        for requested_output in requested_outputs:
            for tensor_output in TENSOR_OUTPUTS:
                if tensor_output in requested_output.name:
                    detected_tensor_outputs.append(tensor_output)
                    break
        if len(detected_tensor_outputs):
            _fmt_list = "".join([f" - {outpt} \n" for outpt in detected_tensor_outputs])
            _msg = (
                f"\n[WARNING] tensor output detected:\n{_fmt_list}"
                f"The outputs are incorrectly rendered in VTK and JSON output types:\n"
                f"For VECTOR the output is incorrectly produced ans misses "
                f"components.\n"
                f"For TENSOR the output is ignored.\n"
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
        file_format (str): format of output (gid_post_mode flag)
    """

    # general inputs
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])
    # GiD specific inputs
    file_format: str = "GiD_PostBinary"
    file_label: str = "step"
    output_control_type: str = "step"
    output_interval: int = 1
    body_output: bool = True
    node_output: bool = False
    skin_output: bool = False
    # optional
    plane_output: List[str] = field(default_factory=lambda: [])
    point_data_configuration: List[str] = field(default_factory=lambda: [])

    def validate(self):
        detect_vector_in_tensor_outputs(requested_outputs=self.gauss_point_results)


@dataclass
class VtkOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Attributes:
        output_interval (float):
        file_format (str): file format for VTK, either `binary` or `ascii` are allowed.
        output_precision (int):
        output_control_type (str): type of output control, either `step` or `time`.
        nodal_results (List[str]):
        gauss_point_results (List[str]):
    """

    # VTK specif inputs
    output_interval: float
    file_format: str = "binary"
    output_precision: int = 7
    output_control_type: str = "step"
    # general inputs
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])

    def validate(self):
        detect_tensor_outputs(requested_outputs=self.gauss_point_results)


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
    # JSON specif inputs
    time_frequency: float
    sub_model_part_name: str = ""
    # general inputs
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])

    def validate(self):
        detect_tensor_outputs(requested_outputs=self.gauss_point_results)


class Output:
    """
    Class containing output information for postprocessing

    Attributes:
        part_name (str): name of the model part
        output_dir (str): Optional input. output directory for the relative or
            absolute path to the output file. The path will be created if it does
            not exist yet. If not specified, the files it corrensponds to the workiing
            directory.
            example1=`test1` results in the test1 output folder relative to
                current folder as ".test1"
            example2=`path1/path2/test2` saves the outputs in
                current_folder/path1/path2/test2
            example3=`C:/Documents/yourproject/test3` saves the outputs in
                `C:/Documents/yourproject/test3`.
        output_name (str): Optional input. Name for the output file. This parameter is
            used by GiD and JSON outputs while is ignored in VTK. If the name is not
            given, the part_name is used instead.

        output_parameters (OutputParametersABC): class containing the output parameters
    """

    def __init__(
        self,
        part_name: str,
        output_parameters: OutputParametersABC,
        output_dir: str = "",
        output_name: str = "",
    ):
        """
        Constructor of the output process class

        Args:
            part_name (str): name of the model part
            output_name (str): name for the output file
            output_dir (str): path to the output files
            output_parameters (OutputParametersABC): class containing output parameters
        """

        self.output_name: str = output_name
        self.part_name: str = part_name
        self.output_dir: Path = Path(output_dir)
        self.output_parameters: OutputParametersABC = output_parameters
