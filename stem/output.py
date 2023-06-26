from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class NodalOutput(Enum):
    """
    Enum class for variables at the nodes
    """
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
    """
    Enum class for variables at the Gauss Point
    """
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


def detect_vector_in_tensor_outputs(requested_outputs: List[GaussPointOutput]):
    """
    Detects whether gauss point outputs are requested and warns the user

    Args:
        - requested_outputs (List[:class:`GaussPointOutput`]): list of requested outputs (gauss point)
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


def detect_tensor_outputs(requested_outputs: List[GaussPointOutput]):
    """
    Detects whether gauss point outputs are requested and warns the user

    Args:
        - requested_outputs (List[:class:`GaussPointOutput`]): list of requested outputs (gauss point)
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

    Inheritance:
        - :class:`OutputParametersABC`

    Attributes:
        - output_interval (float): frequency of the output, either step interval if
              `output_control_type` is `step` or time interval in seconds if
              `output_control_type` is `time`.
        - output_control_type (str): type of output control, either `step` or `time`.
        - file_format (str): format of output (`binary`,`ascii` or `hdf5`) for the
              gid_post_mode flag
        - nodal_results (List[:class:`NodalOutput`]): list of nodal outputs as defined in
              :class:`NodalOutput`.
        - gauss_point_results (List[:class:`GaussPointOutput`]): list of gauss point outputs as
              defined in :class:`GaussPointOutput`.
        - file_label (str): labelling format for the files (`step` or `time`)
        - body_output (bool):
        - node_output (bool):
        - skin_output (bool):
        - plane_output (List[str]):
        - point_data_configuration (List[str]):
    """

    # general inputs
    output_interval: float
    output_control_type: str = "step"
    file_format: str = "binary"
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])
    # GiD specific inputs
    file_label: str = "step"
    # optional
    body_output: bool = True
    node_output: bool = False
    skin_output: bool = False
    plane_output: List[str] = field(default_factory=lambda: [])
    point_data_configuration: List[str] = field(default_factory=lambda: [])

    def validate(self):
        """
        Validates the gauss point results requested for GiD output.
        Prints warnings if vector format is requested for tensor output.
        """
        detect_vector_in_tensor_outputs(requested_outputs=self.gauss_point_results)


@dataclass
class VtkOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Inheritance:
        - :class:`OutputParametersABC`

    Attributes:
        - output_interval (float): frequency of the output, either step interval if
              `output_control_type` is `step` or time interval in seconds if
              `output_control_type` is `time`.
        - output_control_type (str): type of output control, either `step` or `time`.
        - file_format (str): file format for VTK, either `binary` or `ascii` are allowed.
        - nodal_results (List[:class:`NodalOutput`]): list of nodal outputs as defined in
              :class:`NodalOutput`.
        - gauss_point_results (List[:class:`GaussPointOutput`]): list of gauss point outputs as
              defined in :class:`GaussPointOutput`.
          output_precision (int): precision of the output for ascii. Default is 7.
    """

    # general inputs
    output_interval: float
    output_control_type: str = "step"
    file_format: str = "binary"
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])
    # VTK specif inputs
    output_precision: int = 7

    def validate(self):
        """
        Validates the gauss point results requested for VTK output.
        Prints warnings if tensor are asked in output.
        """
        detect_tensor_outputs(requested_outputs=self.gauss_point_results)


@dataclass
class JsonOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for JSON output

    Inheritance:
        - :class:`OutputParametersABC`

    Attributes:
        - time_frequency (float): time frequency of the output [s].
        - nodal_results (List[:class:`NodalOutput`]): list of nodal outputs as defined in
              :class:`NodalOutput`.
        - gauss_point_results (List[:class:`GaussPointOutput`]): list of gauss point outputs as
              defined in :class:`GaussPointOutput`.
    """

    # JSON specif inputs
    time_frequency: float
    # general inputs
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])

    def validate(self):
        """
        Validates the gauss point results requested for JSON output.
        Prints warnings if tensor are asked in output.
        """
        detect_tensor_outputs(requested_outputs=self.gauss_point_results)


class Output:
    """
    Class containing output information for postprocessing

    Attributes:
        - part_name (str): name of the model part
        - output_dir (str): Optional input. output directory for the relative or
              absolute path to the output file. The path will be created if it does
              not exist yet. If not specified, the files it corresponds to the working
              directory.
              example1=`test1` results in the test1 output folder relative to
                  current folder as ".test1"
              example2=`path1/path2/test2` saves the outputs in
                  current_folder/path1/path2/test2
              example3=`C:/Documents/yourproject/test3` saves the outputs in
                  `C:/Documents/yourproject/test3`.
        - output_name (str): Optional input. Name for the output file. This parameter is
              used by GiD and JSON outputs while is ignored in VTK. If the name is not
              given, the part_name is used instead.
        - output_parameters (:class:`OutputParametersABC`): class containing the output parameters
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
            - part_name (str): name of the model part
            - output_name (str): name for the output file
            - output_dir (str): path to the output files
            - output_parameters (:class:`OutputParametersABC`): class containing output parameters
        """

        self.output_name: str = output_name
        self.part_name: str = part_name
        self.output_dir: Path = Path(output_dir)
        self.output_parameters: OutputParametersABC = output_parameters
