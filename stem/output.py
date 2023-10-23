from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Sequence, Union


class NodalOutput(Enum):
    """
    Enum class for variables at the nodes
    """
    DISPLACEMENT = auto()
    DISPLACEMENT_X = auto()
    DISPLACEMENT_Y = auto()
    DISPLACEMENT_Z = auto()
    VELOCITY = auto()
    VELOCITY_X = auto()
    VELOCITY_Y = auto()
    VELOCITY_Z = auto()
    ACCELERATION = auto()
    ACCELERATION_X = auto()
    ACCELERATION_Y = auto()
    ACCELERATION_Z = auto()
    TOTAL_DISPLACEMENT = auto()
    TOTAL_DISPLACEMENT_X = auto()
    TOTAL_DISPLACEMENT_Y = auto()
    TOTAL_DISPLACEMENT_Z = auto()
    WATER_PRESSURE = auto()
    VOLUME_ACCELERATION = auto()
    VOLUME_ACCELERATION_X = auto()
    VOLUME_ACCELERATION_Y = auto()
    VOLUME_ACCELERATION_Z = auto()


class GaussPointOutput(Enum):
    """
    Enum class for variables at the Gauss Point
    """
    VON_MISES_STRESS = auto()
    FLUID_FLUX_VECTOR = auto()
    HYDRAULIC_HEAD = auto()
    GREEN_LAGRANGE_STRAIN_VECTOR = auto()
    GREEN_LAGRANGE_STRAIN_TENSOR = auto()
    ENGINEERING_STRAIN_VECTOR = auto()
    ENGINEERING_STRAIN_TENSOR = auto()
    CAUCHY_STRESS_VECTOR = auto()
    CAUCHY_STRESS_TENSOR = auto()
    TOTAL_STRESS_VECTOR = auto()
    TOTAL_STRESS_TENSOR = auto()

    # MATERIAL PARAMETERS - soil
    YOUNG_MODULUS = auto()
    POISSON_RATIO = auto()
    DENSITY_SOLID = auto()
    POROSITY = auto()
    PERMEABILITY_XX = auto()
    PERMEABILITY_YY = auto()
    PERMEABILITY_XY = auto()
    BULK_MODULUS_SOLID = auto()
    BIOT_COEFFICIENT = auto()

    PERMEABILITY_YZ = auto()
    PERMEABILITY_ZX = auto()
    PERMEABILITY_ZZ = auto()
    # Material parameters - fluid
    DENSITY_FLUID = auto()
    DYNAMIC_VISCOSITY = auto()
    BULK_MODULUS_FLUID = auto()
    # umat
    UMAT_PARAMETERS = auto()


TENSOR_OUTPUTS = [
    "GREEN_LAGRANGE_STRAIN",
    "ENGINEERING_STRAIN",
    "CAUCHY_STRESS",
    "TOTAL_STRESS",
]

# def validate_gauss_point(requested_outputs:)
#
#     list(GaussPointOutput.__members__.keys())


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


def detect_tensor_outputs(requested_outputs: Sequence[Union[GaussPointOutput, str]]):
    """
    Detects whether gauss point outputs are requested and warns the user if some cause problems
    for the considered output. It also checks if input types are correct

    Args:
        - requested_outputs (List[:class:`GaussPointOutput`]): list of requested outputs (gauss point)
    """
    if len(requested_outputs) > 0:
        detected_tensor_outputs = []
        for requested_output in requested_outputs:
            for tensor_output in TENSOR_OUTPUTS:
                if isinstance(requested_output, str):
                    _req_out = GaussPointOutput[requested_output]
                elif isinstance(requested_output, GaussPointOutput):
                    _name = requested_output.name
                else:
                    raise ValueError("Incorrect type specified for Gauss point output:"
                                     f"{requested_output.__class__.__name__}.")
                if tensor_output in _name:
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

    @abstractmethod
    def validate(self):
        """
        Abstract method for validating user inputs

        Raises:
            - Exception: Abstract method for validate output parameters is called
        """
        raise Exception("Abstract method for validate output parameters is called")


@dataclass
class GiDOutputParameters(OutputParametersABC):
    """
    Class containing the output parameters for GiD output

    Inheritance:
        - :class:`OutputParametersABC`

    Attributes:
        - output_interval (float): frequency of the output, either step interval if\
              `output_control_type` is `step` or time interval in seconds if\
              `output_control_type` is `time`.
        - output_control_type (str): type of output control, either `step` or `time`.
        - file_format (str): format of output (`binary`,`ascii` or `hdf5`) for the gid_post_mode flag
        - nodal_results (List[:class:`NodalOutput`]): list of nodal outputs as defined in :class:`NodalOutput`.
        - gauss_point_results (List[:class:`GaussPointOutput`]): list of gauss point outputs as \
            defined in :class:`GaussPointOutput`.
        - file_label (str): labelling format for the files (`step` or `time`)
    """

    # general inputs
    output_interval: float
    output_control_type: str = "step"
    file_format: str = "binary"
    nodal_results: List[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: List[GaussPointOutput] = field(default_factory=lambda: [])
    # GiD specific inputs
    file_label: str = "step"

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
        - output_interval (float): frequency of the output, either step interval if \
              `output_control_type` is `step` or time interval in seconds if \
              `output_control_type` is `time`.
        - output_control_type (str): type of output control, either `step` or `time`.
        - file_format (str): file format for VTK, either `binary` or `ascii` are allowed.
        - nodal_results (List[:class:`NodalOutput`]): list of nodal outputs as defined in :class:`NodalOutput`.
        - gauss_point_results (List[:class:`GaussPointOutput`]): list of gauss point outputs as \
              defined in :class:`GaussPointOutput`.
          output_precision (int): precision of the output for ascii. Default is 7.
    """

    # general inputs
    output_interval: float
    output_control_type: str = "step"
    file_format: str = "binary"
    nodal_results: Sequence[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: Sequence[GaussPointOutput] = field(default_factory=lambda: [])
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
        - output_interval (float): time frequency of the output [s].
        - nodal_results (List[:class:`NodalOutput`]): list of nodal outputs as defined in :class:`NodalOutput`.
        - gauss_point_results (List[:class:`GaussPointOutput`]): list of gauss point outputs as \
              defined in :class:`GaussPointOutput`.
    """

    # JSON specif inputs
    output_interval: float
    # general inputs
    nodal_results: Sequence[NodalOutput] = field(default_factory=lambda: [])
    gauss_point_results: Sequence[GaussPointOutput] = field(default_factory=lambda: [])

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
        - output_parameters (:class:`OutputParametersABC`): class containing output parameters
        - part_name (Optional[str]): name of the model part
        - output_dir (str): path to the output files
        - output_name (Optional[str]): name for the output file
    """

    def __init__(
        self,
        output_parameters: OutputParametersABC,
        part_name: Optional[str] = None,
        output_dir: str = "./",
        output_name: Optional[str] = None
    ):
        """
        Constructor of the output process class

        Args:
            - output_parameters (:class:`OutputParametersABC`): class containing the output parameters
            - part_name (Optional[str]): name of the submodelpart to be given in output. If None, all the model is
                provided in  output.
            - output_dir (Optional[str]): output directory for the relative or absolute path to the output file. The \
                path will be created if it does not exist yet. \n

                example1='test1' results in the test1 output folder relative to current folder as './test1'\
                example2='path1/path2/test2' saves the outputs in './path1/path2/test2' \
                example3='C:/Documents/yourproject/test3' saves the outputs in 'C:/Documents/yourproject/test3'.

                if output_dir is None, then the current directory is assumed.

                [NOTE]: for VTK file type, the content of the target directory will be deleted. Therefore a subfolder is
                always appended to the specified output directory to avoid erasing important memory content.
                The appended folder is defined based on the submodelpart name specified.

            - output_name (Optional[str]): Name for the output file. This parameter is \
                  used by GiD and JSON outputs while is ignored in VTK. If the name is not \
                  given, the part_name is used instead.
        """

        new_output_dir = Path(output_dir)

        if isinstance(output_parameters, VtkOutputParameters):
            if part_name is None:
                new_output_dir = new_output_dir.joinpath("output_vtk_full_model")
            else:
                new_output_dir = new_output_dir.joinpath("output_vtk_" + part_name)

        self.output_parameters = output_parameters
        self.part_name = part_name
        self.output_dir = new_output_dir
        self.output_name = output_name
