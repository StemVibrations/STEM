import json
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any


from stem.output import (
    Output,
    GiDOutputParameters,
    VtkOutputParameters,
    JsonOutputParameters,
    OutputParametersABC,
)


class KratosOutputsIO:
    """
    Class containing methods to write outputs to Kratos

    Attributes:
        - domain (str): name of the Kratos domain
    """

    def __init__(self, domain: str):
        """
        Constructor of KratosOutputsIO class

        Args:
            - domain (str): name of the Kratos domain
        """
        self.domain = domain

    def __create_gid_output_dict(
        self,
        part_name: str,
        output_dir: Path,
        output_name: str,
        output_parameters: GiDOutputParameters,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in GiD
        format. To visualize the outputs, the software GiD is required.

        Args:
            - part_name (str): name of the model part
            - output_dir (Path): output path for the GiD output
            - output_name (str): Name for the output file.
            - output_parameters (:class:`stem.output.GiDOutputParameters`): class containing GiD output
                  parameters

        Returns:
            - Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        if output_name is None or output_name == "":
            output_name = f"{part_name}"

        _output_path = output_dir.joinpath(output_name)
        __output_path_gid = str(_output_path).replace("\\", "/")

        if output_parameters.file_format == "binary":
            gid_post_mode = "GiD_PostBinary"
        elif output_parameters.file_format == "ascii":
            gid_post_mode = "GiD_PostAscii"

        elif output_parameters.file_format == "hdf5":
            gid_post_mode = "GiD_PostHDF5"
        else:
            raise ValueError(
                "Incorrect selected output for GiD `file_format` "
                f"variable:{output_parameters.file_format}.\n"
                f"Accepted inputs are `binary`, `ascii` or `hdf5`."
            )

        parameters_dict = {
            "model_part_name": f"{self.domain}.{part_name}",
            "output_name": __output_path_gid,
            "postprocess_parameters": {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "WriteDeformedMeshFlag": "WriteUndeformed",
                        "WriteConditionsFlag": "WriteElementsOnly",
                        "GiDPostMode": gid_post_mode,
                        "MultiFileFlag": "SingleFile",
                    },
                    "file_label": output_parameters.file_label,
                    "output_control_type": output_parameters.output_control_type,
                    "output_interval": output_parameters.output_interval,
                    "body_output": output_parameters.body_output,
                    "node_output": output_parameters.node_output,
                    "skin_output": output_parameters.skin_output,
                    "plane_output": output_parameters.plane_output,
                    "nodal_results": [
                        op.name for op in output_parameters.nodal_results
                    ],
                    "gauss_point_results": [
                        op.name for op in output_parameters.gauss_point_results
                    ],
                },
                "point_data_configuration": output_parameters.point_data_configuration,
            },
        }
        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "gid_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "GiDOutputProcess",
            "Parameters": parameters_dict,
        }

        return output_dict

    def __create_vtk_output_dict(
        self,
        part_name: str,
        output_dir: Path,
        output_name: str,
        output_parameters: VtkOutputParameters,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in vtk
        format. The format can be visualized e.g., using Paraview.

        Args:
            - part_name (str): name of the model part
            - output_dir (Path): output path for the VTK output
            - output_name (str): Name for the output file. This parameter is ignored by
                VTK output process.
            - output_parameters (:class:`stem.output.VtkOutputParameters`): class containing VTK output
                  parameters

        Returns:
            - Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        _output_path = output_dir
        __output_path_vtk = str(_output_path).replace("\\", "/")

        parameters_dict = {
            "model_part_name": f"{self.domain}.{part_name}",
            "output_path": __output_path_vtk,
            "file_format": output_parameters.file_format,
            "output_precision": output_parameters.output_precision,
            "output_control_type": output_parameters.output_control_type,
            "output_interval": output_parameters.output_interval,
            "nodal_solution_step_data_variables": [
                op.name for op in output_parameters.nodal_results
            ],
            "gauss_point_variables_in_elements": [
                op.name for op in output_parameters.gauss_point_results
            ],
        }

        # initialize load dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "vtk_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "VtkOutputProcess",
            "Parameters": parameters_dict,
        }

        return output_dict

    def __create_json_output_dict(
        self,
        part_name,
        output_dir: Path,
        output_name: str,
        output_parameters: JsonOutputParameters,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in
        JSON format.

        Args:
            - part_name (str): name of the model part
            - output_dir (Path): output path for the VTK output
            - output_name (str): Name for the output file. If not needed, the name is
                  taken as the part_name.
            - output_parameters (:class:`stem.output.JsonOutputParameters`): class containing JSON output
                  parameters

        Returns:
            - Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        if output_name is None or output_name == "":
            output_name = f"{part_name}" + ".json"

        # create the target folder for json or simulation will not run.
        output_dir.mkdir(parents=True, exist_ok=True)

        _output_path = output_dir.joinpath(output_name)
        if _output_path.suffix == "":
            _output_path = _output_path.with_suffix(".json")

        __output_path_json = str(_output_path).replace("\\", "/")

        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "json_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "JsonOutputProcess",
            "Parameters": {
                "model_part_name": f"{self.domain}.{part_name}",
                "output_file_name": __output_path_json,
                "output_variables": [op.name for op in output_parameters.nodal_results],
                "gauss_points_output_variables": [
                    op.name for op in output_parameters.gauss_point_results
                ],
                "time_frequency": output_parameters.time_frequency,
            },
        }
        return output_dict

    def __create_output_dict(self, output: Output) -> Tuple[str, Dict[str, Any]]:
        """
        Creates a dictionary containing the output parameters for the desired format.
        Allowed format are GiD, VTK and JSON.

        Args:
            - output (:class:`stem.output.Output`): output process object

        Returns:
            - str: string specifying the format of the output
            - Dict[str, Any]: dictionary containing the output parameters
        """
        # add output keys and parameters to dictionary based on output process type.
        if isinstance(output.output_parameters, GiDOutputParameters):
            return "gid_output", self.__create_gid_output_dict(**output.__dict__)
        elif isinstance(output.output_parameters, VtkOutputParameters):
            return "vtk_output", self.__create_vtk_output_dict(**output.__dict__)
        elif isinstance(output.output_parameters, JsonOutputParameters):
            return "json_output", self.__create_json_output_dict(**output.__dict__)
        else:
            raise NotImplementedError

    @staticmethod
    def __get_process_type_for_output(output_parameters: OutputParametersABC) -> str:
        """
        Checks if considered output parameters are `output_process` as for VTK and
        GiD JSON output types or `process` as for JSON output type.

        Args:
            - output_parameters (:class:`stem.output.Output`): class containing output parameters

        Returns:
            - Output dictionary key: string specifying the location of the output definition
        """

        if isinstance(output_parameters, (VtkOutputParameters, GiDOutputParameters)):
            return "output_processes"
        return "processes"

    def create_output_process_dictionary(self, outputs: List[Output]) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output_processes, that specifies which
        output to request Kratos and the type of output ('GiD', 'VTK', 'JSON')

        Args:
            - outputs (List[:class:`stem.output.Output`]): list of output process objects

        Returns:
            - output_dict (Dict[str, Any]): dictionary containing two other dictionary \
                                            for output properties: \n
                                            - the first containing the "output_process" dictionary.
                                            - the second containing the "processes" dictionary, which includes JSON outputs.
        """
        output_dict: Dict[str, Any] = {"output_processes": {}, "processes": {}}

        for output in outputs:
            output.output_parameters.validate()
            key_output, _parameters_output = self.__create_output_dict(output)
            key_process = KratosOutputsIO.__get_process_type_for_output(
                output.output_parameters
            )
            if key_output in output_dict[key_process].keys():
                output_dict[key_process][key_output].append(_parameters_output)
            else:
                output_dict[key_process][key_output] = [_parameters_output]
        return output_dict
