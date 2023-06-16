from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

from abc import ABC

import numpy as np

from stem.load import Load, PointLoad, MovingLoad
from stem.output import (
    OutputProcess,
    GiDOutputParameters,
    VtkOutputParameters,
    JsonOutputParameters,
)

DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        -

    """
    # TODO:
    #  add project folder to the attributes for relative output paths

    def __init__(self):
        pass

    def write_mesh_to_mdpa(self, nodes, elements, filename):
        """
        Saves mesh data to mdpa file

        Args:
            nodes (np.array): node id followed by node coordinates in an array
            elements (np.array): element id followed by connectivities in an array
            filename (str): filename of mdpa file

        Returns:
            -

        """

        # todo improve this such that nodes and elements are written in the same mdpa file, where the elements are split per physical group

        np.savetxt(
            "0.nodes.mdpa", nodes, fmt=["%.f", "%.10f", "%.10f", "%.10f"], delimiter=" "
        )
        # np.savetxt('1.lines.mdpa', lines, delimiter=' ')
        # np.savetxt('2.surfaces.mdpa', surfaces, delimiter=' ')
        # np.savetxt('3.volumes.mdpa', volumes, delimiter=' ')

    def __write_problem_data(self):
        pass

    def __write_solver_settings(self):
        pass

    def __write_output_processes(self):
        pass

    def __write_input_processes(self):
        pass

    def __write_constraints(self):
        pass

    def __write_loads(self):
        pass

    @staticmethod
    def __create_point_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the point load parameters

        Args:
            load (Load): point load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": load.load_parameters.__dict__,
        }

        load_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{load.name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    @staticmethod
    def __create_moving_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the moving load parameters

        Args:
            load (Load): moving load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "set_moving_load_process",
            "kratos_module": "StructuralMechanicsApplication",
            "process_name": "SetMovingLoadProcess",
            "Parameters": load.load_parameters.__dict__,
        }

        load_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{load.name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"

        return load_dict

    @staticmethod
    def __create_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the load parameters

        Args:
            load (Load): load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # add load parameters to dictionary based on load type.
        if isinstance(load.load_parameters, PointLoad):
            return KratosIO.__create_point_load_dict(load=load)
        elif isinstance(load.load_parameters, MovingLoad):
            return KratosIO.__create_moving_load_dict(load=load)
        else:
            raise NotImplementedError

    def create_loads_process_dictionary(self, loads: List[Load]) -> Dict[str, Any]:
        """
        Creates a dictionary containing the load_process_list (list of
        dictionaries to specify the loads for the model)

        Args:
            loads (List[Load]): list of load objects

        Returns:
            loads_dict (Dict): dictionary of a list containing the load properties
        """

        loads_dict: Dict[str, Any] = {"loads_process_list": []}

        for load in loads:
            loads_dict["loads_process_list"].append(self.__create_load_dict(load))

        return loads_dict

    @staticmethod
    def __create_gid_output_dict(
            part_name: str, output_path: Path, output_parameters: GiDOutputParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in GiD
        format. To visualize the outputs, the software GiD is required.

        Args:
            part_name (str): name of the model part
            output_path (Path): output path for the GiD output
            output_parameters (GiDOutputParameters): class containing GiD output
                parameters

        Returns:
            Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """
        _output_path_gid = str(output_path).replace('\\', '/')

        parameters_dict = {
            "model_part_name": f"{DOMAIN}.{part_name}",
            "output_name": _output_path_gid,
            "postprocess_parameters": {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "WriteDeformedMeshFlag": "WriteUndeformed",
                        "WriteConditionsFlag": "WriteElementsOnly",
                        "GiDPostMode": output_parameters.gid_post_mode,
                        "MultiFileFlag": "SingleFile"
                    },
                    "file_label": output_parameters.file_label,
                    "output_control_type": output_parameters.output_control_type,
                    "output_interval": output_parameters.output_interval,
                    "body_output": output_parameters.body_output,
                    "node_output": output_parameters.node_output,
                    "skin_output": output_parameters.skin_output,
                    "plane_output": output_parameters.plane_output,
                    "nodal_results": output_parameters.nodal_results,
                    "gauss_point_results": output_parameters.gauss_point_results,
                },
                "point_data_configuration": output_parameters.point_data_configuration,
            }
        }
        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "gid_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "GiDOutputProcess",
            "Parameters": parameters_dict,
        }

        return output_dict

    @staticmethod
    def __create_vtk_output_dict(
        part_name: str, output_path: Path, output_parameters: VtkOutputParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in vtk
        format. The format can be visualized e.g., using Paraview.

        Args:
            part_name (str): name of the model part
            output_path (Path): output path for the GiD output
            output_parameters (VtkOutputParameters): class containing VTK output
                parameters

        Returns:
            Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        _output_path_vtk = str(output_path).replace('\\', '/')
        parameters_dict = {
            "model_part_name": f"{DOMAIN}.{part_name}",
            "output_path": _output_path_vtk,
            "file_format": output_parameters.file_format,
            "output_precision": output_parameters.output_precision,
            "output_control_type": output_parameters.output_control_type,
            "output_interval": output_parameters.output_interval,
            "nodal_solution_step_data_variables": output_parameters.nodal_results,
            "gauss_point_variables_in_elements": output_parameters.gauss_point_results
        }

        # initialize load dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "vtk_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "VtkOutputProcess",
            "Parameters": parameters_dict
        }

        return output_dict

    @staticmethod
    def __create_json_output_dict(
        part_name, output_path: Path, output_parameters: JsonOutputParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in
        JSON format.

        Args:
            part_name (str): name of the model part
            output_path (str): output path for the GiD output
            output_parameters (JsonOutputParameters): class containing JSON output
                parameters

        Returns:
            Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        _output_path_json = output_path
        if _output_path_json.suffix == "":
            # assume is a folder
            _output_path_json = _output_path_json.joinpath(f"{part_name}" + ".json")
        elif _output_path_json.suffix == "":
            _output_path_json = _output_path_json.with_suffix(".json")

        _output_path_json = str(_output_path_json).replace('\\', '/')
        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "json_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "JsonOutputProcess",
            "Parameters": {
                "model_part_name": f"{DOMAIN}.{part_name}",
                "output_file_name": _output_path_json,
                "output_variables": output_parameters.nodal_results,
                "gauss_points_output_variables": output_parameters.gauss_point_results,
                "sub_model_part_name": output_parameters.sub_model_part_name,
            },
        }

        return output_dict

    def __create_output_dict(self, output: OutputProcess) -> Tuple[str, Dict[str, Any]]:
        """
        Creates a dictionary containing the output parameters for the desired format.
        Allowed format are GiD, VTK and JSON.

        Args:
            output (OutputProcess): output process object

        Returns:
            str: string specifying the format of the output
            Dict[str, Any]: dictionary containing the output parameters
        """
        # add output keys and parameters to dictionary based on output process type.
        if isinstance(output.output_parameters, GiDOutputParameters):
            return "gid_output", KratosIO.__create_gid_output_dict(**output.__dict__)
        elif isinstance(output.output_parameters, VtkOutputParameters):
            return "vtk_output", KratosIO.__create_vtk_output_dict(**output.__dict__)
        elif isinstance(output.output_parameters, JsonOutputParameters):
            return "json_output", KratosIO.__create_json_output_dict(**output.__dict__)
        else:
            raise NotImplementedError

    def create_output_process_dictionary(
        self, outputs: List[OutputProcess]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Creates a dictionary containing the output_processes, that specifies
        which output to request Kratos and the type of output ('GiD', 'VTK',
        'JSON')

        Args:
            outputs (List[OutputProcess]): list of output process objects

        Returns:
            Tuple[Dict[str, Any]]: Tuple of two dictionaries containing the output
                properties.
                - the first containing the "output_process" dictionary. This is a
                  separate dictionary.
                - the second containing the "json_output" dictionary. This is to be
                  placed under "processes".
        """
        output_dict: Dict[str, Any] = {"output_processes": {}}
        json_dict: Dict[str, Any] = {"json_output": []}

        for output in outputs:
            output.output_parameters.validate()
            key_output, _parameters_output = self.__create_output_dict(output=output)
            if output.output_parameters.is_output_process():
                if key_output in output_dict["output_processes"].keys():
                    output_dict["output_processes"][key_output].append(
                        _parameters_output
                    )
                else:
                    output_dict["output_processes"][key_output] = [_parameters_output]

            else:
                json_dict[key_output].append(_parameters_output)

        return output_dict, json_dict

    def write_project_parameters_json(self, filename):
        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()
        self.__write_constraints()
        self.__write_loads()
        # todo write Projectparameters.json
        pass

    def write_material_parameters_json(self, materials, filename):
        pass
