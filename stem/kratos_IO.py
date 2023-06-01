import json

import numpy as np
from stem.load import Load, PointLoad
from typing import List, Dict, Any


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        -

    """

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

    def write_project_parameters_json(self, filename):

        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()

        # todo write Projectparameters.json
        pass

    def write_material_parameters_json(self, materials, filename):
        pass

    def __create_load_dict(self, load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the load parameters
        Args:
            load (Load): load object
        Returns:  Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": None,
            "kratos_module": None,
            "process_name": None,
            "Parameters": {},
        }

        # add load parameters to dictionary based on load type.
        if isinstance(load.load_parameters, PointLoad):
            load_dict.update(
                {
                    "python_module": "apply_vector_constraint_table_process",
                    "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                    "process_name": "ApplyVectorConstraintTableProcess",
                }
            )
            load_dict.update(
                {
                    "Parameters": {
                        "model_part_name": f"PorousDomain.{load.name}",
                        "variable_name": "POINT_LOAD",
                        "active": load.load_parameters.active,
                        "value": load.load_parameters.value,
                        "table": [0, 0, 0],
                    }
                }
            )

        return load_dict

    def create_loads_process_dictionary(self, loads: List[Load], filename: str):
        """
        Writes the project parameters to a json file
        Args:
            constraints (List[Constraint]): list of contraint
            loads (List[Material]): list of material objects
            filename: filename of the output json file
        """

        loads_dict: Dict[str, Any] = {"loads_process_list": []}

        for load in loads:
            loads_dict["loads_process_list"].append(self.__create_load_dict(load))

        return loads_dict
