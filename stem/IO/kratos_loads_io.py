from typing import Dict, List, Any
from copy import deepcopy

from stem.load import *


class KratosLoadsIO:
    """
    Class containing methods to write loads to Kratos

    Attributes:
        domain (str): name of the Kratos domain

    """

    def __init__(self, domain: str):
        """
        Constructor of KratosLoadsIO class

        Args:
            domain (str): name of the Kratos domain

        """
        self.domain = domain

    def __create_point_load_dict(self, load: Load) -> Dict[str, Any]:
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
            "Parameters": deepcopy(load.load_parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{load.part_name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def __create_moving_load_dict(self, load: Load) -> Dict[str, Any]:
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
            "Parameters": deepcopy(load.load_parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{load.part_name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"

        return load_dict

    def __create_line_load_dict(self, load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the line load parameters

        Args:
            load (Load): load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(load.load_parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{load.part_name}"
        load_dict["Parameters"]["variable_name"] = "LINE_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def __create_surface_load_dict(self, load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the surface load parameters

        Args:
            load (Load): load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(load.load_parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{load.part_name}"
        load_dict["Parameters"]["variable_name"] = "SURFACE_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def __create_load_dict(self, load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the load parameters

        Args:
            load (Load): load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # add load parameters to dictionary based on load type.
        if isinstance(load.load_parameters, PointLoad):
            return self.__create_point_load_dict(load=load)
        elif isinstance(load.load_parameters, MovingLoad):
            return self.__create_moving_load_dict(load=load)
        elif isinstance(load.load_parameters, LineLoad):
            return self.__create_line_load_dict(load=load)
        elif isinstance(load.load_parameters, SurfaceLoad):
            return self.__create_surface_load_dict(load=load)
        else:
            raise NotImplementedError

    def create_loads_process_dict(self, loads: List[Load]) -> Dict[str, Any]:
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
