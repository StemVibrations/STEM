from copy import deepcopy
from typing import Dict, List, Any, Union

from stem.load import *


class KratosLoadsIO:
    """
    Class containing methods to write loads to Kratos

    Attributes:
        - domain (str): name of the Kratos domain
    """

    def __init__(self, domain: str):
        """
        Constructor of KratosLoadsIO class

        Args:
            - domain (str): name of the Kratos domain
        """
        self.domain = domain

    def __create_point_load_dict(self, part_name:str, parameters: PointLoad) -> Dict[str, Any]:
        """
        Creates a dictionary containing the point load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.PointLoad`): point load parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def __create_moving_load_dict(self, part_name:str, parameters: MovingLoad) -> Dict[str, Any]:
        """
        Creates a dictionary containing the moving load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.MovingLoad`): moving load parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "set_moving_load_process",
            "kratos_module": "StructuralMechanicsApplication",
            "process_name": "SetMovingLoadProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"

        return load_dict

    def __create_line_load_dict(self, part_name:str, parameters: LineLoad) -> Dict[str, Any]:
        """
        Creates a dictionary containing the line load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.LineLoad`): line load parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        load_dict["Parameters"]["variable_name"] = "LINE_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def __create_surface_load_dict(self, part_name:str, parameters: LoadParametersABC) -> Dict[str, Any]:
        """
        Creates a dictionary containing the surface load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.SurfaceLoad`): surface load parameters object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        load_dict["Parameters"]["variable_name"] = "SURFACE_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def __create_gravity_load_dict(self, part_name:str, parameters: GravityLoad) -> Dict[str, Any]:
        """
        Creates a dictionary containing the surface load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.GravityLoad`): gravity load parameters object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name":  "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        load_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        load_dict["Parameters"]["variable_name"] = "VOLUME_ACCELERATION"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    def create_load_dict(self, part_name:str, parameters: LoadParametersABC) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the load parameters

        Args:
            - model_part (:class:`stem.model_part.ModelPart`): model part (load) object

        Returns:
            - Dict[str, Any]: dictionary containing the load parameters
        """

        # add load parameters to dictionary based on load type.
        if isinstance(parameters, PointLoad):
            return self.__create_point_load_dict(part_name, parameters)
        elif isinstance(parameters, MovingLoad):
            return self.__create_moving_load_dict(part_name, parameters)
        elif isinstance(parameters, LineLoad):
            return self.__create_line_load_dict(part_name, parameters)
        elif isinstance(parameters, SurfaceLoad):
            return self.__create_surface_load_dict(part_name, parameters)
        elif isinstance(parameters, GravityLoad):
            return self.__create_gravity_load_dict(part_name, parameters)
        else:
            raise NotImplementedError
