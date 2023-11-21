from copy import deepcopy
from typing import Dict, List, Any, Union
from stem.load import *
from stem.IO.io_utils import IOUtils


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

    def __create_moving_load_dict(self, part_name: str, parameters: MovingLoad) -> Dict[str, Any]:
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

    def __create_uvec_dict(self, part_name: str, parameters: UvecLoad) -> Dict[str, Any]:
        """
        Creates a dictionary containing the UVEC parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.UvecLoad`): UVEC load parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the UVEC load parameters

        """

        # initialize load parameters dictionary
        parameters_dict = {"model_part_name":  f"{self.domain}.{part_name}",
                           "compute_model_part_name": f"porous_computational_model_part",  # as hard-coded in Kratos
                           "variable_name": "POINT_LOAD",
                           "load": [1, 1, 1],  # dummy parameter
                           "direction": parameters.direction,
                           "velocity": parameters.velocity,
                           "origin": parameters.origin,
                           "configuration": parameters.wheel_configuration}

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "set_multiple_moving_loads_process",
            "kratos_module": "StemApplication",
            "process_name": "SetMultipleMovingLoadsProcess",
            "Parameters": parameters_dict
        }

        return load_dict

    def create_load_dict(self, part_name: str, parameters: LoadParametersABC) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.LoadParametersABC`): load parameters object

        Raises:
            - NotImplementedError: if the load type is not implemented

        Returns:
            - Dict[str, Any]: dictionary containing the load parameters
        """

        # add load parameters to dictionary based on load type.
        if isinstance(parameters, PointLoad):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters, "POINT_LOAD")
        elif isinstance(parameters, MovingLoad):
            return self.__create_moving_load_dict(part_name, parameters)
        elif isinstance(parameters, UvecLoad):
            return self.__create_uvec_dict(part_name, parameters)
        elif isinstance(parameters, LineLoad):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters, "LINE_LOAD")
        elif isinstance(parameters, SurfaceLoad):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters,
                                                                       "SURFACE_LOAD")
        elif isinstance(parameters, GravityLoad):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters,
                                                                       "VOLUME_ACCELERATION")
        else:
            raise NotImplementedError(f"Load type {type(parameters)} not implemented")
