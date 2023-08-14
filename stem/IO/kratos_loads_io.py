from copy import deepcopy
from typing import Dict, List, Any, Union
from stem.table import Table
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

    @staticmethod
    def __create_value_and_table(part_name:str, parameters: LoadParametersABC):
        """
        Assemble from the `value` attribute of the load parameters the values and tables for the load.
        Tables describe if a load is time-dependant, and values if the load is fixed.
        For any direction either a table or a value are given, therefore if the type of `parameters.value` is
        `[float, Table, float]`, the returned tables and values sequences will be:
        `value = [float, 0, float]`
        `table = [0, Table.id, 0]`

        Args:
            - part_name (str): name of the model part on which the load is applied.
            - parameters (:class:`stem.load.LoadParametersABC`): load parameters object.

        Raises:
            - ValueError: if table ids are not initialised.
            - ValueError: when element in `parameters.value` is not of type int, float or Table.
            - ValueError: when provided parameters class doesn't implement values (e.g. MovingLoad).

        Returns:
            - _value (List[Union[float, int]]): list of values for the load
            - _table (List[Union[float, int, :class:`stem.table.Table`]]): list of tables for the load
        """

        _value: List[Union[float, int]] = []
        _table: List[Union[float, int, Table]] = []

        if hasattr(parameters, "value"):

            for vv in parameters.value:
                if isinstance(vv, Table):
                    if vv.id is None:
                        raise ValueError(f"Table id is not initialised for values in {parameters.__class__.__name__}"
                                         f" for part {part_name}.")
                    _table.append(vv.id)
                    _value.append(0)
                elif isinstance(vv, (int, float)):
                    _table.append(0)
                    _value.append(vv)
                else:
                    raise ValueError(f"Value in parameters `value` is not either a Table object,a float or "
                                     f"integer from class {parameters.__class__.__name__} for part {part_name}.")
        else:
            raise ValueError(f"Attribute `value` is not implemented by class {parameters.__class__.__name__}.")

        return _value, _table

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

        # get tables and values
        _value, _table = self.__create_value_and_table(part_name, parameters)
        load_dict["Parameters"]["table"] = _table
        load_dict["Parameters"]["value"] = _value

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

        # get tables and values
        _value, _table = self.__create_value_and_table(part_name, parameters)
        load_dict["Parameters"]["table"] = _table
        load_dict["Parameters"]["value"] = _value

        return load_dict

    def __create_surface_load_dict(self, part_name:str, parameters: LoadParametersABC) -> Dict[str, Any]:
        """
        Creates a dictionary containing the surface load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.SurfaceLoad`): surface load parameters object

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
        load_dict["Parameters"]["variable_name"] = "SURFACE_LOAD"

        # get tables and values
        _value, _table = self.__create_value_and_table(part_name, parameters)
        load_dict["Parameters"]["table"] = _table
        load_dict["Parameters"]["value"] = _value

        return load_dict

    def __create_gravity_load_dict(self, part_name:str, parameters: GravityLoad) -> Dict[str, Any]:
        """
        Creates a dictionary containing the surface load parameters

        Args:
            - part_name (str): name of the model part on which the load is applied
            - parameters (:class:`stem.load.GravityLoad`): gravity load parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the load parameters
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

        # get tables and values
        _value, _table = self.__create_value_and_table(part_name, parameters)
        load_dict["Parameters"]["table"] = _table
        load_dict["Parameters"]["value"] = _value

        return load_dict

    def create_load_dict(self, part_name:str, parameters: LoadParametersABC) -> Union[Dict[str, Any], None]:
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
            raise NotImplementedError(f"Load type {type(parameters)} not implemented")
