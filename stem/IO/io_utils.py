from typing import Sequence, Dict, Any, List, Union, Optional, Tuple
from pathlib import Path
import json
from copy import deepcopy

from stem.table import Table
from stem.boundary import BoundaryParametersABC
from stem.load import LoadParametersABC

class IOUtils:

    @staticmethod
    def write_json_file(output_folder: str, file_name: str, dictionary: Dict[Any, Any]):
        """
        Write a dictionary to a json file.

        Args:
            - output_folder (str): output folder path
            - file_name (str): name of the json file
            - dictionary (Dict[Any, Any]): dictionary to be written to the json file

        """
        # write json file
        output_folder_pth = Path(output_folder)
        output_folder_pth.mkdir(exist_ok=True, parents=True)

        output_path_file = output_folder_pth.joinpath(file_name)
        json.dump(dictionary, open(output_path_file, "w"), indent=4)

    @staticmethod
    def create_value_and_table(part_name: str, parameters: Union["LoadParametersABC", "BoundaryParametersABC"]) \
            -> Tuple[List[float], List[int]]:
        """
        Assemble values and tables for the boundary condition from the `value` attribute of the boundary parameters.
        If the displacement or rotation is time-dependent, a `table` is required. If the displacement or rotation is
        constant, a `value` is required. Each direction (x,y,z), requires either a `table` or a `value`. When a `table`
        is provided, the `value` is set to 0. If a `value` is provided, the `table` is set to 0.

        Args:
            - part_name (str): name of the model part on which the boundary is applied.
            - parameters (Union[:class:`stem.load.LoadParametersABC`, \
                                :class:`stem.boundary.BoundaryParametersABC`]): boundary parameters object.

        Raises:
            - ValueError: if table ids are not initialised.
            - ValueError: when element in `parameters.value` is not of type int, float or Table.
            - ValueError: when provided parameters class doesn't implement values (e.g. AbsorbingBoundary).

        Returns:
            - _value (List[float]): list of values for the boundary condition or load
            - _table (List[int]): list of table ids for the boundary condition \
                or load
        """

        _value: List[float] = []
        _table: List[int] = []

        if hasattr(parameters, "value"):

            # check the values per direction
            for vv in parameters.value:
                # if a table is provided, the value is set to 0
                if isinstance(vv, Table):
                    if vv.id is None:
                        raise ValueError(f"Table id is not initialised for values in {parameters.__class__.__name__}"
                                         f" in model part: {part_name}.")
                    _table.append(vv.id)
                    _value.append(0.0)
                # if a value is provided, the table is set to 0
                elif isinstance(vv, (int, float)):
                    _table.append(0)
                    _value.append(float(vv))
                else:
                    raise ValueError(f"'value' attribute in {parameters.__class__.__name__} in model part "
                                     f"`{part_name}`. The value ({vv}) is a `{vv.__class__.__name__}` object"
                                     f" but only a Table, float or integer are valid inputs.")

        else:
            raise ValueError(f"Attribute `value` does not exist in class: {parameters.__class__.__name__}.")

        return _value, _table

    @staticmethod
    def create_vector_constraint_table_process_dict(global_domain: str, part_name: str,
                                                    parameters: Union[LoadParametersABC, BoundaryParametersABC],
                                                    variable_name: str) -> Dict[str, Any]:
        """
        Creates a dictionary containing the vector constraint table process parameters. The vector constraint table
        process is used for loads and boundary conditions. This process applies either a constant value or a
        time-dependent table to the load or boundary condition.

        Args:
            - global_domain (str): name of the global domain
            - part_name (str): name of the model part on which the load is applied
            - parameters (Union[:class:`stem.load.LoadParametersABC`, \
                                :class:`stem.boundary.BoundaryParametersABC`]): boundary parameters object.
            - variable_name (str): name of the variable to which the boundary condition or load is applied

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

        load_dict["Parameters"]["model_part_name"] = f"{global_domain}.{part_name}"
        load_dict["Parameters"]["variable_name"] = variable_name

        # get tables and values
        _value, _table = IOUtils.create_value_and_table(part_name, parameters)
        load_dict["Parameters"]["table"] = _table
        load_dict["Parameters"]["value"] = _value

        return load_dict