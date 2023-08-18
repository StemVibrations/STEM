from copy import deepcopy
from typing import Any, Dict, Union, List

from stem.boundary import *
from stem.table import Table


class KratosBoundariesIO:
    """
    Class containing methods to write boundary conditions to Kratos

    Attributes:
        - domain (str): name of the Kratos domain

    """

    @staticmethod
    def __create_value_and_table(part_name: str, parameters: BoundaryParametersABC):
        """
        Assemble values and tables for the boundary condition from the `value` attribute of the boundary parameters.
        Tables describe if an imposed displacement/rotation is time-dependant, and values if the displacement/rotation
        is at specific instant (or even fixed if is_fixed = True for the direction).
        For any direction either a table or a value are given, therefore if the type of `parameters.value` is
        `[float, Table, float]`, the returned tables and values sequences will be:
        `value = [float, 0, float]`
        `table = [0, Table, 0]`

        Args:
            - part_name (str): name of the model part on which the boundary is applied.
            - parameters (:class:`stem.boundary.BoundaryParametersABC`): boundary parameters object.

        Raises:
            - ValueError: if table ids are not initialised.
            - ValueError: when element in `parameters.value` is not of type int, float or Table.
            - ValueError: when provided parameters class doesn't implement values (e.g. AbsorbingBoundary).

        Returns:
            - _value (List[Union[float, int]]): list of values for the boundary condition (constraint)
            - _table (List[Union[float, int, :class:`stem.table.Table`]]): list of tables for the boundary condition \
                (constraint)
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

    def __init__(self, domain: str):
        """
        Constructor of KratosBoundariesIO class

        Args:
            - domain (str): name of the Kratos domain

        """
        self.domain = domain

    def __create_displacement_constraint_dict(
        self, part_name: str, parameters: DisplacementConstraint
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the displacement constraint parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.DisplacementConstraint`): displacement constraint parameters object
        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        boundary_dict["Parameters"]["variable_name"] = "DISPLACEMENT"

        # get tables and values
        _value, _table = self.__create_value_and_table(part_name, parameters)
        boundary_dict["Parameters"]["table"] = _table
        boundary_dict["Parameters"]["value"] = _value

        return boundary_dict

    def __create_rotation_constraint_dict(
        self, part_name: str, parameters: RotationConstraint
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the rotation constraint parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.RotationConstraint`): rotation constraint parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        boundary_dict["Parameters"]["variable_name"] = "ROTATION"

        # get tables and values
        _value, _table = self.__create_value_and_table(part_name, parameters)
        boundary_dict["Parameters"]["table"] = _table
        boundary_dict["Parameters"]["value"] = _value

        return boundary_dict

    def __create_absorbing_boundary_dict(
        self, part_name: str, parameters: AbsorbingBoundary
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the absorbing boundary parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.AbsorbingBoundary`): absorbing boundary parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "set_absorbing_boundary_parameters_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "SetAbsorbingBoundaryParametersProcess",
            "Parameters": parameters.__dict__,
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"

        return boundary_dict

    def create_boundary_condition_dict(
        self, part_name: str, parameters: BoundaryParametersABC
    ) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the boundary parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.BoundaryParametersABC`): boundary parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # add boundary parameters to dictionary based on boundary type.

        if isinstance(parameters, DisplacementConstraint):
            return self.__create_displacement_constraint_dict(part_name, parameters)
        elif isinstance(parameters, RotationConstraint):
            return self.__create_rotation_constraint_dict(part_name, parameters)
        elif isinstance(parameters, AbsorbingBoundary):
            return self.__create_absorbing_boundary_dict(part_name, parameters)
        else:
            raise NotImplementedError
