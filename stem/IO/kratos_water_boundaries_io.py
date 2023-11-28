from copy import deepcopy
from typing import Any, Dict, Union, List

from stem.water_boundaries import *
from stem.IO.io_utils import IOUtils


class KratosWaterBoundariesIO:
    """
    Class containing methods to write boundary conditions to Kratos

    Attributes:
        - domain (str): name of the Kratos domain

    """

    def __init__(self, domain: str):
        """
        Constructor of KratosBoundariesIO class

        Args:
            - domain (str): name of the Kratos domain

        """
        self.domain = domain

    def __create_uniform_water_pressure_dict(
        self, part_name: str, parameters: UniformWaterBoundary
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
        water_dict: Dict[str, Any] = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": {"model_part_name": f"{self.domain}.{part_name}",
                           "variable_name": "WATER_PRESSURE",
                           "table": 0,
                           "value": parameters.WATER_PRESSURE,
                           "is_fixed": True,
                           "fluid_pressure_type": "Uniform"}
        }

        return water_dict

    def create_water_boundary_condition_dict(
        self, part_name: str, parameters: WaterBoundaryParametersABC
    ) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the boundary parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.BoundaryParametersABC`): boundary parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # add water parameters to dictionary based on water boundary type.

        if isinstance(parameters, UniformWaterBoundary):
            return self.__create_uniform_water_pressure_dict(part_name, parameters)

        else:
            raise NotImplementedError
