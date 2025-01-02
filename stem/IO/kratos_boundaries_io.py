from copy import deepcopy
from typing import Any, Dict, Union

from stem.boundary import *
from stem.IO.io_utils import IOUtils


class KratosBoundariesIO:
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

    def __create_absorbing_boundary_dict(self, part_name: str, parameters: AbsorbingBoundary,
                                         use_linear_elastic_strategy: bool) -> Dict[str, Any]:
        """
        Creates a dictionary containing the absorbing boundary parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.AbsorbingBoundary`): absorbing boundary parameters object
            - use_linear_elastic_strategy (bool): flag to determine if the linear elastic strategy is used

        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "set_absorbing_boundary_parameters_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "SetAbsorbingBoundaryParametersProcess",
            "Parameters": deepcopy(parameters.__dict__),
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        boundary_dict["Parameters"]["skip_internal_forces"] = use_linear_elastic_strategy

        return boundary_dict

    def create_boundary_condition_dict(self, part_name: str, parameters: BoundaryParametersABC, current_time: float,
                                       use_linear_elastic_strategy: bool) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the boundary parameters

        Args:
            - part_name (str): part name where the boundary condition is applied
            - parameters (:class:`stem.boundary.BoundaryParametersABC`): boundary parameters object
            - current_time (float): current time of the analysis
            - use_linear_elastic_strategy (bool): flag to determine if the linear elastic strategy is used

        Returns:
            - Dict[str, Any]: dictionary containing the boundary parameters
        """

        # add boundary parameters to dictionary based on boundary type.

        if isinstance(parameters, DisplacementConstraint):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters,
                                                                       "DISPLACEMENT", current_time)
        elif isinstance(parameters, RotationConstraint):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters, "ROTATION",
                                                                       current_time)
        elif isinstance(parameters, AbsorbingBoundary):
            return self.__create_absorbing_boundary_dict(part_name, parameters, use_linear_elastic_strategy)
        else:
            raise NotImplementedError
