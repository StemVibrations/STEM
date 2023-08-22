from copy import deepcopy
from typing import Any, Dict, Union, List

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
            "Parameters": deepcopy(parameters.__dict__),
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
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters,
                                                                       "DISPLACEMENT")
        elif isinstance(parameters, RotationConstraint):
            return IOUtils.create_vector_constraint_table_process_dict(self.domain, part_name, parameters,
                                                                       "ROTATION")
        elif isinstance(parameters, AbsorbingBoundary):
            return self.__create_absorbing_boundary_dict(part_name, parameters)
        else:
            raise NotImplementedError
