from copy import deepcopy
from typing import List, Dict, Tuple, Any

from stem.boundary import *


class KratosBoundariesIO:
    """
    Class containing methods to write boundary conditions to Kratos

    Attributes:
        domain (str): name of the Kratos domain

    """

    def __init__(self, domain: str):
        """
        Constructor of KratosBoundariesIO class

        Args:
            domain (str): name of the Kratos domain

        """
        self.domain = domain

    def __create_displacement_constraint_dict(
        self, part_name: str, boundary_parameters: DisplacementConstraint
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the displacement constraint parameters

        Args:
            part_name (str): part name where the boundary condition is applied
            boundary_parameters (DisplacementConstraint): displacement constraint
                parameters object
        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(boundary_parameters.__dict__),
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        boundary_dict["Parameters"]["variable_name"] = "DISPLACEMENT"
        boundary_dict["Parameters"]["table"] = [0, 0, 0]

        return boundary_dict

    def __create_rotation_constraint_dict(
        self, part_name: str, boundary_parameters: RotationConstraint
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the rotation constraint parameters

        Args:
            part_name (str): part name where the boundary condition is applied
            boundary_parameters (RotationConstraint): rotation constraint parameters
                object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": deepcopy(boundary_parameters.__dict__),
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"
        boundary_dict["Parameters"]["variable_name"] = "ROTATION"
        boundary_dict["Parameters"]["table"] = [0, 0, 0]

        return boundary_dict

    def __create_absorbing_boundary_dict(
        self, part_name: str, boundary_parameters: AbsorbingBoundary
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the absorbing boundary parameters

        Args:
            part_name (str): part name where the boundary condition is applied
            boundary_parameters (AbsorbingBoundary): absorbing boundary parameters
                object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "set_absorbing_boundary_parameters_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "SetAbsorbingBoundaryParametersProcess",
            "Parameters": boundary_parameters.__dict__,
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{self.domain}.{part_name}"

        return boundary_dict

    def __create_boundary_dict(self, boundary: Boundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the boundary parameters

        Args:
            boundary (Load): boundary object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # add boundary parameters to dictionary based on boundary type.

        if isinstance(boundary.boundary_parameters, DisplacementConstraint):
            return self.__create_displacement_constraint_dict(**boundary.__dict__)
        elif isinstance(boundary.boundary_parameters, RotationConstraint):
            return self.__create_rotation_constraint_dict(**boundary.__dict__)
        elif isinstance(boundary.boundary_parameters, AbsorbingBoundary):
            return self.__create_absorbing_boundary_dict(**boundary.__dict__)
        else:
            raise NotImplementedError

    def create_dictionaries_for_boundaries(
        self, boundaries: List[Boundary]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Creates a dictionary containing the `constraint_process_list` (list of
        dictionaries to specify the constraints for the model) and a list of
        dictionaries for the absorbing boundaries to be given to `load_process_list`.

        Args:
            boundaries (List[Boundary]): list of boundary objects.

        Returns:
            constraints_dict (Dict[str, Any]): dictionary of a list containing the
                constraints acting on the model.
            load_boundaries (List[Dict[str, Any]]): dictionary of a list
                containing the non-constraint (load) boundaries of the model.
        """

        constraints_dict: Dict[str, Any] = {"constraints_process_list": []}
        load_boundaries: List[Dict[str, Any]] = []

        for boundary in boundaries:
            boundary_dict = self.__create_boundary_dict(boundary)
            if boundary.boundary_parameters.is_constraint:
                constraints_dict["constraints_process_list"].append(boundary_dict)
            else:
                load_boundaries.append(boundary_dict)

        return constraints_dict, load_boundaries
