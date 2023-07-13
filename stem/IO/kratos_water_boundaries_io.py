from typing import Dict, Any

from stem.water_boundaries import WaterBoundary, PhreaticMultiLineBoundary, InterpolateLineBoundary, WaterBoundaryParameters, PhreaticLine


class KratosWaterBoundariesIO:
    """
    Class to create the water boundary process dictionary for the ProjectParameters.json file in Kratos


    """

    def __init__(self, domain: str):
        """
        Constructor of KratosWaterBoundariesIO class

        Args:
            domain: Name of the Kratos domain


        """
        self.domain = domain

    def __create_phreatic_line_dict(self, name: str, type: str, water_boundary: PhreaticLine) -> Dict[str, Any]:
        boundary_dict_phreatic_line: Dict[str, Any] = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": {
                "model_part_name": f"{self.domain}.{name}",
                "variable_name": "WATER_PRESSURE",
                "is_fixed": water_boundary.is_fixed,
                "table": [0, 0],
                "fluid_pressure_type": type,
                "gravity_direction": water_boundary.gravity_direction,
                "out_of_plane_direction": water_boundary.out_of_plane_direction,
                "specific_weight": water_boundary.specific_weight,
                "first_reference_coordinate": water_boundary.first_reference_coordinate,
                "second_reference_coordinate": water_boundary.second_reference_coordinate,
                "value": water_boundary.value,
            }
        }
        return boundary_dict_phreatic_line

    def __create_water_boundary_dict(self, name: str, type: str, water_boundary: WaterBoundaryParameters) -> Dict[str, Any]:
        """
        Creates a dictionary containing the water boundary parameters

        Args:
            - name: name of the water boundary
            - type: type of the water boundary
            - water_boundary: water boundary object

        Returns:
            - Dict[str, Any]: dictionary containing the water boundary parameters

        """
        if isinstance(water_boundary, PhreaticMultiLineBoundary):
            parameters: Dict[str, Any] = {
                "model_part_name": f"{self.domain}.{name}",
                "variable_name": "WATER_PRESSURE",
                "table": [0, 0, 0],
                "value": water_boundary.water_pressure,
                "is_fixed": water_boundary.is_fixed,
                "gravity_direction": water_boundary.gravity_direction,
                "out_of_plane_direction": water_boundary.out_of_plane_direction,
                "fluid_pressure_type": type,
                "specific_weight": water_boundary.specific_weight,
                "x_coordinates": water_boundary.x_coordinates,
                "y_coordinates": water_boundary.y_coordinates,
                "z_coordinates": water_boundary.z_coordinates,
            }
            boundary_dict_multi_line: Dict[str, Any] = {
                "python_module": "apply_scalar_constraint_table_process",
                "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                "process_name": "ApplyScalarConstraintTableProcess",
                "Parameters": parameters,
            }
            return boundary_dict_multi_line
        elif isinstance(water_boundary, InterpolateLineBoundary):
            boundary_dict_interpolate: Dict[str, Any] = {
                "python_module": "apply_scalar_constraint_table_process",
                "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                "process_name": "ApplyScalarConstraintTableProcess",
                "Parameters": {
                    "model_part_name": f"{self.domain}.{name}",
                    "variable_name": "WATER_PRESSURE",
                    "is_fixed": water_boundary.is_fixed,
                    "table": 0,
                    "fluid_pressure_type": type,
                    "gravity_direction": water_boundary.gravity_direction,
                    "out_of_plane_direction": water_boundary.out_of_plane_direction,
                }
            }
            return boundary_dict_interpolate
        elif isinstance(water_boundary, PhreaticLine):
            temp_phreatic_line: PhreaticLine = water_boundary
            boundary_dict_phreatic_line: Dict[str, Any] = self.__create_phreatic_line_dict(name, type, temp_phreatic_line)
            return boundary_dict_phreatic_line
        else:
            raise NotImplementedError("This type of boundary is not implemented")

    def create_water_boundary_dict(self, water_boundary: WaterBoundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the water boundary parameters

        Args:
            - water_boundary: water boundary object

        Returns:
            - Dict[str, Any]: dictionary containing the water boundary parameters

        """
        return self.__create_water_boundary_dict(water_boundary.name,
                                                 water_boundary.water_boundary.type,
                                                 water_boundary.water_boundary)
