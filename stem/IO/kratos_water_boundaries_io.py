from typing import Dict, List, Any
from copy import deepcopy

from stem.water_boundaries import WaterBoundary


class KratosWaterBoundariesIO:

    def __init__(self, domain: str):
        self.domain = domain

    def __phreatic_multi_line_boundary_dict(self, water_boundary: WaterBoundary):
        """
        Creates a dictionary containing the water boundary parameters for phreatic multi line boundary

        Attributes:
            - water_boundary: water boundary object

        Returns: dictionary containing the water boundary parameters

        """

        parameters : Dict[str, Any] = {
            "model_part_name": f"{self.domain}.{water_boundary.name}",
            "variable_name": "WATER_PRESSURE",
            "table": [0, 0, 0],
            "value": water_boundary.water_boundary.water_pressure,
            "is_fixed": water_boundary.water_boundary.is_fixed,
            "gravity_direction": water_boundary.water_boundary.gravity_direction,
            "out_of_plane_direction": water_boundary.water_boundary.out_of_plane_direction,
            "fluid_pressure_type": water_boundary.type,
            "specific_weight": water_boundary.water_boundary.specific_weight,
            "x_coordinates": water_boundary.water_boundary.x_coordinates,
            "y_coordinates": water_boundary.water_boundary.y_coordinates,
            "z_coordinates": water_boundary.water_boundary.z_coordinates,
        }
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": parameters,
        }
        return boundary_dict

    def __interpolate_line_boundary_dict(self, water_boundary: WaterBoundary):
        """
        Creates a dictionary containing the water boundary parameters for interpolate line boundary

        Attributes:
            - water_boundary:  water boundary object

        Returns: dictionary containing the water boundary parameters

        """
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": {
                "model_part_name": f"{self.domain}.{water_boundary.name}",
                "variable_name": "WATER_PRESSURE",
                "is_fixed": water_boundary.water_boundary.is_fixed,
                "table": 0,
                "fluid_pressure_type": water_boundary.type,
                "gravity_direction": water_boundary.water_boundary.gravity_direction,
                "out_of_plane_direction": water_boundary.water_boundary.out_of_plane_direction,
            }
        }
        return boundary_dict

    def create_water_boundary_dict(self, water_boundary: WaterBoundary):
        """
        Creates a dictionary containing the water boundary parameters

        Attributes:
            - water_boundary: water boundary object

        Returns: dictionary containing the water boundary parameters

        """
        if water_boundary.water_boundary.type == "Phreatic_Multi_Line":
            return self.__phreatic_multi_line_boundary_dict(water_boundary)
        elif water_boundary.water_boundary.type == "Interpolate_Line":
            return self.__interpolate_line_boundary_dict(water_boundary)
        else:
            raise ValueError(f"Unknown water boundary type: {water_boundary.water_boundary.type}")