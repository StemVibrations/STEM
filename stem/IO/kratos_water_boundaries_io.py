from typing import Dict, Any

from stem.water_boundaries import WaterBoundary, PhreaticMultiLineBoundary, InterpolateLineBoundary


class KratosWaterBoundariesIO:

    def __init__(self, domain: str):
        self.domain = domain

    def __phreatic_multi_line_boundary_dict(self, name: str, type: str, water_boundary: PhreaticMultiLineBoundary):
        """
        Creates a dictionary containing the water boundary parameters for phreatic multi line boundary

        Attributes:
            - water_boundary: water boundary object

        Returns: dictionary containing the water boundary parameters

        """

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
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": parameters,
        }
        return boundary_dict

    def __interpolate_line_boundary_dict(self, name: str, type: str, water_boundary: InterpolateLineBoundary):
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
                "model_part_name": f"{self.domain}.{name}",
                "variable_name": "WATER_PRESSURE",
                "is_fixed": water_boundary.is_fixed,
                "table": 0,
                "fluid_pressure_type": type,
                "gravity_direction": water_boundary.gravity_direction,
                "out_of_plane_direction": water_boundary.out_of_plane_direction,
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
        if water_boundary.water_boundary.__class__ == PhreaticMultiLineBoundary:
            multi_line_boundary: PhreaticMultiLineBoundary = water_boundary.water_boundary
            return self.__phreatic_multi_line_boundary_dict(water_boundary.name, water_boundary.water_boundary.type,
                                                            multi_line_boundary)
        elif water_boundary.water_boundary.__class__ == InterpolateLineBoundary:
            interpolate_line_boundary: InterpolateLineBoundary = water_boundary.water_boundary
            return self.__interpolate_line_boundary_dict(water_boundary.name, water_boundary.water_boundary.type,
                                                         interpolate_line_boundary)
        else:
            raise ValueError(f"Unknown water boundary type: {water_boundary.water_boundary.type}")
