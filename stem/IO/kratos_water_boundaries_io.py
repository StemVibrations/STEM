from typing import Dict, Any
from typing import overload

from stem.water_boundaries import WaterBoundary, PhreaticMultiLineBoundary, InterpolateLineBoundary


class KratosWaterBoundariesIO:

    def __init__(self, domain: str):
        self.domain = domain

    @overload
    def __water_boundary_dict(self, name: str, type: str, water_boundary: PhreaticMultiLineBoundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the water boundary parameters for phreatic multi line boundary

        Attributes:
            - water_boundary: water boundary object
            - type: type of the water boundary
            - name: name of the water boundary

        Returns: dictionary containing the water boundary parameters

        """
        ...

    @overload
    def __water_boundary_dict(self, name: str, type: str, water_boundary: InterpolateLineBoundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the water boundary parameters for interpolate line boundary

        Attributes:
            - water_boundary:  water boundary object
            - type: type of the water boundary
            - name: name of the water boundary

        Returns: dictionary containing the water boundary parameters


        """
        ...

    def __water_boundary_dict(self, name: str, type: str, water_boundary: Any) -> Dict[str, Any]:
        """
        Creates a dictionary containing the water boundary parameters

        Attributes:
            - name: name of the water boundary
            - type: type of the water boundary
            - water_boundary: water boundary object

        Returns: None at the moment

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
        else:
            raise NotImplementedError("This type of boundary is not implemented")

    def create_water_boundary_dict(self, water_boundary: WaterBoundary):
        """
        Creates a dictionary containing the water boundary parameters

        Attributes:
            - water_boundary: water boundary object

        Returns: dictionary containing the water boundary parameters

        """
        return self.__water_boundary_dict(water_boundary.name,
                                          water_boundary.water_boundary.type,
                                          water_boundary.water_boundary)
