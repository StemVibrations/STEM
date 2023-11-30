from typing import Any, Dict, Union, List

from stem.water_processes import *


class KratosWaterProcessesIO:
    """
    Class containing methods to write water processes to Kratos

    Attributes:
        - domain (str): name of the Kratos domain

    """

    def __init__(self, domain: str):
        """
        Constructor of KratosWaterProcessesIO class

        Args:
            - domain (str): name of the Kratos domain

        """
        self.domain = domain

    def __create_uniform_water_pressure_dict(
        self, part_name: str, parameters: UniformWaterPressure
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the uniform water pressure parameters

        Args:
            - part_name (str): part name where the water process is applied
            - parameters (:class:`stem.water_processes.UniformWaterPressure`): uniform water pressure parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the uniform water pressure process parameters
        """

        # initialize boundary dictionary
        water_dict: Dict[str, Any] = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": {"model_part_name": f"{self.domain}.{part_name}",
                           "variable_name": "WATER_PRESSURE",
                           "table": 0,
                           "value": parameters.water_pressure,
                           "is_fixed": parameters.is_fixed,
                           "fluid_pressure_type": "Uniform"}
        }

        return water_dict

    def create_water_process_dict(
        self, part_name: str, parameters: WaterProcessParametersABC
    ) -> Union[Dict[str, Any], None]:
        """
        Creates a dictionary containing the water process parameters

        Args:
            - part_name (str): part name where the water process is applied
            - parameters (:class:`stem.water_processes.UniformWaterPressure`): water process object

        Returns:
            - Dict[str, Any]: dictionary containing the water process parameters
        """

        # add water parameters to dictionary based on water process type.
        if isinstance(parameters, UniformWaterPressure):
            return self.__create_uniform_water_pressure_dict(part_name, parameters)

        else:
            raise NotImplementedError(f"Water boundary type: {parameters.__class__.__name__} not implemented.")
