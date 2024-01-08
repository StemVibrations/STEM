from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from stem.globals import GRAVITY_VALUE, VERTICAL_AXIS
from stem.solver import AnalysisType
from stem.table import Table
from stem.utils import Utils

@dataclass
class LoadParametersABC(ABC):
    """
    Abstract base class for load parameters
    """

    @staticmethod
    @abstractmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Abstract static method to get the element name for a load.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per condition-element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - Exception: abstract method is called

        """
        raise Exception("abstract method 'get_element_name' of load parameters class is called")


@dataclass
class PointLoad(LoadParametersABC):
    """
    Class containing the load parameters for a point load

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction.
        - value (List[Union[float, :class:`stem.table.Table`]]): Entity of the load in the 3 directions [N]. \
            It should be a list of either float or table for each load. If a float is specified, the \
            load is time-independent, otherwise the table specifies the amplitude of the \
            load [N] over time [s] for each direction.
    """

    active: List[bool]
    value: List[Union[float, Table]]

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a point load. Point load does not have a name.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per condition-element (1)
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type

        Raises:
            - ValueError: If the analysis type is not mechanical or mechanical groundwater flow

        Returns:
            - Optional[str]: The element name for a point load

        """

        available_node_dim_combinations = {
            2: [1],
            3: [1],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Point load")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"PointLoadCondition{n_dim_model}D{n_nodes_element}N"

        else:
            raise ValueError("Point load can only be applied in mechanical or mechanical groundwater flow analysis")

        # Point load does not have a name
        return element_name


@dataclass
class LineLoad(LoadParametersABC):
    """
    Class containing the load parameters for a line load

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction.
        - value (List[Union[float, :class:`stem.table.Table`]]): Entity of the load in the 3 directions [N/m]. \
            It should be a list of either float or table for each load. If a float is specified, the \
            load is time-independent, otherwise the table specifies the amplitude of the \
            load [N/m] over time [s] for each direction.
    """
    active: List[bool]
    value: List[Union[float, Table]]

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a line load.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per condition-element (2, 3)
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type

        Raises:
            - ValueError: If the analysis type is not mechanical or mechanical groundwater flow

        Returns:
            - Optional[str]: The element name for a line load

        """

        available_node_dim_combinations = {
            2: [2, 3],
            3: [2, 3],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Line load")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            if n_dim_model == 2 and n_nodes_element > 2:
                # 2d quadratic line load is set on outer nodes, but displacement is calculated on all nodes for
                # stability reasons
                element_name = f"LineLoadDiffOrderCondition{n_dim_model}D{n_nodes_element}N"
            else:
                element_name = f"LineLoadCondition{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError("Line load can only be applied in mechanical or mechanical groundwater flow analysis")

        return element_name


@dataclass
class SurfaceLoad(LoadParametersABC):
    """
    Class containing the load parameters for a surface load

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction.
        - value (List[Union[float, :class:`stem.table.Table`]]): Entity of the load in the 3 directions [Pa]. \
            It should be a list of either float or table for each load. If a float is specified, the \
            load is time-independent, otherwise the table specifies the amplitude of the \
            load [Pa] over time [s] for each direction.
    """
    active: List[bool]
    value: Union[List[float], List[Table]]

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a surface load.

        Args:
            - n_dim_model (int): The number of dimensions of the model (3)
            - n_nodes_element (int): The number of nodes per condition-element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type

        Raises:
            - ValueError: If the analysis type is not mechanical or mechanical groundwater flow

        Returns:
            - Optional[str]: The element name for a surface load
        """

        available_node_dim_combinations = {
            3: [3, 4, 6, 8],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Surface load")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            if n_nodes_element == 3 or n_nodes_element == 4:
                element_name = f"UPwFaceLoadCondition{n_dim_model}D{n_nodes_element}N"
            else:
                element_name = f"SurfaceLoadDiffOrderCondition{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError("Surface load can only be applied in mechanical or mechanical groundwater flow analysis")

        return element_name


@dataclass
class MovingLoad(LoadParametersABC):
    """
    Class containing the load parameters for a moving load.

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - load (Union[List[float], List[str]]): Entity of the load [N] in the 3 \
               directions. Can be defined as strings (when function of time) or as float. \
               Mixed types are not accepted.
        - direction (List[int]):  Direction of the moving load (-1 or +1 in x, y, z direction) [-].
        - velocity (Union[float, str]): Velocity of the moving load [m/s].
        - origin (List[float]): Starting coordinates of the moving load [m].
        - offset (float): Offset of the moving load [m].
    """

    load: Union[List[float], List[str]]
    direction: List[float]
    velocity: Union[float, str]
    origin: List[float]
    offset: float = 0.0

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a moving load.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per condition-element (2, 3)
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type

        Raises:
            - ValueError: If the analysis type is not mechanical or mechanical groundwater flow

        Returns:
            - Optional[str]: The element name for a moving load
        """

        available_node_dim_combinations = {
            2: [2, 3],
            3: [2, 3],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Moving load")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"MovingLoadCondition{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError("Moving load can only be applied in mechanical or mechanical groundwater flow analysis")

        return element_name

@dataclass
class UvecLoad(LoadParametersABC):
    """
    Class containing the load parameters for a UVEC (User-defined VEhiCle) load.

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - direction (List[int]):  Direction of the moving load (-1 or +1 in x, y, z direction) [-].
        - velocity (Union[float, str]): Velocity of the moving load [m/s].
        - origin (List[float]): Starting coordinates of the first wheel [m].
        - wheel_configuration (List[float]): Wheel configuration, i.e. distances from the origin of each wheel [m].
        - uvec_file (str): Path to the UVEC file.
        - uvec_function_name (str): Name of the UVEC function.
        - uvec_parameters (Dict[str, Any]): Parameters of the UVEC function.
        - uvec_state_variables (Dict[str, Any]): State variables of the UVEC function.


    """

    direction: List[float]
    velocity: Union[float, str]
    origin: List[float]
    wheel_configuration: List[float]
    uvec_file: str
    uvec_function_name: str
    uvec_parameters: Dict[str, Any] = field(default_factory=dict)
    uvec_state_variables: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a UVEC load.
        """

        # check if the number of nodes per element is correct
        available_node_dim_combinations = {
            2: [2, 3],
            3: [2, 3],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "UVEC load")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"MovingLoadCondition{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError("UVEC load can only be applied in mechanical or mechanical groundwater flow analysis")

        return element_name

@dataclass
class GravityLoad(LoadParametersABC):
    """
    Class containing the load parameters for a gravity load.

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate load for each direction. Input True only in the vertical direction.
        - value (List[float]): Entity of the gravity acceleration in the 3 directions [m/s^2]. Should be -9.81 only in \
            the vertical direction
    """
    active: List[bool] = field(default_factory=lambda: [True, True, True])
    value: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        """
        Adds global gravity acceleration if it is not defined
        """
        if np.allclose(self.value, [0, 0, 0]):
            self.value[VERTICAL_AXIS] = GRAVITY_VALUE

    @staticmethod
    def get_element_name(n_dim_model, n_nodes_element, analysis_type) -> Optional[str]:
        """
        Static method to get the element name for a gravity load.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type

        Raises:
            - ValueError: If the analysis type is not mechanical or mechanical groundwater flow

        Returns:
            - None: Gravity load does not have a name
        """

        if analysis_type != AnalysisType.MECHANICAL_GROUNDWATER_FLOW and analysis_type != AnalysisType.MECHANICAL:
            raise ValueError("Point load can only be applied in mechanical or mechanical groundwater flow analysis")

        # Gravity load does not have a name
        return None
