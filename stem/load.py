import os
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from types import ModuleType

import numpy as np

from stem.globals import GlobalSettings, VERTICAL_AXIS
from stem.solver import AnalysisType
from stem.table import Table
from stem.utils import Utils


class TrainType(Enum):
    """
    Enum class containing the supported train types.

    Inheritance:
        - :class:`Enum`
    """
    CUSTOM = "custom"
    LOCOMOTIVE = "locomotive"
    PASSENGERS_HEAVY = "passengers_heavy"
    PASSENGERS_LIGHT = "passengers_light"
    FREIGHT_LOADED = "freight_loaded"
    FREIGHT_UNLOADED = "freight_unloaded"


_TRAIN_PARAMETER_PRESETS: Dict[TrainType, Dict[str, Any]] = {
    TrainType.LOCOMOTIVE: {
        "cart_mass": 5.5e4 / 2,  # mass of the cart [kg]
        "bogie_mass": 6e3 / 2,  # mass of the bogie [kg] / primary suspension mass [kg]
        "wheel_mass": 4.5e3 / 2,  # mass of the wheel [kg] / secondary suspension mass [kg]
        "cart_inertia": 9.7e5 / 2,  # inertia of the cart [kgm2]
        "bogie_inertia": 7.8e3 / 2,  # inertia of the bogie [kgm2] / primary suspension inertia [kgm2]
        "cart_stiffness": 8e5,  # stiffness between the cart and bogies [N/m] / primary suspension
        "cart_damping": 4.5e4,  # damping coefficient between the cart and bogies [Ns/m] / primary suspension
        "wheel_stiffness": 2e6,  # stiffness between the wheel and the bogie [N/m] / secondary suspension
        "wheel_damping": 3.3e4,  # damping coefficient between the wheel and the bogie [Ns/m] / secondary suspension
        "bogie_distances": [-10.4, 10.4],  # distances of the bogies from the centre of the cart [m]
        "wheel_distances": [-1.3, 1.3],  # distances of the wheels from the centre of the bogie [m]
        "cart_length": 26.8,  # length of the train [m]
        "gravity_axis": 1,  # axis on which gravity works [x =0, y = 1, z = 2]
        "contact_coefficient": 5.13e-08,  # Hertzian contact coefficient between the wheel and the rail [N/m]
        "contact_power": 1.5,  # Hertzian contact power between the wheel and the rail [-]
        "wheel_configuration": [0, 2.6, 20.8, 23.4],
    },
    TrainType.PASSENGERS_HEAVY: {
        "cart_mass": 3.8e4 / 2,
        "bogie_mass": 3.1e3 / 2,
        "wheel_mass": 1.9e3 / 2,
        "cart_inertia": 2.3e6 / 2,
        "bogie_inertia": 2.4e3 / 2,
        "cart_stiffness": 3.5e5,
        "cart_damping": 3.2e4,
        "wheel_stiffness": 1.2e6,
        "wheel_damping": 2.5e3,
        "bogie_distances": [-10, 10],
        "wheel_distances": [-1.25, 1.25],
        "cart_length": 27,
        "gravity_axis": 1,
        "contact_coefficient": 5.13e-08,
        "contact_power": 1.5,
        "wheel_configuration": [0, 2.5, 20, 22.5],
    },
    TrainType.PASSENGERS_LIGHT: {
        "cart_mass": 2.9e4 / 2,
        "bogie_mass": 2.6e3 / 2,
        "wheel_mass": 1.7e3 / 2,
        "cart_inertia": 1.8e6 / 2,
        "bogie_inertia": 2.3e3 / 2,
        "cart_stiffness": 4e5,
        "cart_damping": 2.2e4,
        "wheel_stiffness": 7.4e5,
        "wheel_damping": 6.4e3,
        "bogie_distances": [-9.5, 9.5],
        "wheel_distances": [-1.28, 1.28],
        "cart_length": 25,
        "gravity_axis": 1,
        "contact_coefficient": 5.13e-08,
        "contact_power": 1.5,
        "wheel_configuration": [0, 2.56, 19, 21.56],
    },
    TrainType.FREIGHT_UNLOADED: {
        "cart_mass": 1.3e4 / 2,
        "bogie_mass": 1.8e3 / 2,
        "wheel_mass": 1.4e3 / 2,
        "cart_inertia": 2.3e5 / 2,
        "bogie_inertia": 1.7e3 / 2,
        "cart_stiffness": 6e6,
        "cart_damping": 6.5e4,
        "wheel_stiffness": 5e5,
        "wheel_damping": 5.7e3,
        "bogie_distances": [-7, 7],
        "wheel_distances": [-0.9, 0.9],
        "cart_length": 17,
        "gravity_axis": 1,
        "contact_coefficient": 5.13e-08,
        "contact_power": 1.5,
        "wheel_configuration": [0, 1.8, 14, 15.8],
    },
    TrainType.FREIGHT_LOADED: {
        "cart_mass": 8.1e4 / 2,
        "bogie_mass": 1.8e3 / 2,
        "wheel_mass": 1.4e3 / 2,
        "cart_inertia": 7.3e5 / 2,
        "bogie_inertia": 1.7e3 / 2,
        "cart_stiffness": 6e6,
        "cart_damping": 6.5e4,
        "wheel_stiffness": 2.6e6,
        "wheel_damping": 1.5e4,
        "bogie_distances": [-7, 7],
        "wheel_distances": [-0.9, 0.9],
        "cart_length": 17,
        "gravity_axis": 1,
        "contact_coefficient": 5.13e-08,
        "contact_power": 1.5,
        "wheel_configuration": [0, 1.8, 14, 15.8],
    },
}

_REQUIRED_TRAIN_PARAMETER_KEYS = {
    "cart_mass",
    "bogie_mass",
    "wheel_mass",
    "cart_inertia",
    "bogie_inertia",
    "cart_stiffness",
    "cart_damping",
    "wheel_stiffness",
    "wheel_damping",
    "bogie_distances",
    "wheel_distances",
    "cart_length",
    "gravity_axis",
    "contact_coefficient",
    "contact_power",
    "wheel_configuration",
}


def _validate_train_parameters(parameters: Dict[str, Any]) -> None:
    """
    Validate that all required train parameters for the CUSTOM train type are present.

    Args:
        parameters (Dict[str, Any]): The train parameters to validate.

    Raises:
        ValueError: If any required train parameters are missing.
    """
    missing_keys = _REQUIRED_TRAIN_PARAMETER_KEYS.difference(parameters)
    if missing_keys:
        raise ValueError(f"Missing train parameters: {sorted(missing_keys)}")


def _build_train_parameters(train_type: TrainType, nb_carts: int, offset: float, velocity: Union[float, str],
                            static_initialisation: bool, initialisation_steps: Optional[int],
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialise the parameters of the train based on the train type.
    If the train type is custom, the parameters must be provided by the user.
    The train parameters are for 2D half train model, and are available on :cite:`Ricardo_2025`.

    Args:
    - train_type (TrainType): The type of the train. If CUSTOM, the parameters must be provided by the user.
    - nb_carts (int): The number of carts in the train. Must be >= 1.
    - offset (float): The offset of the train from the origin [m].
    - velocity (Union[float, str]): The velocity of the train [m/s]. Can be defined as a string \
      (when function of time) or as a float.
    - static_initialisation (bool): Whether to perform a static initialisation of the train.
    - initialisation_steps (Optional[int]): The number of steps to perform for the static initialisation.
    - parameters (Dict[str, Any]): The train parameters to use if the train type is CUSTOM.

    Raises:
    - TypeError: If nb_carts is not an integer.
    - ValueError: If nb_carts is less than 1.

    Returns:
    - Dict[str, Any]: The parameters of the train to be used in the UVEC function.
    """

    if not isinstance(nb_carts, int):
        raise TypeError("nb_carts must be an integer")

    if nb_carts < 1:
        raise ValueError("nb_carts must be >= 1")

    if train_type == TrainType.CUSTOM:
        _validate_train_parameters(parameters)
        uvec_parameters = parameters
    else:
        uvec_parameters = _TRAIN_PARAMETER_PRESETS[train_type]

    uvec_parameters["n_carts"] = nb_carts
    uvec_parameters["velocity"] = velocity
    uvec_parameters["static_initialisation"] = static_initialisation
    uvec_parameters["initialisation_steps"] = initialisation_steps

    # compute the wheel configuration based on the number of carts and the wheel configuration of each cart
    total_wheel_configuration = [u + offset for u in uvec_parameters["wheel_configuration"]]
    for n in range(1, nb_carts):
        total_wheel_configuration.extend(
            [u + n * uvec_parameters["cart_length"] for u in uvec_parameters["wheel_configuration"]])
    uvec_parameters["wheel_configuration"] = total_wheel_configuration

    return uvec_parameters


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
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations, "Line load")

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
        - direction_signs (List[int]): Sign of motion along each axis (+1 or -1 for x, y, z).
               The actual direction is defined by the existing load path geometry.
        - velocity (Union[float, str]): Velocity of the moving load [m/s].
        - origin (List[float]): Starting coordinates of the moving load [m].
        - offset (float): Offset of the moving load [m].
    """

    load: Union[List[float], List[str]]
    direction_signs: List[int]
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


class UvecSupportedModels(Enum):
    """
    Enum class containing the supported UVEC models.

    Inheritance:
        - :class:`Enum`
    Attributes:
        - TEN_DOF (str): 10 degrees of freedom UVEC model.
        - TWO_DOF (str): 2 degrees of freedom UVEC model.
    """
    TEN_DOF = "UVEC.uvec_ten_dof_vehicle_2D"
    TWO_DOF = "UVEC.uvec_two_dof_vehicle_2D"


@dataclass
class UvecLoad(LoadParametersABC):
    """
    Class containing the load parameters for a UVEC (User-defined VEhiCle) load.

    Inheritance:
        - :class:`LoadParametersABC`

    Attributes:
        - direction_signs (List[int]): Sign of motion along each axis (+1 or -1 for x, y, z).
               The actual direction is defined by the existing load path geometry.
        - velocity (Union[float, str]): Velocity of the moving load [m/s].
        - origin (List[float]): Starting coordinates of the first wheel [m].
        - uvec_parameters (Dict[str, Any]): Parameters of the UVEC function.
        - uvec_state_variables (Dict[str, Any]): State variables of the UVEC function.
        - uvec_model (ModuleType): UVEC model.
        - uvec_file (str): Path to the UVEC file.
        - uvec_function_name (str): Name of the UVEC function.
        - nb_carts (int): Number of carts in the UVEC model.
        - offset (float): Offset of the first wheel of the load, in relation to the origin [m].
        - train_type (TrainType): Type of the train. If CUSTOM, the uvec_parameters must be provided by the user. \
          If not CUSTOM, the uvec_parameters are defined based on the train type.
        - irregularities (Optional[Dict[str, Any]]): Parameters of the track irregularities to be included in the \
          UVEC model.
        - rail_joint (Optional[Dict[str, Any]]): Parameters of the rail joint to be included in the UVEC model.
        - static_initialisation (bool): Whether to perform a static initialisation of the train.
    """

    direction_signs: List[int]
    velocity: Union[float, str]
    origin: List[float]
    uvec_parameters: Dict[str, Any] = field(default_factory=dict)
    uvec_state_variables: Dict[str, Any] = field(default_factory=dict)
    uvec_model: Union[ModuleType, Any] = None
    uvec_file: str = ""
    uvec_function_name: str = ""
    nb_carts: int = 1
    offset: float = 0.0
    train_type: TrainType = TrainType.CUSTOM
    irregularities: Optional[Dict[str, Any]] = None
    rail_joint: Optional[Dict[str, Any]] = None
    static_initialisation: bool = False
    initialisation_steps: Optional[int] = None

    def __post_init__(self):
        """
        Initialise the uvec model and its definitions.

        Raises:
        - ValueError: If the specified UVEC model is not supported.
        - ValueError: If the train type is CUSTOM and uvec_parameters are not provided.
        - ValueError: If the train type is not CUSTOM and uvec_parameters are provided.
        - ValueError: If uvec_parameters are not provided after initialization.
        """

        if self.uvec_model is not None:
            if self.uvec_model.__name__ not in (model.value for model in UvecSupportedModels):
                raise ValueError(
                    f"UVEC model {self.uvec_model} is not supported. Please use one of the following models: \
                        {[model.value for model in UvecSupportedModels]}")
            self.uvec_file = os.path.join(self.uvec_model.get_path_file(self.uvec_model.UVEC_NAME), "uvec.py")
            self.uvec_function_name = "uvec"
            self.uvec_model = None  # necessary to allow for a deep copy for a stage in Kratos

            if self.train_type == TrainType.CUSTOM and self.uvec_parameters == {}:
                raise ValueError("For custom train type, uvec_parameters must be provided")

            if self.train_type != TrainType.CUSTOM and self.uvec_parameters != {}:
                raise ValueError("For non-custom train type, uvec_parameters should not be provided")

            self.uvec_parameters = _build_train_parameters(self.train_type, self.nb_carts, self.offset, self.velocity,
                                                           self.static_initialisation, self.initialisation_steps,
                                                           self.uvec_parameters)

        if self.uvec_parameters == {}:
            raise ValueError("uvec_parameters must be provided ")

        self.wheel_configuration: List[float] = self.uvec_parameters["wheel_configuration"]
        self.uvec_parameters["irr_parameters"] = self.irregularities
        self.uvec_parameters["joint_parameters"] = self.rail_joint

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
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations, "UVEC load")

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
            self.value[VERTICAL_AXIS] = GlobalSettings.gravity_value

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
