from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from stem.solver import AnalysisType
from stem.utils import Utils
from stem.table import Table


@dataclass
class BoundaryParametersABC(ABC):
    """
    Abstract base class for boundary parameters
    """

    @property
    @abstractmethod
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint.

        Raises:
            - Exception: abstract method is called

        """
        raise Exception("abstract method 'is_constraint' of boundary parameters class is called")

    @staticmethod
    @abstractmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Abstract static method to get the element name for a boundary condition.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - Exception: abstract method is called
        """
        raise Exception("abstract method 'get_element_name' of boundary parameters class is called")


@dataclass
class DisplacementConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for displacement constraint

    Inheritance:
        - :class:`BoundaryParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate constraint for each direction.
        - is_fixed (List[bool]): Specify if constraint is fixed for each direction.
        - value (List[Union[float, :class:`stem.table.Table`]]): Displacement value for direction [m]. \
            It should be a list of either float or table for each displacement. If a float is specified, the \
            displacement is time-independent, otherwise the table specifies the amplitude of the amplitude of the \
            displacement [m] over time [s] for each direction.
    """

    active: List[bool]
    is_fixed: List[bool]
    value: List[Union[float, Table]]

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. True for DisplacementConstraint.
        
        Returns:
            - bool
        """
        return True

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a displacement constraint. Displacement constraint does not have a
        name.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: Displacement constraint can only be applied in mechanical or mechanical groundwater flow

        Returns:
            - None: Displacement constraint does not have a name

        """

        available_node_dim_combinations = {
            2: [None],
            3: [None],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, None, available_node_dim_combinations,
                                             "Displacement constraint")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            # displacement constraint does not have an element name
            element_name = None
        else:
            raise ValueError("Displacement constraint can only be applied in mechanical or mechanical groundwater "
                             "flow analysis")

        return element_name


@dataclass
class RotationConstraint(BoundaryParametersABC):
    """
    Class containing the boundary parameters for rotation constraint

    Inheritance:
        - :class:`BoundaryParametersABC`

    Attributes:
        - active (List[bool]): Activate/deactivate constraint for each direction.
        - is_fixed (List[bool]): Specify if constraint is fixed around each axis.
        - value (List[float]): Rotation constraint
        - value (List[Union[float, :class:`stem.table.Table`]]): Rotation value around x, y and z axis [Rad]. \
            It should be a list of either float or table for each direction. If a float is specified, the rotation is \
            time-independent, otherwise the table specifies the amplitude of the rotation [Rad] over \
            time [s] around each axis.
    """

    active: List[bool]
    is_fixed: List[bool]
    value: List[Union[float, Table]]

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. True for RotationConstraint.
        
        Returns:
            - bool
        """
        return True

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for a rotation constraint. Rotation constraint does not have a
        name.

        Args:
            - n_dim_model (int): The number of dimensions of the model
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: Rotation constraint can only be applied in mechanical or mechanical groundwater flow

        Returns:
            - None: Rotation constraint does not have a name

        """
        available_node_dim_combinations = {
            2: [None],
            3: [None],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, None, available_node_dim_combinations,
                                             "Rotation constraint")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            # rotation constraint does not have an element name
            element_name = None
        else:
            raise ValueError("Rotation constraint can only be applied in mechanical or mechanical groundwater "
                             "flow analysis")

        return element_name


@dataclass
class AbsorbingBoundary(BoundaryParametersABC):
    """
    Class containing the boundary parameters for a point boundary

    Inheritance:
        - :class:`BoundaryParametersABC`

    Attributes:
        - absorbing_factors (List[float]): Indicated how much of the P-wave \
            and S-wave should be damped from the boundaries and is comprised between \
            0 (no damping) and 1 (full damping).
        - virtual_thickness (float): Entity of the virtual thickness [m].
    """

    absorbing_factors: List[float]
    virtual_thickness: float

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. False for AbsorbingBoundary.
        
        Returns:
            - bool
        """
        return False

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Static method to get the element name for an absorbing boundary.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: Absorbing boundary conditions can only be applied in mechanical or mechanical groundwater \
                flow analysis

        Returns:
            - Optional[str]: The element name

        """

        available_node_dim_combinations = {
            2: [2, 3],
            3: [3, 4],
        }
        Utils.check_ndim_nnodes_combinations(n_dim_model, n_nodes_element, available_node_dim_combinations,
                                             "Absorbing boundary")

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:
            element_name = f"UPwLysmerAbsorbingCondition{n_dim_model}D{n_nodes_element}N"
        else:
            raise ValueError("Absorbing boundary conditions can only be applied in mechanical or mechanical "
                            "groundwater flow analysis")

        return element_name
