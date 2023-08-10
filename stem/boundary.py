from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from stem.solver import AnalysisType


@dataclass
class BoundaryParametersABC(ABC):
    """
    Abstract base class for boundary parameters
    """

    pass

    @property
    @abstractmethod
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint.
        
        Returns:
            - bool
        """
        raise Exception("abstract method 'is_constraint' of boundary parameters class is called")

    @staticmethod
    @abstractmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType):
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
        - value (List[float]): Displacement constraint for direction [m].
    """

    active: List[bool]
    is_fixed: List[bool]
    value: List[float]

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. True for DisplacementConstraint.
        
        Returns:
            - bool
        """
        return True

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType):
        """
        Static method to get the element name for a displacement constraint. Displacement constraint does not have a
        name.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (stem.solver.AnalysisType): The analysis type.

        Raises:
            - ValueError: Displacement constraint can only be applied in mechanical or mechanical groundwater flow

        Returns:
            - None: Displacement constraint does not have a name

        """

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
        - value (List[float]): Rotation constraint around x, y and axis.
    """

    active: List[bool]
    is_fixed: List[bool]
    value: List[float]

    @property
    def is_constraint(self) -> bool:
        """
        Property which indicates if boundary condition is a constraint. True for RotationConstraint.
        
        Returns:
            - bool
        """
        return True

    @staticmethod
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType):
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
    def get_element_name(n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType):
        """
        Static method to get the element name for an absorbing boundary.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type.

        Raises:
            - ValueError: Absorbing boundary conditions are only implemented for 2D and 3D geometries.
            - ValueError: Absorbing boundary conditions are not implemented for quadratic elements in a 3D geometry.
            - ValueError: Absorbing boundary conditions can only be applied in mechanical or mechanical groundwater \
                flow analysis

        Returns:
            - str: The element name

        """

        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW or analysis_type == AnalysisType.MECHANICAL:

            if n_dim_model != 2 or n_dim_model != 3:
                raise ValueError(f"Absorbing boundary conditions are only implemented for 2D and 3D geometries.")

            if n_dim_model == 3 and n_nodes_element > 4:
                raise ValueError(f"Absorbing boundary conditions are not implemented for quadratic elements in a "
                                 f"3D geometry.")
            else:
                element_name = f"UPwLysmerAbsorbingCondition{n_dim_model}D{n_nodes_element}N"

        else:
            raise ValueError("Absorbing boundary conditions can only be applied in mechanical or mechanical "
                            "groundwater flow analysis")

        return element_name
