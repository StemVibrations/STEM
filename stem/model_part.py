from typing import Optional, Union, Dict, Any

from stem.additional_processes import AdditionalProcessesParametersABC
from stem.boundary import BoundaryParametersABC
from stem.water_processes import WaterProcessParametersABC
from stem.load import LoadParametersABC, MovingLoad, UvecLoad
from stem.output import OutputParametersABC
from stem.soil_material import SoilMaterial
from stem.structural_material import StructuralMaterial

from stem.geometry import Geometry
from stem.mesh import Mesh
from stem.solver import AnalysisType
from stem.utils import Utils

# define type aliases
ProcessParameters = Union[LoadParametersABC, BoundaryParametersABC, AdditionalProcessesParametersABC,
                          WaterProcessParametersABC, OutputParametersABC]
"""
TypeAlias:
    - ProcessParameters: Union[:class:`stem.load.LoadParametersABC`, \
        :class:`stem.boundary.BoundaryParametersABC`, \
        :class:`stem.additional_processes.AdditionalProcessesParametersABC`, \
        :class:`stem.water_boundaries.WaterBoundaryParametersABC`, \
        :class:`stem.output.OutputParametersABC`]
"""

Material = Union[SoilMaterial, StructuralMaterial]
"""
TypeAlias:
    - Material: Union[:class:`stem.soil_material.SoilMaterial`, :class:`stem.structural_material.StructuralMaterial`]
"""

MovingLoadTypes = (MovingLoad, UvecLoad)
"""
TypeAlias:
    - MovingLoadTypes: Tuple[:class:`stem.load.MovingLoad`, :class:`stem.load.UvecLoad`]
"""


class ModelPart:
    """
    One part of the complete model, this can be a boundary condition, a loading or another special process
    like excavation.

    Attributes:
        - __name (str): name of the model part
        - geometry (Optional[:class:`stem.geometry.Geometry`]): geometry of the model part
        - parameters (Optional[:class:`ProcessParameters`]): process parameters containing the \
            model part parameters.
        - mesh (Optional[:class:`stem.mesh.Mesh`]): mesh of the model part
        - id (Optional[int]): the id of the model part
    """

    def __init__(self, name: str):
        """
        Initialize the model part

        Args:
            - name (str): name of the model part
        """
        self.__name: str = name
        self.geometry: Optional[Geometry] = None
        self.parameters: Optional[ProcessParameters] = None
        self.mesh: Optional[Mesh] = None
        self.id: Optional[int] = None

    def __repr__(self):
        """Repr method to provide a human-readable version of the ModelPart object

        Returns:
            - str: string representing the ModelPart object and it's parameters.

        """
        return f"ModelPart(name={self.name}, parameters={self.parameters.__class__.__name__})"

    @property
    def name(self) -> str:
        """
        Get the name of the model part

        Returns:
            - str: name of the model part

        """
        return self.__name

    def get_geometry_from_geo_data(self, geo_data: Dict[str, Any], name: str):
        """
        Get the geometry from the geo_data and set the nodes and elements attributes.

        Args:
            - geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io

        """

        self.geometry = Geometry.create_geometry_from_gmsh_group(geo_data, name)

    def get_element_name(self, n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Get the element name of the model part. Only loads and boundary conditions currently may have an element name.

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type of the model

        Returns:
            - Optional[str]: element name of the model part

        """

        if isinstance(self.parameters, (LoadParametersABC, BoundaryParametersABC)):
            return self.parameters.get_element_name(n_dim_model, n_nodes_element, analysis_type)
        else:
            return None

    def validate_input(self):
        """
        Validate the input of the model part

        Raises:
            - ValueError: if the geometry of the model part is not defined
            - ValueError: if the parameters of the model part is not defined
            - ValueError: if the origin of the moving load is not aligned with the lines of the geometry
            - ValueError: if the lines of the geometry are not aligned on a path, i.e. \
            there are loops or branching points

        """
        if self.geometry is None:
            raise ValueError(f"Geometry of model part {self.name} is not defined.")

        if self.parameters is None:
            raise ValueError(f"Parameters of model part {self.name} is not defined.")

        if isinstance(self.parameters, MovingLoadTypes):
            # retrieve the coordinates of the points in the path of the load
            coordinates = []
            for line in self.geometry.lines.values():
                line_coords = [self.geometry.points[k].coordinates for k in line.point_ids]
                coordinates.append(line_coords)

            # check origin of moving load is in the path
            if not Utils.is_point_aligned_and_between_any_of_points(coordinates, self.parameters.origin):
                raise ValueError("None of the lines are aligned with the origin of the moving load. Error.")
            # check that the path provided by geometry is correct (no loops, no branching out
            # and no discontinuities in the path)
            if not Utils.check_lines_geometry_are_path(self.geometry):
                raise ValueError("The lines defined for the moving load are not aligned on a path."
                                 "Discontinuities or loops/branching points are found.")


class BodyModelPart(ModelPart):
    """
    This class contains model parts which are part of the body, e.g. a soil layer or track components.

    Inheritance:
        - :class:`ModelPart`

    Attributes:
        - __name (str): name of the model part
        - geometry (Optional[:class:`stem.geometry.Geometry`]): geometry of the model part
        - mesh (Optional[:class:`stem.mesh.Mesh`]): mesh of the model part
        - parameters (Dict[str, Any]): dictionary containing the model part parameters
        - material (:class:`Material`): material of the model part
    """

    def __init__(self, name: str):
        """
        Initialize the body model part

        Args:
            - name (str): name of the body model part
        """
        super().__init__(name)

        self.material: Optional[Material] = None

    def get_element_name(self, n_dim_model: int, n_nodes_element: int, analysis_type: AnalysisType) -> Optional[str]:
        """
        Get the element name of the elements within the model part

        Args:
            - n_dim_model (int): The number of dimensions of the model (2 or 3)
            - n_nodes_element (int): The number of nodes per element
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type of the model

        Returns:
            - Optional[str]: element name of the model part

        """

        if self.material is not None:
            return self.material.get_element_name(n_dim_model, n_nodes_element, analysis_type)
        else:
            return None

    def __repr__(self):
        """Repr method to provide a human-readable version of the BodyModelPart object

        Returns:
            - str: string representing the BodyModelPart object and it's parameters.

        """
        return f"BodyModelPart(name={self.name}, material={self.material.__class__.__name__})"
