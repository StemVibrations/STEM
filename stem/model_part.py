from dataclasses import dataclass
from typing import Union, Optional, List, Type, Any, get_args
import numpy as np

from stem.boundary import *
from stem.geometry import Point, Line, Surface, Volume
from stem.load import *
from stem.mesh import Node, Element, PhysicalGroup, write_sub_model_block
from stem.soil_material import *
from stem.structural_material import *


@dataclass
class Part:
    """
    One part of the complete model, this can be a boundary condition, a loading, another special process like excavation
     (model part) or parts that part of the body (body model part).

    Attributes:
        - name (str): name of the model part
        - parameters (Union[:class:`stem.boundary.BoundaryParametersABC`, :class:`stem.load.LoadParametersABC`, \
            :class:`stem.soil_material.StructuralMaterial`, :class:`stem.structural_material.SoilMaterial`]):
            parameter object containing the parameters as well as the information on the type of model part.
        - nodes (List[:class:`stem.mesh.Node`]): nodes in the model part
        - elements (Optional[List[:class:`stem.mesh.Element`]]): elements in the model part
        - points (Optional[List[:class:`stem.geometry.Point`]]): list of points in the model part required to create
            the mesh.
        - lines (Optional[List[:class:`stem.geometry.Line`]]): list of lines in the model part required to create
            the mesh.
        - surfaces (Optional[List[:class:`stem.geometry.Surface`]]): list of surfaces in the model part required to
            create the mesh
        - volumes (Optional[List[:class:`stem.geometry.Volume`]]): list of volumes in the model part required to
            create the mesh.
        - id (Optional[int]): Unique id of the body model part required for material definition.
    """

    name: str
    parameters: Any
    nodes: List[Node]
    elements: Optional[List[Element]] = None

    points: Optional[List[Point]] = None
    lines: Optional[List[Line]] = None
    surfaces: Optional[List[Surface]] = None
    volumes: Optional[List[Volume]] = None
    id: Optional[int] = None

    @property
    def is_condition(self):
        """
        Property describing whether the part constitutes elements or condition elements (e.g. loads).

        Returns:
            bool
        """
        if isinstance(self.parameters, get_args(Union[LineLoad, SurfaceLoad, AbsorbingBoundary])):
            return True
        return False

    @classmethod
    def from_physical_group_and_parameters(
        cls,
        physical_group: PhysicalGroup,
        parameters: Union[
            BoundaryParametersABC, LoadParametersABC, StructuralMaterial, SoilMaterial
        ],
    ):
        """
        Create a model part or body model part from a PhyicalGroup and Parameters defining the part (load,
        boundary condition, material properties etc.).

        Args:
            - physical_group (:class:`stem.mesh.PhysicalGroup`): physical group object containing info on the part name,
                nodes and elements in the model part or body model part.
            - parameters (Union[:class:`stem.boundary.BoundaryParametersABC`, :class:`stem.load.LoadParametersABC`, \
                :class:`stem.soil_material.StructuralMaterial`, :class:`stem.structural_material.SoilMaterial`]):
                parameter object containing the parameters as well as the information on the type of model part.

        Returns:

        """
        return cls(
            name=physical_group.name,
            nodes=physical_group.nodes,
            elements=physical_group.elements,
            parameters=parameters,
        )

    def map_condition_to_condition_element(self, ndim: int = 2):
        if ndim not in [2, 3]:
            raise ValueError(
                f"Dimension of the model are not 2 or 3, but {ndim} was given."
                f"Please enter a valid dimension."
            )

        if isinstance(self.parameters, LineLoad):
            return f"UPwFaceLoadCondition{ndim}D2N"
        else:
            raise NotImplementedError

    def write_elements(self, ndim: int, ind: int, fmt_id: str = "{:d}") -> List[str]:
        """
        Writes the element blocks within a model part or a body model part.
        From Body model parts, elements are written and for model parts conditions
        are written.

        Args:
            - ndim (int): number of dimension of the model (2 or 3).
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_id (str): format of the ids to be printed. Default is `d`.

        Returns:

        """
        if self.elements is None or len(self.elements) == 0:
            return []

        if self.is_condition:
            element_class = "Conditions"
            element_type = self.map_condition_to_condition_element(ndim=ndim)

            ss = " " * ind
            block = ["", f"Begin {element_class} {element_type}"]

            for _el in self.elements:
                _el.property_id = self.id
                n_nodes = len(self.nodes)
                format_coord = " ".join([fmt_id] * n_nodes)
                _format = f"{ss}{fmt_id}{ss}{fmt_id}{ss}" + format_coord
                block.append(_format.format(*_el.flat()))
            block += [f"End {element_class}", ""]

        else:
            element_class = "Elements"
            element_type = np.unique([_el.element_type for _el in self.elements])
            if len(element_type) > 1:
                raise ValueError(
                    f"More than 1 element type specified" f" for part {self.name}."
                )
            else:
                element_type = element_type[0]

            ss = " " * ind
            block = ["", f"Begin {element_class} {element_type}"]
            for _el in self.elements:
                _el.property_id = self.id
                n_nodes = len(_el.node_ids)
                format_coord = " ".join([fmt_id] * n_nodes)
                _format = f"{ss}{fmt_id}{ss}{fmt_id}{ss}" + format_coord
                block.append(_format.format(*_el.flat()))
            block += [f"End {element_class}", ""]

        return block

    def write_sub_model_blocks(self, ind: int, fmt_id: str = "{:d}") -> List[str]:
        """
        Writes the sub-model blocks of the given model part or body model part.

        Args:
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_id (str): format of the ids to be printed. Default is `d`.

        Returns:
            - _out List[str]: list containing the string for the mdpa files
        """

        _out = ["", f"Begin SubModelPart {self.name}"]
        _out += write_sub_model_block("Tables", ind=ind, fmt_id=fmt_id)
        _out += write_sub_model_block(
            "Nodes", ind=ind, fmt_id=fmt_id, content=self.nodes
        )
        if self.is_condition:
            _out += write_sub_model_block(
                "Conditions", ind=ind, fmt_id=fmt_id, content=self.elements
            )
        else:
            _out += write_sub_model_block(
                "Elements", ind=ind, fmt_id=fmt_id, content=self.elements
            )
        _out += [f"End SubModelPart", ""]
        return _out


@dataclass
class ModelPart(Part):
    """
    One part of the complete model, this can be a boundary condition, a loading or
    another special process like excavation.

    Inheritance:
        - :class:`Part`

    Attributes:
        - name (str): name of the model part
        - parameters (Union[:class:`stem.load.LoadParametersABC`, \
            :class:`stem.boundary.BoundaryParametersABC`]): parameter object containing the parameters as well as the
            information on the type of model part.
        - nodes (List[:class:`stem.mesh.Node`]): nodes in the model part
        - elements (Optional[List[:class:`stem.mesh.Element`]]): elements in the model part
        - points (Optional[List[:class:`stem.geometry.Point`]]): list of points in the model part required to create
            the mesh.
        - lines (Optional[List[:class:`stem.geometry.Line`]]): list of lines in the model part required to create
            the mesh.
        - surfaces (Optional[List[:class:`stem.geometry.Surface`]]): list of surfaces in the model part required to
            create the mesh
        - volumes (Optional[List[:class:`stem.geometry.Volume`]]): list of volumes in the model part required to
            create the mesh.
    """

    parameters: Union[LoadParametersABC, BoundaryParametersABC]


@dataclass
class BodyModelPart(Part):
    """
    This class contains model parts which are part of the body, e.g. a soil layer or track components.

    Inheritance:
        - :class:`Part`

    Attributes:
        - name (str): name of the model part
        - parameters (Optional[Union[:class:`stem.soil_material.SoilMaterial`, \
            :class:`stem.structural_material.StructuralMaterial`]]): material parameters for the body model part
        - nodes (List[:class:`stem.mesh.Node`]): nodes in the model part
        - elements (Optional[List[:class:`stem.mesh.Element`]]): elements in the model part
        - points (Optional[List[:class:`stem.geometry.Point`]]): list of points in the model part required to create
            the mesh.
        - lines (Optional[List[:class:`stem.geometry.Line`]]): list of lines in the model part required to create
            the mesh.
        - surfaces (Optional[List[:class:`stem.geometry.Surface`]]): list of surfaces in the model part required to
            create the mesh
        - volumes (Optional[List[:class:`stem.geometry.Volume`]]): list of volumes in the model part required to
            create the mesh.
    """

    parameters: Union[SoilMaterial, StructuralMaterial]
