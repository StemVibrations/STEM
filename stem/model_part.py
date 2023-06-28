from pathlib import Path
from typing import Optional, Union, Iterable, List, Dict, Any
from dataclasses import dataclass
from abc import ABC

from stem.soil_material import SoilMaterial
from stem.structural_material import StructuralMaterial


@dataclass
class GeometryEntity(ABC):
    """
    An abstract class to represent geometrical elements.

    Attributes:
        id (int or None): A unique identifier for the entity.
    """
    id: int


@dataclass
class Node(GeometryEntity):
    """
    A class to represent a point in space.

    Attributes:
        coordinates (Iterable or None): An iterable of floats representing the x, y and
            z coordinates of the point.
    """
    coordinates: Iterable

    def flat(self) -> List[int]:
        """
        Flattens the attributes into a list
        Returns:
            List[int]
        """
        return [self.id] + list(c for c in self.coordinates)


@dataclass
class Element(GeometryEntity):
    """
    A class to represent an element.

    Attributes:
        property_id (int): the integer identifying the property id
        nodes (List[Node]): list of point element comprising
    """

    property_id: int
    nodes: List[Node]

    def flat(self) -> List[int]:
        """
        Flattens the attributes into a list
        Returns:
            List[int]
        """
        return [self.id, self.property_id] + [c.id for c in self.nodes]


@dataclass
class Condition(GeometryEntity):
    """
    A class to represent a Kratos condition.

    Attributes:
        property_id (int): the integer identifying the property id
        nodes (List[Point]): list of point element comprising
    """
    property_id: int
    nodes: List[Node]

    def flat(self) -> List[int]:
        """
        Flattens the attributes into a list
        Returns:
            List[int]
        """
        return [self.id, self.property_id] + [c.id for c in self.nodes]


@dataclass
class GeometrySet:
    """
    A class to store elements and condition of a named set.

    Attributes:
        set_type (str): type of the geometry set considered. Choose either `Elements` or
            `Conditions`.
        name (str): name of the element type or the condition to be applied to the set
        items (List[Union[Condition, Element]]):
            list of Element or Condition objects in the set
    """

    set_type: str
    name: str
    items: List[Union[Condition, Element]]


@dataclass
class ModelPart:
    """
    One part of the complete model, this can be a boundary condition, a loading or another special process
    like excavation.

    Attributes:
        name (str): name of the model part
        nodes (Iterable[Node]): node id followed by node coordinates in an array
        elements (Iterable[Element]): element id followed by connectivities in an array
        conditions (Iterable[Condition]): condition id followed by connectivities in an
            array
        parameters (dict): dictionary containing the model part parameters

    """

    name: str = None
    nodes: Iterable[Node] = None
    elements: Iterable[Element] = None
    conditions: Iterable[Condition] = None
    parameters: dict = None

    @staticmethod
    def write_submodel_block(
        entity: str, ind: int, content: Iterable[GeometryEntity] = None
    ):
        """
        Helping function to write the submodel blocks for the model parts.

        Args:
            ind (int): indentation level of the mdpa file. Default is 2.
            entity (int): name of the sub-block (Nodes, Elements ...)
            content (Iterable): ids to be written to the sub-block
        Returns:
            _block List[str]: list of strings for the property sub-block.
        """

        _block = [f"{' ' * ind}Begin SubModelPart{entity}"]
        if content is not None:
            for cc in content:
                _block.append(f"{' ' * ind * 2}{cc.id:d}")
        _block.append(f"{' ' * ind}End SubModelPart{entity}")

        return _block

    def write_part(self, ind=2):
        _out = [
            f"Begin SubModelPart {self.name}",
            *ModelPart.write_submodel_block("Tables", ind=ind),
            *ModelPart.write_submodel_block("Nodes", ind=ind, content=self.nodes),
            *ModelPart.write_submodel_block("Elements", ind=ind, content=self.elements),
            *ModelPart.write_submodel_block("Conditions", ind=ind,
                                            content=self.conditions),
            f"End SubModelPart",
        ]
        return _out


@dataclass
class BodyModelPart(ModelPart):
    """
    This class contains model parts which are part of the body, e.g. a soil layer or track components.
    # TODO: later change to Iterable[Node], Iterable[Element] etc. ...

    Attributes:
        name (str): name of the model part
        nodes (Iterable[int]): node id followed by node coordinates in an array
        elements (Iterable[int]): element id followed by connectivities in an array
        conditions (Iterable[int]): condition id followed by connectivities in an array
        parameters (dict): dictionary containing the model part parameters
        material (Union[SoilMaterial, StructuralMaterial]): material of the model part

    """

    material: Optional[Union[SoilMaterial, StructuralMaterial]] = None

    def write_part(self, ind: int = 2) -> List[str]:
        """
        Helping function to write the model part as mdpa format for Kratos.

        Args:
            ind (int): indentation level of the mdpa file. Default is 2.
        Returns:
            _out List[str]: list of strings for the mdpa file.
        """
        _out = [
            f"Begin SubModelPart {self.name}",
            *BodyModelPart.write_submodel_block("Tables", ind=ind),
            *BodyModelPart.write_submodel_block("Nodes", ind=ind, content=self.nodes),
            *BodyModelPart.write_submodel_block(
                "Elements", ind=ind, content=self.elements
            ),
            *BodyModelPart.write_submodel_block(
                "Conditions", ind=ind, content=self.conditions
            ),
            *BodyModelPart.write_submodel_block(
                "Material", ind=ind, content=self.material
            ),
            f"End SubModelPart",
        ]
        return _out


def write_nodes_block(
    ind: int,
    content: List[Node] = None,
    fmt_coord: str = " {:.10f}",
    fmt_id: str = "{:d}",
) -> List[str]:
    """
    Helping function to write the nodes as mdpa format for Kratos.

    Args:
        ind (int): indentation level of the mdpa file. Default is 2.
        content (List[Node]): content to write to the block,
            i.e. a list of the nodes of the model.
        fmt_coord (str): format of the coordinates to be printed. Default is `.10f`.
        fmt_id (str): format of the ids to be printed. Default is `d`.

    Returns:
        _block List[str]: list of strings for the node block.
    """

    ss = " " * ind
    if content is not None:
        _block = ["", f"Begin Nodes"]

        for cc in content:
            _format_coord = " ".join([fmt_coord] * 3)
            _format = f"{ss}{fmt_id}{ss}" + _format_coord
            _block.append(_format.format(*cc.flat()))
        _block += ["End Nodes", ""]
        return _block
    return []


def write_property_block(
    ind: int,
    property_ind: int,
    property_parms: Any,
    fmt_id: str = "{:d}",
):
    """
    Helping function to write the property block as mdpa format for Kratos.
    # TODO implement property_parms different than `None`.

    Args:
        ind (int): indentation level of the mdpa file. Default is 2.
        property_ind (int): index of the property.
        property_parms (Any): properties to be written in the property block
        fmt_id (str): format of the ids to be printed. Default is `d`.

    Returns:
        _block List[str]: list of strings for the property block.
    """
    ss = " " * ind
    _block = ["", f"Begin Properties {property_ind}"]
    if property_parms is not None:
        pass
    _block += ["End Properties", ""]
    return _block


def write_set_block(geo_set: GeometrySet, ind: int, fmt_id: str = "{:d}"):
    """
    Helping function to write the a geometry set (elements or conditions) as mdpa
    format for Kratos.

    Args:
        geo_set (GeometrySet): geometry set object to be
            written
        ind (int): indentation level of the mdpa file. Default is 2.
        fmt_id (str): format of the ids to be printed. Default is `d`.

    Returns:
        _block List[str]: list of strings for the geometry set block.
    """
    ss = " " * ind
    _block = ["", f"Begin {geo_set.set_type} {geo_set.name}"]
    for cc in geo_set.items:
        _n_nodes = len(cc.nodes)
        _format_coord = " ".join([fmt_id] * _n_nodes)
        _format = f"{ss}{fmt_id}{ss}{fmt_id}{ss}" + _format_coord
        _block.append(_format.format(*cc.flat()))
    _block += [f"End {geo_set.set_type}", ""]
    return _block


def write_sets(
    sets_list: List[GeometrySet],
    ind: int,
    fmt_id: str = "{:d}",
):
    """
    Helping function to write all the sets in the model (collected in a list) as mdpa
    format for Kratos.

    Args:
        sets_list (List[GeometrySet]): list of geometry set
            object to be written
        ind (int): indentation level of the mdpa file. Default is 2.
        fmt_id (str): format of the ids to be printed. Default is `d`.

    Returns:
        _block List[str]: list of strings for all the geometry set blocks to be written.
    """

    if sets_list is not None:
        _block = []
        for geo_set in sets_list:
            _block += write_set_block(geo_set=geo_set, ind=ind, fmt_id=fmt_id)
        return _block
    return []


@dataclass
class PartCollection:
    """
    Class to represent all the parts, sets and nodes in the model and write the mdpa
    file for Kratos.

    A class to store elements and condition of a named set.

    Attributes:
        nodes (List[Node`]): list of nodes objects in the model
        element_sets (List[GeometrySet`]): list of element sets
        condition_sets (List[GeometrySet]): list of condition sets
        condition_sets (List[ModelPart]): list of the model
            part in the model
        properties (Dict[int,Any]): dictionary collecting the properties id and
            attributes #TODO still to be implemented with non-None values
    """

    nodes: List[Node] = None
    element_sets: List[GeometrySet] = None
    condition_sets: List[GeometrySet] = None
    model_parts: List[ModelPart] = None
    properties: Dict[int, Any] = None

    def write_mdpa(
        self,
        ind: int,
        fmt_coord: str = "{:.10f}",
        fmt_id: str = "{:d}",
        output_name: str = None,
        output_dir: str = "",
    ):
        """
        Writes the parts, sets and nodes in the model as mdpa
        format for Kratos and returns a list containg the strings making the mdpa file.

        Args:
            ind (int): indentation level of the mdpa file. Default is 2.
            fmt_coord (str): format of the coordinates to be printed. Default is `.10f`.
            fmt_id (str): format of the ids to be printed. Default is `d`.
            output_name (str): name of the mdpa file. if suffix is not provided or is
                not mdpa, mdpa is added instead. If `None`, no file is created.
            output_dir (str): relative of absolute path to the directory where
                the mdpa file is to be stored.

        Returns:
            _out List[str]: list containing the string for the mdpa files
        """
        _out = []
        if self.properties is not None:
            for property_ind, property_parms in self.properties.items():
                _out += write_property_block(
                    ind=ind,
                    property_ind=property_ind,
                    property_parms=property_parms,
                    fmt_id=fmt_id,
                )
        _out += write_nodes_block(ind=ind, content=self.nodes, fmt_coord=fmt_coord)
        _out += write_sets(sets_list=self.element_sets, ind=ind, fmt_id=fmt_id)
        _out += write_sets(sets_list=self.condition_sets, ind=ind, fmt_id=fmt_id)

        if self.model_parts is not None:
            for model_part in self.model_parts:
                _out.append("")
                _out += model_part.write_part(ind=ind)
                _out.append("")

        if output_name is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir.joinpath(output_name)
            output_path = output_path.with_suffix(".mdpa")
            _out_txt = [_ + "\n" for _ in _out]

            with open(output_path, "w") as _buf:
                _buf.writelines(_out_txt)

        return _out
