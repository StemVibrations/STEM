from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any, Optional, Iterable

import numpy as np
import numpy.typing as npt


@dataclass
class MeshEntity(ABC):
    """
    An abstract class to represent geometrical elements.

    Attributes:
        - id (int): A unique identifier for the entity.
    """

    id: int


@dataclass
class Node(MeshEntity):
    """
    A class to represent a point in space.

    Attributes:
        - id (int): Unique identifier for the node.
        - coordinates (Iterable[Union[int, float]]): An iterable of floats representing
            the x, y and z coordinates of the point.
    """

    coordinates: Iterable[Union[int, float]]

    def flat(self) -> List[Union[int, float]]:
        """
        Flattens the attributes into a list
        Returns:
            - List[Union[int, float]]
        """
        return [self.id] + [c for c in self.coordinates]


@dataclass
class Element(MeshEntity):
    """
    A class to represent an element.

    Inheritance:
        - :class:`MeshEntity`

    Attributes:
        - id (int): Unique identifier for the element.
        - element_type (str): element type
        - node_ids (Union[List[int], npt.NDArray[np.int64]]): node ids
        - property_id (Optional[int]): the integer identifying the property id. It is assigned when writing the mdpa
            file to link the materials.
    """

    element_type: str
    node_ids: Union[List[int], npt.NDArray[np.int64]]
    property_id: Optional[int] = None

    def flat(self) -> List[Any]:
        """
        Flattens the attributes into a list
        Returns:
            - List[int]
        """
        return [self.id, self.property_id] + [c for c in self.node_ids]


@dataclass
class Condition(MeshEntity):
    """
    A class to represent a Kratos condition.

    Inheritance:
        - :class:`MeshEntity`

    Attributes:
        - id (int): Unique identifier for the condition element.
        - element_type (str): element type
        - node_ids (Union[List[int], npt.NDArray[np.int64]]): node ids
        - property_id (Optional[int]): the integer identifying the property id. It is assigned when writing the mdpa
            file to link the materials.
    """

    element_type: str
    node_ids: Union[List[int], npt.NDArray[np.int64]]
    property_id: Optional[int] = None

    def flat(self) -> List[Any]:
        """
        Flattens the attributes into a list
        Returns:
            - List[int]
        """
        return [self.id, self.property_id] + [c for c in self.node_ids]


@dataclass
class PhysicalGroup:
    """
    A class to represent a gmesh physical group to link with Kratos part names and body part names.

    Attributes:
        - name (int): Unique name identifying the physical group.
        - node_ids (Optional[List[:class:`stem.mesh.Node`]]): node ids
        - element_ids (Optional[List[:class:`stem.mesh.Element`]]): element or connectivity ids
    """

    name: str
    nodes: List[Node]
    elements: Optional[List[Element]]


def write_sub_model_block(
    entity: str, content: Optional[Iterable[MeshEntity]]=None, ind: int = 2, fmt_id: str = "{:d}"
):
    """
    Helping function to write the submodel blocks for the model parts.

    Args:
        - entity (int): name of the sub-block (Nodes, Elements ...)
        - content (Optional[Iterable[:class:`MeshEntity`]]): ids to be
            written to the sub-block.
        - ind (int): indentation level of the mdpa file. Default is 2.
    Returns:
        - _block List[str]: list of strings for the property sub-block.
    """
    ss = " " * ind
    _block = [f"{ss}Begin SubModelPart{entity}"]
    if content is not None:
        for cc in content:
            _format = f"{ss}{fmt_id}"
            _block.append(_format.format(cc.id))
    _block.append(f"{' ' * ind}End SubModelPart{entity}")

    return _block


@dataclass
class Mesh:
    """
    Class to represent all the parts, sets and nodes in the model and write the mdpa
    file for Kratos.

    A class to store elements and condition of a named set.

    Attributes:
        - ndim (int): number of dimensions of the mesh
        - nodes (List[:class:`Node`]): list of nodes objects in the model
        - elements (Optional[List[:class:`MeshSet`]]): list of element sets
        - conditions (Optional[List[:class:`MeshSet`]]): list of condition sets
    """

    ndim: int
    nodes: Optional[Dict[int, Node]]
    elements: Optional[Dict[int, Element]]
    physical_groups: Optional[Dict[str, PhysicalGroup]]
    conditions: Optional[Dict[int, Condition]] = None

    @classmethod
    def read_mesh_from_gmsh(cls, mesh_file_name: str) -> None:
        # Todo implement this method to read mesh from gmsh file and create a mesh object with the data read from the
        #  file.
        pass

    @classmethod
    def read_mesh_from_dictionary(cls, mesh_dict: Dict[str, Any]):
        """
        Convert a dictionary containing the nodes, elements and physical groups into a mesh object.

        Args:
            mesh_dict(dict): the dictionary containing `nodes`, `elements` and `physical_groups` keys

        Returns:
            (:class:`Mesh`)
        """

        _nodes = {}
        _elements = {}
        _ph_groups = {}

        for key_entity, entities in mesh_dict.items():
            if key_entity == "nodes":
                for nk, coords in entities.items():
                    _nodes[nk] = Node(nk, coords)
            if key_entity == "elements":
                for ek, vls in entities.items():
                    _elements[ek] = Element(ek, vls["type"], vls["connectivity"])

        for key_entity, entities in mesh_dict.items():
            if key_entity == "physical_groups":
                for pg_name, vls in entities.items():
                    _nn, _ee = ([], [])
                    if "node_ids" in vls.keys():
                        _nn = [_nodes[_n] for _n in vls["node_ids"]]
                    if "element_ids" in vls.keys():
                        _ee = [_elements[_e] for _e in vls["element_ids"]]

                    _ph_groups[pg_name] = PhysicalGroup(
                        pg_name, _nn, _ee
                    )
        return cls(ndim=3, nodes=_nodes, elements=_elements, physical_groups=_ph_groups)

    @staticmethod
    def prepare_data_for_kratos(
        mesh_data: Dict[str, Any]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Prepares mesh data for Kratos

        Args:
            - mesh_data (Dict[str, Any]): dictionary of mesh data


        Returns:
            - nodes (npt.NDArray[np.float64]): node id followed by node coordinates in an array
            - elements (npt.NDArray[np.int64]): element id followed by connectivities in an array
        """

        # create array of nodes where each row is represented by [id, x,y,z]
        nodes = np.concatenate(
            (mesh_data["nodes"]["ids"][:, None], mesh_data["nodes"]["coordinates"]),
            axis=1,
        )

        all_elements_list = []
        # create array of elements where each row is represented by [id, node connectivities]
        for v in mesh_data["elements"].values():
            all_elements_list.append(
                np.concatenate((v["element_ids"][:, None], v["element_nodes"]), axis=1)
            )

        all_elements = np.array(all_elements_list).astype(int)

        return nodes, all_elements

    def write_nodes_for_mdpa(
        self,
        ind: int,
        fmt_coord: str = " {:.10f}",
        fmt_id: str = "{:d}",
    ) -> List[Union[str, Any]]:
        """
        Helping function to write the nodes as mdpa format for Kratos.

        Args:
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_coord (str): format of the coordinates to be printed. Default is `.10f`.
            - fmt_id (str): format of the ids to be printed. Default is `d`.

        Returns:
            - _block List[str]: list of strings for the node block.
        """

        ss = " " * ind
        if self.nodes is not None:
            _block = ["", f"Begin Nodes"]

            for cc in self.nodes.values():
                _format_coord = " ".join([fmt_coord] * 3)
                _format = f"{ss}{fmt_id}{ss}" + _format_coord
                _block.append(_format.format(*cc.flat()))
            _block += ["End Nodes", ""]
            return _block
        return []
