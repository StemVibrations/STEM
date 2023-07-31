from typing import Sequence, Optional, Dict, List

import numpy as np

from stem.load import LineLoad, MovingLoad, SurfaceLoad
from stem.mesh import Element, Node
from stem.model import Model
from stem.model_part import ModelPart, BodyModelPart

MAPPER_GMSH_TO_KRATOS: Dict[int, Dict[str, Dict[str,str]]] = {
    2: {
        "element": {"TRIANGLE_3N": "UPwSmallStrainElement2D3N"},
        "condition": {"LINE_2N": "UPwFaceLoadCondition2D2N"}
    },
    3: {
        "element": {"TRIANGLE_3N": "UPwSmallStrainElement3D3N"},
        "condition": {}
    }
}


class KratosModelIO:
    """
    Class containing methods to write Kratos model parts and body model parts

    Attributes:
        - ndim (int): The number of dimensions of the problem (2 or 3).
        - domain (str): The name of the Kratos domain.
    """

    def __init__(self,
                 ndim: int,
                 domain: str,
                 ind: int = 2,
                 format_int: str = "{:d}",
                 format_float: str = " {:.10f}"):
        """
        Class to write Kratos problem parts in mpda format.

        Args:
            - ndim (int): The number of dimensions of the problem (2 or 3).
            - domain (str): The name of the Kratos domain.
            - ind (int): indentation between entries in the mdpa file. Defaults to 2.
            - format_int (int): Format for integers (ids) in the mdpa file. Defaults to {:d}.
            - format_float (int): Format for float (coordinates) in the mdpa file.
                Defaults to {:d}.
        """
        self.ndim: int = ndim
        self.domain: str = domain
        self.ind = ind
        self.format_int = format_int
        self.format_float = format_float

    @staticmethod
    def __validate_mesh(model_part: ModelPart):

        if model_part.mesh is None:
            raise ValueError(f"Model part {model_part.name} has not been meshed."
                             f"Before creating the mdpa file, the model part needs to be meshed."
                             f"Please run Model.mesh_")

    @staticmethod
    def __is_body_model_part(model_part:ModelPart):
        """Check if the model part is a body model part

        Args:
            model_part:

        Returns:

        """
        return isinstance(model_part, BodyModelPart)

    @staticmethod
    def __get_element_model_part(model_part: ModelPart):
        """Return the gmsh element of the model part.

        Args:
            - model_part (:class:`stem.model_part.ModelPart`): the model part

        Raises:
            - ValueError: if mesh not initialised first
            - ValueError: if element types are not unique in the model part.

        Returns:
            - str: the gmsh element type
        """

        # infer part name
        if model_part.mesh is None:
            raise ValueError(f"Model part {model_part.name} has not been meshed."
                             f"Before creating the mdpa file, the model part needs to be meshed."
                             f"Please run Model.mesh_")

        # check unique_elements
        element_part_type = np.unique(
            [element.element_type for element in model_part.mesh.elements]
        )

        if len(element_part_type) > 1:
            raise ValueError(f"Model part {model_part.name} has more than 1 element type assigned."
                             f"\n{element_part_type}. Error.")
        return str(element_part_type[0])

    def __write_submodel_block(self, buffer:List[str], block_name:str, block_entities: Optional[Sequence[int]]=None):
        """
        Helping function to write the submodel blocks for the model parts.

        Args:
            - buffer (List[str]): buffer containing the submodelpart info to be updated with the current block.
            - block_name (str): block name, it can be one of
            - block_entities (Optional[Sequence[int]]): ids to be written to the block. If None, an empty block is written.

        Returns:
            - buffer (List[str]): updated buffer with info of the current block.
        """

        # check if mesh is initialised
        sp = " " * self.ind
        buffer.append(f"{sp}Begin SubModelPart{block_name}")
        if block_entities is not None:
            fmt = f"{sp}{self.format_int}"
            buffer += [fmt.format(entity) for entity in block_entities]
        buffer.append(f"{sp}End SubModelPart{block_name}")
        return buffer

    def write_submodelpart_body_model_part(self, body_model_part: BodyModelPart):
        """
        Writes the submodelpart block for a body model part (physical parts with materials).

        Args:
            - body_model_part (:class:`stem.model_part.BodyModelPart`): the body model part to write to mdpa.

        Raises:
            - ValueError: if model part is not a body model part
            - ValueError: if mesh not initialised first

        Returns:
            - block_text (List[str]): list of strings for the submodelpart. Each element is a line in the mdpa file.
        """
        # validate part is body model part
        if not self.__is_body_model_part(body_model_part):
            raise ValueError(f"Model part {body_model_part.name} is not a body model part!")

        # check if mesh is initialised
        if body_model_part.mesh is None:
            raise ValueError(f"Model part {body_model_part.name} has not been meshed."
                             f"Before creating the mdpa file, the model part needs to be meshed."
                             f"Please run Model.generate_mesh()")

        # initialise block
        block_text = ["", f"Begin SubModelPart {body_model_part.name}"]
        block_text = self.__write_submodel_block(
            block_text, block_name="Tables", block_entities=None
        )

        # write nodes
        entities = [node.id for node in body_model_part.mesh.nodes]
        block_text = self.__write_submodel_block(
            block_text, block_name="Nodes", block_entities=entities
        )

        # write elements
        entities = [el.id for el in body_model_part.mesh.elements]
        block_text = self.__write_submodel_block(
                    block_text, block_name="Elements", block_entities=entities
                )
        block_text += ["", f"End SubModelPart", ""]
        return block_text

    def write_submodelpart_process_model_part(self, process_model_part: ModelPart):
        """
        Writes the submodelpart block for a process model part (loads, boundary conditions or
        additional processes such as excavations).

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): the process model part to write to mdpa.

        Raises:
            - ValueError: if model part is not a process model part
            - ValueError: if mesh not initialised first

        Returns:
            - block_text (List[str]): list of strings for the submodelpart. Each element is a line in the mdpa file.
        """

        # validate part is process model part
        if self.__is_body_model_part(process_model_part):
            raise ValueError(f"Model part {process_model_part.name} is not a process model part!")

        # check if mesh is initialised
        if process_model_part.mesh is None:
            raise ValueError(f"Model part {process_model_part.name} has not been meshed."
                             f"Before creating the mdpa file, the model part needs to be meshed."
                             f"Please run Model.generate_mesh()")

        # initialise block
        block_text = ["", f"Begin SubModelPart {process_model_part.name}"]
        block_text = self.__write_submodel_block(
            block_text, block_name="Tables", block_entities=None
        )

        # write nodes
        entities = [node.id for node in process_model_part.mesh.nodes]
        block_text = self.__write_submodel_block(
            block_text, block_name="Nodes", block_entities=entities
        )

        # write conditions
        if process_model_part.mesh.elements is not None:
            # check if part contains elements
            entities = [el.id for el in process_model_part.mesh.elements]
            # model part is process model part and
            block_text = self.__write_submodel_block(
                block_text, block_name="Conditions", block_entities=entities
            )

        block_text += ["", f"Begin SubModelPart", ""]
        return block_text

    def __write_element_line(self, mat_id:int, element:Element):
        """ Write a node to the mdpa format for Kratos

        Args:
            - mat_id (int): integer representing the material id connected to the element.
            - element (:class:`stem.mesh.Element`): element object to write to Kratos.

        Returns:
            - line (str): string representing an element (or condition) in Kratos.
        """
        sp = " " * self.ind
        _node_ids = element.node_ids
        # assemble format for string
        _fmt = f"{sp}{self.format_int}{sp}{self.format_int}{sp}" + " ".join([self.format_int] * len(_node_ids))
        line = _fmt.format(mat_id, element.id, *_node_ids)
        return line

    def write_node_line(self, node:Node):
        """ Write a node to the mdpa format for Kratos

        Args:
            - node (:class:`stem.mesh.Node`): node object to write to Kratos.

        Returns:
            - line: string representing a node in Kratos.
        """
        sp = " " * self.ind
        node_coords = node.coordinates
        # assemble format for string
        _fmt = f"{sp}{self.format_int}{sp}" + " ".join([self.format_float] * len(node_coords))
        line = _fmt.format(node.id, *node_coords)
        return line

    def write_elements_body_model_part(self, body_model_part: BodyModelPart, mat_id:int, kratos_element_type:str):
        """
        Writes the elements of the body model part to the mdpa file

        Args:
            - body_model_part (:class:`stem.model_part.BodyModelPart`): the body model part to write to mdpa.
            - mat_id (int): the material id connected to the element block
            - kratos_element_type (str): the kratos element type

        Raises:
            - ValueError: if model part is not a body model part
            - ValueError: if mesh not initialised first

        Returns:
            - block_text (List[str]): list of strings for the elements of the body model part. \
                Each element is a line in the mdpa file.
        """
        # validate part is body model part
        if not self.__is_body_model_part(body_model_part):
            raise ValueError(f"Model part {body_model_part.name} is not a body model part!")

        # check if mesh is initialised
        if body_model_part.mesh is None:
            raise ValueError(f"Model part {body_model_part.name} has not been meshed."
                             f"Before creating the mdpa file, the model part needs to be meshed."
                             f"Please run Model.generate_mesh()")

        # initialise block
        block_text = ["", f"Begin Elements {kratos_element_type}"]
        block_text.extend(
            [self.__write_element_line(mat_id, el) for el in body_model_part.mesh.elements]
        )
        block_text += ["", f"End Elements", ""]
        return block_text

    @staticmethod
    def __check_if_process_writes_conditions(process_model_part: ModelPart):
        """ Check whether process needs to write condition elements. For example PointLoad,
        Excavation and DisplacementConstraint doesn't need condition elements.

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): the process model part to write to mdpa.

        Returns:
            - bool: whether the process model part writes condition elements
        """
        return isinstance(process_model_part.parameters, (LineLoad, MovingLoad, SurfaceLoad))

    def write_conditions_process_model_part(self, process_model_part: ModelPart, mat_id:int,
                                            kratos_element_type:str):
        """
        Writes the conditions of the process model part to the mdpa file

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): the process model part to write to mdpa.
            - mat_id (int): the material id connected to the conditions block
            - kratos_element_type (str): the kratos element type for the condition.

        Raises:
            - ValueError: if model part is not a process model part
            - ValueError: if mesh not initialised first

        Returns:
            - block_text (List[str]): list of strings for the elements of the body model part. \
                Each element is a line in the mdpa file.
        """
        # validate part is body model part
        if self.__is_body_model_part(process_model_part):
            raise ValueError(f"Model part {process_model_part.name} is not a process model part!")

        # check if mesh is initialised
        if process_model_part.mesh is None:
            raise ValueError(f"Model part {process_model_part.name} has not been meshed."
                             f"Before creating the mdpa file, the model part needs to be meshed."
                             f"Please run Model.generate_mesh()")

        # no elements to write to conditions!
        if process_model_part.mesh.elements is None:
            return []
        # check if conditions has to write condition elements
        if not self.__check_if_process_writes_conditions(process_model_part):
            return []

        block_text = ["", f"Begin Conditions {kratos_element_type}"]
        block_text.extend(
            [self.__write_element_line(mat_id, el) for el in process_model_part.mesh.elements]
        )
        block_text += ["", f"End Conditions", ""]
        return block_text

    def __write_all_nodes(self, model):
        """
        Saves nodes to mdpa format

        Args:
            - filename (str): filename of mdpa file

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        nodes_dict = model.get_all_nodes()
        # sort by key
        nodes_dict = dict(sorted(nodes_dict.items()))
        block_text = ["", "Begin Nodes"]
        block_text.extend(
            [self.write_node_line(node) for node in nodes_dict.values()]
        )
        block_text += ["End Nodes", ""]
        return block_text

    @staticmethod
    def __write_material_ids(model: Model):
        """returns the block initalising the material ids (properties).

        Args:
            - model (:class:`stem.model.Model`]): The model object containing the info on the materials.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []
        for bmp in model.body_model_parts:
            block_text.extend(
                ["", f"Begin Properties {bmp.id}", "End Properties", ""]
            )
        return block_text

    def __write_elements_model(self, model: Model):
        """returns the mdpa block related to elements.

        Args:
            - model (:class:`stem.model.Model`]): The model object containing the info on the elements.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []
        # write elements per body model part
        for bmp in model.body_model_parts:

            if bmp.id is None:
                raise ValueError(f"Body model part {bmp.name} has no id."
                                 "First, material parameters needs to be written to json.")

            # TODO: map to right material!!
            # get the element type here
            element_type = "TEMPORARY_PLACEHOLDER"
            block_text.extend(
                self.write_elements_body_model_part(
                    mat_id=bmp.id, kratos_element_type=element_type, body_model_part=bmp
                )
            )
        return block_text

    def __write_conditions_model(self, model: Model):
        """returns the mdpa block related to conditions.

        Args:
            - model (:class:`stem.model.Model`]): The model object containing the info on the conditions.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []
        # write per conditions per process model part
        for pmp in model.process_model_parts:
            # TODO: map to right material!!
            # get the element type here
            condition_type = "TEMPORARY_PLACEHOLDER"
            # TODO: match the id of the material of the bmp where the condition is applied to.
            # get the element type here
            block_text.extend(
                self.write_conditions_process_model_part(
                    mat_id=999,
                    kratos_element_type=condition_type,
                    process_model_part=pmp,
                )
            )
        return block_text

    def __write_submodel_parts(self, model: Model):
        """returns the mdpa block related to conditions.

        Args:
            - model (:class:`stem.model.Model`]): The model object containing the info on the conditions.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []

        for bmp in model.body_model_parts:
            block_text.extend(
                self.write_submodelpart_body_model_part(bmp)
            )

        for pmp in model.process_model_parts:
            block_text.extend(
                self.write_submodelpart_process_model_part(pmp)
            )

        return block_text

    def write_mdpa_text(self, model: Model):
        """Saves mesh data to mdpa file.

        Args:
            - model (:class:`stem.model.Model`]): The model object containing all the required info on the \
                materials.
            - mesh_file_name (str): The name of the mesh file to store the mdpa file.
            - output_folder (str): folder to store the project parameters file. Defaults to the working directory.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []
        # retrieve the materials in the model and write mdpa text blocks
        block_text.extend(self.__write_material_ids(model))

        # retrieve the unique nodes of all the model parts
        block_text.extend(self.__write_all_nodes(model))

        # write elements per body model part
        block_text.extend(self.__write_elements_model(model))

        # write conditions per process model part
        block_text.extend(self.__write_conditions_model(model))

        # write submodel parts
        block_text.extend(self.__write_submodel_parts(model))

        return block_text