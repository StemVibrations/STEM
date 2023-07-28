from typing import Sequence, Optional

import numpy as np

from stem.model_part import ModelPart, BodyModelPart

MAPPER_GMSH_TO_KRATOS_2D = {
    "TRIANGLE_3N": "UPwSmallStrainElement2D3N"
}

MAPPER_GMSH_TO_KRATOS_3D = {
    "TRIANGLE_3N": "UPwSmallStrainElement2D3N"
}


def map_gmsh_element_to_kratos(gmsh_element: str, ndim: int):
    if ndim == 2:
        return MAPPER_GMSH_TO_KRATOS_2D[gmsh_element]
    elif ndim == 3:
        return MAPPER_GMSH_TO_KRATOS_3D[gmsh_element]


class KratosModelPartIO:
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

    def __get_kratos_element_type(self, model_part: ModelPart):
        # check unique_elements
        element_part_type = np.unique(
            [element.element_type for element in model_part.mesh.elements]
        )

        if len(element_part_type) > 1:
            raise ValueError(f"Model part {model_part.name} has more than 1 element type assigned."
                             f"\n{element_part_type}. Error.")
        element_part_type = element_part_type[0]

        return map_gmsh_element_to_kratos(element_part_type, self.ndim)

    def __write_submodel_block(self, buffer:list[str], block_name:str, block_entities: Optional[Sequence[int]]=None):
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
            - body_model_parts (:class:`stem.model_part.BodyModelPart`): the body model part to write to mdpa.

        Returns:
            - block_text
        """
        # validate part is body model part
        if not self.__is_body_model_part(body_model_part):
            raise ValueError(f"Model part {body_model_part.name} is not a body model part!")

        # check if mesh is initialised
        self.__validate_mesh(body_model_part)
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
            - process_model_parts (:class:`stem.model_part.ModelPart`): the process model part to write to mdpa.

        Returns:
            - block_text
        """

        # validate part is process model part
        if self.__is_body_model_part(process_model_part):
            raise ValueError(f"Model part {process_model_part.name} is not a process model part!")

        # check if part is process model part
        self.__validate_mesh(process_model_part)

        # check if mesh is initialised
        self.__validate_mesh(process_model_part)
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

    def __create_element_block(self, model_part: ModelPart):
        """
        Creates a dictionary containing the problem data

        Args:
            - problem_data (:class:`stem.solver.Problem`): The problem data

        Returns:
            - Dict[str, Any]: dictionary containing the problem data
        """

        # check if mesh is initialised
        self.__validate_mesh(model_part)


        return None

