import os
from functools import reduce
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence

import numpy as np

from stem.IO.kratos_boundaries_io import KratosBoundariesIO
from stem.IO.kratos_water_processes_io import KratosWaterProcessesIO
from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.IO.kratos_output_io import KratosOutputsIO
from stem.IO.kratos_solver_io import KratosSolverIO
from stem.IO.kratos_additional_processes_io import KratosAdditionalProcessesIO
from stem.structural_material import *
from stem.boundary import BoundaryParametersABC, AbsorbingBoundary, DisplacementConstraint, RotationConstraint
from stem.load import LoadParametersABC, LineLoad, MovingLoad, SurfaceLoad, PointLoad, UvecLoad
from stem.additional_processes import ParameterFieldParameters, AdditionalProcessesParametersABC
from stem.water_processes import WaterProcessParametersABC
from stem.mesh import Element, Node
from stem.model import Model
from stem.model_part import ModelPart, BodyModelPart
from stem.table import Table
from stem.output import Output, OutputParametersABC, JsonOutputParameters
from stem.utils import Utils
from stem.IO.io_utils import IOUtils

# define domain name
DOMAIN = "PorousDomain"

# indentation between entries in the mdpa file. Defaults to 2.
INDENTATION = 2
# format for integers
FORMAT_INTEGER: str = "{:d}"
# format for floats (long)
FORMAT_FLOAT_LONG: str = " {:.10f}"
# format for floats (short)
FORMAT_FLOAT_SHORT: str = " {:.4f}"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        - ndim (int): The number of dimensions of the problem.
        - project_folder (str): folder to store the project files (mesh, project and material parameters as well as \
            json files). Defaults to the working directory.
        - material_io (:class:`stem.IO.kratos_material_io.KratosMaterialIO`): The material IO object.
        - loads_io (:class:`stem.IO.kratos_loads_io.KratosLoadsIO`): The loads IO object.
        - boundaries_io (:class:`stem.IO.kratos_boundaries_io.KratosBoundariesIO`): The boundaries IO object.
        - outputs_io (:class:`stem.IO.kratos_output_io.KratosOutputsIO`): The outputs IO object.
        - solver_io (:class:`stem.IO.kratos_solver_io.KratosSolverIO`): The solver IO object.
        - additional_process_io (:class:`stem.IO.kratos_additional_process_io.KratosAdditionalProcessesIO`): \
            The IO object for the additional processes.

    """

    def __init__(self, ndim: int):
        """
        Constructor of KratosIO class

        Args:
            - ndim (int): The number of dimensions of the problem.
        """

        self.ndim = ndim
        self.project_folder = "./"
        self.material_io = KratosMaterialIO(self.ndim, DOMAIN)
        self.loads_io = KratosLoadsIO(DOMAIN)
        self.boundaries_io = KratosBoundariesIO(DOMAIN)
        self.water_boundaries_io = KratosWaterProcessesIO(DOMAIN)
        self.outputs_io = KratosOutputsIO(DOMAIN)
        self.solver_io = KratosSolverIO(self.ndim, DOMAIN)
        self.additional_process_io = KratosAdditionalProcessesIO(DOMAIN)

    @staticmethod
    def __is_body_model_part(model_part: ModelPart):
        """
        Check whether the model part is a body model part.

        Args:
            - model_part (:class:`stem.model_part.ModelPart`): the model part

        Returns:
            - bool: whether the model part is body
        """
        return isinstance(model_part, BodyModelPart)

    @staticmethod
    def __check_if_process_writes_conditions(process_model_part: ModelPart) -> bool:
        """
        Check whether process needs to write condition elements. For example PointLoad,
        Excavation and DisplacementConstraint do not need condition elements.

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): the process model part to write to mdpa.

        Returns:
            - bool: whether the process model part writes condition elements
        """
        return isinstance(process_model_part.parameters, (PointLoad, LineLoad, MovingLoad, UvecLoad,
                                                           SurfaceLoad, AbsorbingBoundary))

    def __initialise_process_model_part_ids(self, model: Model):
        """
        Resets the process model part ids. Also resets the condition element ids.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the process model parts.

        """

        # reset all condition model part ids
        new_id = 0
        new_cond_id = 1
        for pmp in model.process_model_parts:
            # if the process writes condition add an id
            if self.__check_if_process_writes_conditions(pmp):
                new_id += 1
                pmp.id = new_id

                # Check if mesh in current process model part is initialised with elements
                if pmp.mesh is not None and pmp.mesh.elements is not None:
                    # renew all condition element ids
                    new_cond_dict: Dict[int, Element] = {}
                    for old_id, cond in pmp.mesh.elements.items():
                        cond.id = new_cond_id
                        new_cond_dict[new_cond_id] = cond
                        new_cond_id += 1
                    pmp.mesh.elements = new_cond_dict

    @staticmethod
    def __initialise_body_model_part_ids(model: Model):
        """
        Resets the body model part ids.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the body model parts.

        """
        # reset all body model part ids
        for ix, bmp in enumerate(model.body_model_parts):
            bmp.id = ix + 1

    @staticmethod
    def __get_unique_tables_process_model_part(process_model_part: ModelPart) -> List[Table]:
        """
        Retrieve all the memory-unique tables in the model part.

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): the process model part containing \
                the tables.

        Returns:
            - tables (List[:class:`stem.table.Table`]): list of the tables in the process model part.
        """

        tables: List[Table] = []
        if isinstance(process_model_part.parameters, (PointLoad, LineLoad, SurfaceLoad, DisplacementConstraint,
                                                      RotationConstraint)):
            for vv in process_model_part.parameters.value:
                if isinstance(vv, Table):
                    tables.append(vv)

        return Utils.get_unique_objects(tables)

    def __get_unique_tables(self, model: Model) -> List[Table]:
        """
        Retrieve all the memory-unique tables in the model.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info on the loads.

        Returns:
            - tables (List[:class:`stem.table.Table`]): list of the unique table objects in the models
        """

        tables: List[Table] = []
        for pmp in model.process_model_parts:
            tables.extend(self.__get_unique_tables_process_model_part(pmp))

        return Utils.get_unique_objects(tables)

    def __initialise_table_ids(self, model: Model):
        """
        Initialise or reset the id of the tables contained in the load parameters objects.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the body model parts.

        """

        unique_tables = self.__get_unique_tables(model)

        for ix, table in enumerate(unique_tables):
            table.id = ix + 1

    def initialise_model_ids(self, model: Model):
        """
        Initialise the ids of the model parts and tables.

        Args:
            model (:class:`stem.model.Model`): the model object containing the model parts and tables.

        """

        self.__initialise_table_ids(model)
        self.__initialise_process_model_part_ids(model)
        self.__initialise_body_model_part_ids(model)

    @staticmethod
    def __write_sub_model_part_block(buffer: List[str], block_name: str,
                                     block_entities: Optional[List[Optional[int]]] = None) -> List[str]:
        """
        Helping function to write the sub-model part blocks for the model parts.

        Args:
            - buffer (List[str]): buffer containing the sub-model part info to be updated with the current block.
            - block_name (str): block name, it can be one of Tables, Nodes, Elements or Conditions.
            - block_entities (Optional[List[Optional[int]]]): ids to be written to the block. If None, an empty block is
            written.

        Returns:
            - buffer (List[str]): updated buffer with info of the current block.
        """

        # define indentation
        space = " " * INDENTATION

        # append header
        buffer.append(f"{space}Begin SubModelPart{block_name}")
        # write block entities
        if block_entities is not None:
            fmt = f"{space}{FORMAT_INTEGER}"
            buffer += [fmt.format(entity) for entity in block_entities]

        # append footer
        buffer.append(f"{space}End SubModelPart{block_name}")
        return buffer

    def write_submodelpart_body_model_part(self, body_model_part: BodyModelPart) -> List[str]:
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
            raise ValueError(
                f"Model part {body_model_part.name} is not a body model part!"
            )

        # check if mesh is initialised
        if body_model_part.mesh is None:
            raise ValueError(
                f"Model part {body_model_part.name} has not been meshed."
                f"Before creating the mdpa file, the model part needs to be meshed."
                f"Please run Model.generate_mesh()"
            )

        # initialise block
        block_text = ["", f"Begin SubModelPart {body_model_part.name}"]

        # define type entities
        entities: List[Optional[int]]

        # write tables
        block_text = self.__write_sub_model_part_block(
            block_text, block_name="Tables", block_entities=None
        )

        # write nodes
        entities = list(body_model_part.mesh.nodes.keys())
        block_text = self.__write_sub_model_part_block(
            block_text, block_name="Nodes", block_entities=entities
        )

        # write elements
        entities = list(body_model_part.mesh.elements.keys())
        block_text = self.__write_sub_model_part_block(
            block_text, block_name="Elements", block_entities=entities
        )
        block_text += [f"End SubModelPart", ""]
        return block_text

    def write_submodelpart_process_model_part(self, process_model_part: ModelPart) -> List[str]:
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
            raise ValueError(
                f"Model part {process_model_part.name} is not a process model part!"
            )

        # check if mesh is initialised
        if process_model_part.mesh is None:
            raise ValueError(
                f"Model part {process_model_part.name} has not been meshed."
                f"Before creating the mdpa file, the model part needs to be meshed."
                f"Please run Model.generate_mesh()"
            )

        # initialise block
        block_text = ["", f"Begin SubModelPart {process_model_part.name}"]

        # write tables
        entities = [table.id for table in self.__get_unique_tables_process_model_part(process_model_part)]
        block_text = self.__write_sub_model_part_block(
            block_text, block_name="Tables", block_entities=entities
        )

        # write nodes
        entities = list(process_model_part.mesh.nodes.keys())
        block_text = self.__write_sub_model_part_block(
            block_text, block_name="Nodes", block_entities=entities
        )

        # write conditions if the process contains condition elements
        if process_model_part.mesh.elements is not None:

            entities = list(process_model_part.mesh.elements.keys())

            if self.__check_if_process_writes_conditions(process_model_part):
                # write conditions
                block_text = self.__write_sub_model_part_block(
                    block_text, block_name="Conditions", block_entities=entities
                )
            elif isinstance(process_model_part.parameters, AdditionalProcessesParametersABC):
                # write elements for additional processes
                block_text = self.__write_sub_model_part_block(
                    block_text, block_name="Elements", block_entities=entities
                )

        block_text += [f"End SubModelPart", ""]
        return block_text

    @staticmethod
    def __write_element_line(mat_id: int, element: Element) -> str:
        """
        Writes an element to the mdpa format for Kratos

        Args:
            - mat_id (int): integer representing the material id connected to the element.
            - element (:class:`stem.mesh.Element`): element object to write to Kratos.

        Returns:
            - line (str): string representing an element (or condition) in Kratos.
        """
        # simplify space syntax
        space = " " * INDENTATION
        _node_ids = element.node_ids
        # assemble format for element/condition string
        # `  element_id  property_id  node_1 node_2 node_3 ... node_N`
        # where N=number of nodes of the element/condition
        _fmt = f"{space}{FORMAT_INTEGER}{space}{FORMAT_INTEGER}{space}" + " ".join(
            [FORMAT_INTEGER] * len(_node_ids)
        )
        line = _fmt.format(element.id, mat_id, *_node_ids)
        return line

    @staticmethod
    def __write_node_line(node: Node) -> str:
        """
        Writes a node to the mdpa format for Kratos

        Args:
            - node (:class:`stem.mesh.Node`): node object to write to Kratos.

        Returns:
            - line: string representing a node in Kratos.
        """
        # simplify space syntax
        space = " " * INDENTATION
        node_coords = node.coordinates
        # assemble format for nodal string
        #   node_id  coordinate_1 coordinate_2 coordinate_3
        _fmt = f"{space}{FORMAT_INTEGER}{space}" + " ".join(
            [FORMAT_FLOAT_LONG] * len(node_coords)
        )
        line = _fmt.format(node.id, *node_coords)
        return line

    @staticmethod
    def __write_table_line(time: float, value: float) -> str:
        """
        Write the line for a Kratos table.

        Args:
            - time (Union[int, float]): time at the j-th line of the table
            - value (float): value at the j-th line of the table

        Returns:
            - str: string corresponding to the j-th line in a table for Kratos.
        """
        # simplify space syntax
        space = " " * INDENTATION
        # assemble format for table string at line j
        #   time value
        _fmt = f"{space}{FORMAT_FLOAT_SHORT}{space}{FORMAT_FLOAT_SHORT}"
        return _fmt.format(time, value)

    def __write_table_block(self, table: Table) -> List[str]:
        """
        Writes a table to the mdpa format for Kratos.

        Args:
            - table (:class:`stem.table.Table`): table object to write to Kratos.

        Raises:
            - ValueError: if table id is not initialised.

        Returns:
            - block_text (List[str]): list of strings for the table. Each element is a line in the mdpa file.
        """

        # check initialisation of id
        if table.id is None:
            raise ValueError("Table id not initialised!")

        # initialise block
        block_text = ["", f"Begin Table {table.id} TIME VALUE"]
        block_text.extend(
            [
                self.__write_table_line(table.times[ix], table.values[ix])
                for ix in range(len(table.values))
             ]
        )
        block_text += [f"End Table", ""]
        return block_text

    @staticmethod
    def __map_gmsh_element_to_kratos(model: Model, model_part: ModelPart) -> Optional[str]:
        """
        Returns the corresponding element type based on the analysis type, the model part (condition or body)
        and type of element (e.g. rod vs beam or line load vs moving load).

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info on the solver settings and
                problem.
            - model_part (:class:`stem.model_part.ModelPart`): the model part

        Raises:
            - ValueError: if model dimension is not 2 or 3
            - ValueError: if mesh not initialised first
            - ValueError: if element types are not unique in the model part.
            - ValueError: if the analysis type is not specified.

        Returns:
            - Optional[str]: the Kratos element type
        """

        # get number of dimensions of the model
        if model.ndim != 2 and model.ndim != 3:
            raise ValueError(
                f"Model dimension {model.ndim} is not supported. Only 2D and 3D are supported."
            )
        else:
            n_dimensions = model.ndim

        # Check if mesh is initialised
        if model_part.mesh is None:
            raise ValueError(
                f"Model part {model_part.name} has not been meshed."
                f"Before creating the mdpa file, the model part needs to be meshed."
                f"Please run Model.generate_mesh()"
            )

        # check unique_elements
        element_part_type = np.unique(
            [element.element_type for element in model_part.mesh.elements.values()]
        )

        if len(element_part_type) > 1:
            raise ValueError(
                f"Model part {model_part.name} has more than 1 element type assigned."
                f"\n{element_part_type}. Error."
            )

        # get number of nodes per element
        n_nodes_element = len(next(iter(model_part.mesh.elements.values())).node_ids)

        # check analysis type
        if model.project_parameters is not None:
            analysis_type = model.project_parameters.settings.analysis_type
            # get element name from model part (body or condition)
            element_name = model_part.get_element_name(n_dimensions, n_nodes_element, analysis_type)
        else:
            raise ValueError(
                f"Analysis type not specified in the model. Please initialise the model with the analysis type."
            )

        return element_name

    @staticmethod
    def __check_if_mesh_is_present_in_model_part(model_part: ModelPart):
        """
        Check if the mesh is present in the model part.

        Args:
            - model_part (:class:`stem.model_part.ModelPart`): the model part

        Raises:
            - ValueError: if mesh not initialised first

        """
        # check if mesh is initialised
        if model_part.mesh is None:
            raise ValueError(
                f"Model part {model_part.name} has not been meshed."
                f"Before creating the mdpa file, the model part needs to be meshed."
                f"Please run Model.generate_mesh()"
            )

    def write_elements_body_model_part(self, body_model_part: BodyModelPart, mat_id: int, kratos_element_type: str) \
            -> List[str]:
        """
        Writes the elements of the body model part to the mdpa file

        Args:
            - body_model_part (:class:`stem.model_part.BodyModelPart`): the body model part to write to mdpa.
            - mat_id (int): the material id connected to the element block
            - kratos_element_type (str): the kratos element type

        Raises:
            - ValueError: if model part is not a body model part

        Returns:
            - block_text (List[str]): list of strings for the elements of the body model part. \
                Each element is a line in the mdpa file.
        """
        # validate part is body model part
        if not self.__is_body_model_part(body_model_part):
            raise ValueError(
                f"Model part {body_model_part.name} is not a body model part!"
            )

        # validate if mesh is present
        self.__check_if_mesh_is_present_in_model_part(body_model_part)

        # initialise block
        block_text = ["", f"Begin Elements {kratos_element_type}"]
        if body_model_part.mesh is not None:
            block_text.extend(
                [
                    self.__write_element_line(mat_id, el)
                    for el in body_model_part.mesh.elements.values()
                ]
            )
        block_text += [f"End Elements", ""]
        return block_text

    def write_conditions_process_model_part(self, process_model_part: ModelPart, mat_id: int,
                                            kratos_element_type: str) -> List[str]:
        """
        Writes the conditions of the process model part to the mdpa file.

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
            raise ValueError(
                f"Model part {process_model_part.name} is not a process model part!"
            )

        # validate if mesh is present
        self.__check_if_mesh_is_present_in_model_part(process_model_part)

        # no elements to write to conditions or process doesn't write condition elements

        if ((process_model_part.mesh is not None) and (process_model_part.mesh.elements is not None)
                and self.__check_if_process_writes_conditions(process_model_part)):

            block_text = ["", f"Begin Conditions {kratos_element_type}"]
            block_text.extend(
                [self.__write_element_line(mat_id, el) for el in process_model_part.mesh.elements.values()]
            )
            block_text += [f"End Conditions", ""]
        else:
            block_text = []

        return block_text

    def __write_all_nodes(self, model: Model) -> List[str]:
        """
        Writes nodes to mdpa format.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info on the nodes.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        nodes_dict = model.get_all_nodes()
        # sort by key
        nodes_dict = dict(sorted(nodes_dict.items()))
        block_text = ["", "Begin Nodes"]
        block_text.extend(
            [self.__write_node_line(node) for node in nodes_dict.values()]
        )
        block_text += ["End Nodes", ""]
        return block_text

    def __write_all_tables(self, model: Model) -> List[str]:
        """
        Writes tables to mdpa format.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info on the tables.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """

        unique_tables = self.__get_unique_tables(model)
        block_text = []
        for table in unique_tables:
            block_text.extend(self.__write_table_block(table))
        return block_text

    @staticmethod
    def __write_property_ids(model: Model) -> List[str]:
        """
        Writes the block initialising the material ids (properties).

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info of the materials.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        # get the unique ids in material and conditions
        ids_to_write = list(set([mp.id for mp in model.get_all_model_parts() if mp.id is not None]))

        # get the unique ids and write properties
        block_text = []
        for _id in np.sort(ids_to_write):
            block_text.extend(["", f"Begin Properties {_id}", "End Properties", ""])
        return block_text

    def __write_elements_model(self, model: Model) -> List[str]:
        """
        Returns the mdpa block related to elements.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info of the elements.

        Raises:
            - ValueError: if id of body model part is not initialised.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []
        # write elements per body model part
        for bmp in model.body_model_parts:
            if bmp.id is None:
                raise ValueError(
                    f"Body model part {bmp.name} has no id."
                    "First, material parameters needs to be written to json."
                )

            # get the element type
            element_type = self.__map_gmsh_element_to_kratos(model, bmp)
            # write text block with elements
            if element_type is not None:
                block_text.extend(
                    self.write_elements_body_model_part(
                        mat_id=bmp.id, kratos_element_type=element_type, body_model_part=bmp
                    )
                )
        return block_text

    def __write_conditions_model(self, model: Model) -> List[str]:
        """
        Returns the mdpa block related to conditions.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info of the conditions.

        Raises:
            - ValueError: if id of process model part is not initialised.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []
        # write per conditions per process model part
        for pmp in model.process_model_parts:

            if self.__check_if_process_writes_conditions(pmp):

                # get the condition element type
                condition_type = self.__map_gmsh_element_to_kratos(model, pmp)

                if pmp.id is None:
                    raise ValueError(
                        f"Process model part id of part {pmp.name} not initialised."
                    )

                # write text block with conditions
                if condition_type is not None:
                    block_text.extend(
                        self.write_conditions_process_model_part(
                            mat_id=pmp.id,
                            kratos_element_type=condition_type,
                            process_model_part=pmp,
                        )
                    )
        return block_text

    def __write_submodel_parts(self, model: Model) -> List[str]:
        """
        Returns the mdpa block related to the submodel parts of process and body model parts.

        Args:
            - model (:class:`stem.model.Model`): the model object containing the info of the conditions.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = []

        for bmp in model.body_model_parts:
            block_text.extend(self.write_submodelpart_body_model_part(bmp))

        for pmp in model.process_model_parts:
            block_text.extend(self.write_submodelpart_process_model_part(pmp))

        return block_text

    def __write_mdpa_text(self, model: Model) -> List[str]:
        """
        Returns the  mesh data to mdpa format as list of strings representing each a line in the mdpa file.

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info of the model parts.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """

        # initialise process model part ids
        self.__initialise_process_model_part_ids(model)
        self.__initialise_body_model_part_ids(model)
        self.__initialise_table_ids(model)
        block_text = []
        # retrieve the materials in the model and write mdpa text blocks
        block_text.extend(self.__write_property_ids(model))

        # write the table block
        block_text.extend(self.__write_all_tables(model))

        # retrieve the unique nodes of all the model parts
        block_text.extend(self.__write_all_nodes(model))

        # write elements per body model part
        block_text.extend(self.__write_elements_model(model))

        # write conditions per process model part
        block_text.extend(self.__write_conditions_model(model))

        # write submodel parts
        block_text.extend(self.__write_submodel_parts(model))

        return block_text

    def __write_mesh_to_mdpa(self, model: Model, mesh_file_name: str) -> List[str]:
        """
        Saves mesh data to mdpa file.

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info on the \
                materials.
            - mesh_file_name (str): The name of the mesh file to store the mdpa file.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = self.__write_mdpa_text(model)

        # append EOL to each line
        output_formatted_txt = [f"{line}\n" for line in block_text]

        output_folder_pth = Path(self.project_folder)
        output_folder_pth.mkdir(exist_ok=True, parents=True)

        output_path = output_folder_pth.joinpath(mesh_file_name)
        # if no suffix or wrong suffix change to mdpa
        output_path = output_path.with_suffix(".mdpa")

        with open(output_path, "w") as _buf:
            _buf.writelines(output_formatted_txt)

        return output_formatted_txt

    def __write_material_parameters_json(self, model: Model, materials_file_name: str = "MaterialParameters.json") -> Dict[str, Any]:
        """
        Writes the material parameters to json format for Kratos.

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info on the \
                materials.
            - materials_file_name (str): name of the material parameters file. Defaults to `MaterialParameters.json`.

        Raises:
            - ValueError: if material is not assigned to the body model part
            - ValueError: if material id is not initialised

        Returns:
            - materials_dict Dict([str, Any]): dictionary containing the material parameters' dictionary.
        """

        materials_dict: Dict[str, Any] = {"properties": []}

        # initialise the model ids
        self.initialise_model_ids(model)

        # iterate over the body model parts and create materials
        for bmp in model.body_model_parts:

            if bmp.material is None:
                raise ValueError(f"Body model part {bmp.name} has no material assigned.")

            if bmp.id is None:
                raise ValueError(f"Body model part {bmp.name} has no id initialised.")

            materials_dict["properties"].append(
                self.material_io.create_material_dict(
                    part_name=bmp.name,
                    material=bmp.material,
                    material_id=bmp.id,
                )
            )

        # write the material parameters file to json
        IOUtils.write_json_file(self.project_folder, materials_file_name, materials_dict)

        return materials_dict

    def __create_solver_settings_dictionary(self, model: Model, mesh_file_name: str, materials_file_name: str) \
            -> Dict[str, Any]:
        """
        Creates a dictionary containing the solver settings.

        Args:
            - model (:class:`stem.model.Model`): The model object containing the solver data and model parts.
            - mesh_file_name (str): The name of the mesh file.
            - materials_file_name (str): The name of the materials parameters json file.

        Raises:
            - ValueError: if solver_settings in model are not initialised.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters
                dictionary related to problem data and solver settings.
        """

        if model.project_parameters is None:
            raise ValueError("Solver settings are not initialised in model.")

        return self.solver_io.create_settings_dictionary(
            model.project_parameters,
            Path(mesh_file_name).stem,
            materials_file_name,
            model.get_all_model_parts(),
        )

    def __create_folder_for_json_output(self, output_settings: Optional[List[Output]] = None):
        """
        Creates the output folder for the JSON outputs if specified.

        Args:
            - output_settings (Optional[List[:class:`stem.output.Output`]]): The list of output processes objects to \
              write  in outputs.

        """

        if output_settings is not None and len(output_settings) > 0:
            for output in output_settings:

                # create folder for json output
                if isinstance(output.output_parameters, JsonOutputParameters):

                    # check if the output folder is absolute or relative, if relative create it in the project folder
                    if os.path.isabs(output.output_dir):
                        json_output_folder = output.output_dir

                    else:
                        json_output_folder = os.path.join(self.project_folder, output.output_dir)

                    os.makedirs(json_output_folder, exist_ok=True)

    def __create_output_process_dictionary(self, output_settings: Optional[List[Output]] = None) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output settings.

        Args:
            - output_settings (Optional[List[:class:`stem.output.Output`]]): The list of output processes objects to \
              write  in outputs.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters dictionary related to outputs
        """
        if output_settings is None or len(output_settings) == 0:
            return {"output_processes": {}, "processes": {}}
        else:
            self.__create_folder_for_json_output(output_settings=output_settings)
            return self.outputs_io.create_output_process_dictionary(output_settings=output_settings)

    def __create_process_model_parts_dictionary(self, model: Model) -> Dict[str, Any]:
        """
        Creates a dictionary containing the process dictionaries from the process model parts, i.e. loads, boundary
        conditions and additional processes (random field, excavation, etc.).

        Args:
            - model (:class:`stem.model.Model`): The model object containing the process model parts.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters dictionary related \
                to loads, boundary conditions and additional processes.

        """
        processes_dict: Dict[str, Any] = {
            "processes": {"constraints_process_list": [], "loads_process_list": []}
        }

        # loop on the process model parts
        for mp in model.process_model_parts:

            # add load
            if isinstance(mp.parameters, LoadParametersABC):
                parameters = self.loads_io.create_load_dict(mp.name, mp.parameters)
                processes_dict["processes"]["loads_process_list"].append(parameters)

            # add boundary condition
            elif isinstance(mp.parameters, BoundaryParametersABC):
                parameters = self.boundaries_io.create_boundary_condition_dict(
                    mp.name, mp.parameters
                )

                if mp.parameters.is_constraint:
                    _key = "constraints_process_list"
                else:
                    _key = "loads_process_list"
                processes_dict["processes"][_key].append(parameters)

            elif isinstance(mp.parameters, WaterProcessParametersABC):
                parameters = self.water_boundaries_io.create_water_process_dict(
                    mp.name, mp.parameters
                )
                processes_dict["processes"]["loads_process_list"].append(parameters)

            elif isinstance(mp.parameters, AdditionalProcessesParametersABC):

                # Validations and adjustment for json_file parameter field:
                if isinstance(mp.parameters, ParameterFieldParameters) and mp.parameters.function_type == "json_file":
                    self.__adjust_parameter_field_parameters_and_write_json_file(
                        process_model_part=mp
                    )

                # write the additional process model part parameters for the
                # project parameters file
                processes_dict["processes"]["constraints_process_list"].append(
                    self.additional_process_io.create_additional_processes_dict(
                        mp.name, mp.parameters
                    )
                )

        return processes_dict

    def __adjust_parameter_field_parameters_and_write_json_file(self, process_model_part: ModelPart) -> None:
        """
        Adjusts the additional process parameters when the parameter field parameter is
        of type `json_file`. It also writes the json file with the parameter values.

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): the process model part for which the field \
                parameters require adjustment.

        Raises:
            - ValueError: if the `field_file_name` attribute in the parameters is None.
            - ValueError: if the `field_generator` attribute in the parameters is None.

        Returns:
            - None

        """

        # required for validation:
        # Process model part has to be a field parameter process model part
        if not isinstance(process_model_part.parameters, ParameterFieldParameters):
            return None

        # Process model part has to be of `json_file` type
        if not process_model_part.parameters.function_type == "json_file":
            return None

        # check that the name is not none!
        if process_model_part.parameters.field_file_name is None:
            raise ValueError("No name was provided for the json file containing the "
                             f"field parameters of model part {process_model_part.name} and property"
                             f" {process_model_part.parameters.property_name}.")

        # adjust extension of filename name is not none, check that extension is json and change it if not.
        process_model_part.parameters.field_file_name = Utils.replace_extensions(
            process_model_part.parameters.field_file_name, ".json"
        )

        # check that the name is not none!
        if process_model_part.parameters.field_generator is None:
            raise ValueError("Field generator object not provided for the field generation"
                             f" of model part {process_model_part.name} and "
                             f"property {process_model_part.parameters.property_name}.")

        # write field values in the json input file
        IOUtils.write_json_file(
            output_folder=self.project_folder,
            file_name=process_model_part.parameters.field_file_name,
            dictionary={"values": process_model_part.parameters.field_generator.generated_field}
        )

    @staticmethod
    def __create_set_nodal_parameters_process_dictionary(model_part: BodyModelPart) -> Dict[str, Any]:
        """
        Creates a dictionary containing the nodal parameters for the nodal concentrated element and elastic spring
        damper.

        Args:
            - model_part (:class:`stem.model_part.BodyModelPart`): The body model part containing the nodal parameters.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters dictionary related \
                to nodal parameters

        """

        parameters = {"python_module": "set_nodal_parameters_process",
                      "kratos_module": "StemApplication",
                      "process_name": "SetNodalParametersProcess",
                      "Parameters": {"model_part_name": f"{DOMAIN}.{model_part.name}"}}

        return parameters

    def __create_auxiliary_process_list_dictionary(self, model: Model) -> Dict[str, Any]:
        """
        Creates a dictionary containing the auxiliary processes.

        Args:
            - model (:class:`stem.model.Model`): The model object containing the process model parts.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters dictionary related \
                to auxiliary processes.

        """
        processes_dict: Dict[str, Any] = {
            "processes": {"auxiliary_process_list": []}
        }

        # loop over body model parts
        for bmp in model.body_model_parts:
            if bmp.material is None:
                raise ValueError(f"Body model part {bmp.name} has no material assigned.")

            if bmp.id is None:
                raise ValueError(f"Body model part {bmp.name} has no id initialised.")

            # add nodal parameters from elastic spring damper and nodal concentrated element to auxiliary process list
            if (isinstance(bmp.material, StructuralMaterial) and
                    isinstance(bmp.material.material_parameters, (ElasticSpringDamper, NodalConcentrated))):
                parameters = self.__create_set_nodal_parameters_process_dictionary(bmp)
                processes_dict["processes"]["auxiliary_process_list"].append(parameters)

        return processes_dict

    def __write_project_parameters_json(self, model: Model, mesh_file_name: str, materials_file_name: str,
                                        project_file_name: str = "ProjectParameters.json") -> Dict[str, Any]:
        """
        Writes project parameters to json file

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info, i.e. \
                body and process model parts, boundary conditions, solver settings and problem data.
            - mesh_file_name (str): The name of the mesh file.
            - materials_file_name (str): The name of the materials file.
            - project_file_name (str): name of the project parameters file. Defaults to `ProjectParameters.json`.

        Returns:
            - Dict[str, Any]: the dictionary containing the project parameters.
        """
        # initialise material, tables and process model part ids
        self.initialise_model_ids(model)

        # get the solver dictionary
        solver_dict = self.__create_solver_settings_dictionary(
            model, mesh_file_name, materials_file_name
        )
        # get the output dictionary
        outputs_dict = self.__create_output_process_dictionary(output_settings=model.output_settings)
        # get the boundary condition and loads dictionary
        process_model_part_dict = self.__create_process_model_parts_dictionary(model=model)
        # get the auxiliary processes dictionary
        auxiliary_processes_dict = self.__create_auxiliary_process_list_dictionary(model=model)
        # merge dictionaries into one
        project_parameters_dict: Dict[str, Any] = reduce(
            Utils.merge, (solver_dict, outputs_dict, process_model_part_dict, auxiliary_processes_dict)
        )
        # write json file
        IOUtils.write_json_file(self.project_folder, project_file_name, project_parameters_dict)

        return project_parameters_dict

    def write_input_files_for_kratos(self, model: Model, mesh_file_name: str,
                                     materials_file_name: str = "MaterialParameters.json",
                                     project_file_name: str = "ProjectParameters.json"):
        """
        Writes all required input files for a Kratos simulation, i.e: project parameters json; material parameters json
        and the mdpa mesh file.

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info.
            - mesh_file_name (str): The name of the mesh file.
            - materials_file_name (str): The name of the materials file.
            - project_file_name (str): name of the project parameters file. Defaults to `ProjectParameters.json`.
        """

        # write materials
        self.__write_material_parameters_json(model, materials_file_name)

        # write project parameters
        self.__write_project_parameters_json(model, mesh_file_name, materials_file_name, project_file_name)

        # write mdpa files
        self.__write_mesh_to_mdpa(model, mesh_file_name)
