from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

from stem.mesh import Mesh, Node
from stem.model_part import ModelPart, BodyModelPart


@dataclass
class Model:
    """
    A class to represent the main model.

    Attributes:
        - ndim (int): number of dimensions of the model.
        - solver (:class:`stem.solver.Solver`): The solver used to solve the problem.
        - body_model_parts (List[:class:`stem.model_part.BodyModelPart`]): A list containing the body model parts.
        - model_parts (List[:class:`stem.model_part.ModelPart`]): A list containing the process model parts.
        - project_parameters (Optional[:class:`stem.mesh.Mesh`]): A dictionary containing the project parameters.

    """
    ndim: int
    solver = None
    body_model_parts: List[BodyModelPart] = field(default_factory=list)
    model_parts: List[ModelPart] = field(default_factory=list)
    mesh: Optional[Mesh] = None

    def __write_nodes(self, *args, **kwargs):
        if self.mesh is not None:
            return self.mesh.write_nodes_for_mdpa(*args, **kwargs)

    def __write_element_part_block(self, ind:int = 2, fmt_id: str = "{:d}"):
        """
        Writes the element blocks within a model part or a body model parts.
        From Body model parts, elements are written and for model parts conditions
        are written.

        Args:
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_id (str): format of the ids to be printed. Default is `d`.

        Returns:
            - _out List[str]: list containing the string for the mdpa files
        """
        _elements_txt = []
        # TODO: improve the split in Elements and Conditions
        for bmp in self.body_model_parts:
            _elements_txt += bmp.write_elements(ndim=self.ndim, ind=ind, fmt_id=fmt_id)
        for mp in self.model_parts:
            _elements_txt += mp.write_elements(ndim=self.ndim, ind=ind, fmt_id=fmt_id)

        return _elements_txt

    def __write_sub_model_part_block(self, ind:int = 2, fmt_id: str = "{:d}"):
        """
        Writes the sub-model blocks of all the model parts and body model parts.

        Args:
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_id (str): format of the ids to be printed. Default is `d`.

        Returns:
            - _sub_models_txt List[str]: list containing the string for the mdpa files
        """
        _sub_models_txt = []
        for bmp in self.body_model_parts:
            _sub_models_txt += bmp.write_sub_model_blocks(ind=ind, fmt_id=fmt_id)
        for mp in self.model_parts:
            _sub_models_txt += mp.write_sub_model_blocks(ind=ind, fmt_id=fmt_id)

        return _sub_models_txt

    def __write_property_block(self, property_id:int) -> List[str]:
        """
        Writes the block for the material id.
        Args:
            property_id (int): the material index.

        Returns:

        """
        return ["", f"Begin Properties {property_id}","End Properties", ""]

    def write_mdpa_file(
            self,
            ind: int=2,
            fmt_coord: str = "{:.10f}",
            fmt_id: str = "{:d}",
            filename: Optional[str] = None,
            output_dir: str = "",
            linebreak: str = "\n",
    ):
        """
        Writes the parts, sets and nodes in the model as mdpa
        format for Kratos and returns a list containing the strings making the mdpa file.

        Args:
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_coord (str): format of the coordinates to be printed. Default is
                `.10f`.
            - fmt_id (str): format of the ids to be printed. Default is `d`.
            - filename (str): name of the mdpa file. if suffix is not provided or is
                not mdpa, mdpa is added instead. If `None`, no file is created.
            - output_dir (str): relative of absolute path to the directory where
                the mdpa file is to be stored.
            - linebreak (str): linebreak to provide at the end of each line. If None is
                ignored.

        Returns:
            - _out List[str]: list containing the string for the mdpa files
        """

        # write the nodes!
        nodes_txt = self.__write_nodes(ind=ind, fmt_id=fmt_id, fmt_coord=fmt_coord)
        element_blocks = self.__write_element_part_block(ind=ind, fmt_id=fmt_id)
        sub_model_blocks = self.__write_sub_model_part_block(ind=ind, fmt_id=fmt_id)

        property_ids = [bmp.id for bmp in self.body_model_parts]

        property_txt = []
        for property_id in property_ids:
            if property_id is not None:
                property_txt += self.__write_property_block(property_id=property_id)

        _out = property_txt + nodes_txt + element_blocks + sub_model_blocks

        if linebreak is not None:
            _out = [_ + linebreak for _ in _out]

        if filename is not None:
            output_dir_pth = Path(output_dir)
            output_dir_pth.mkdir(parents=True, exist_ok=True)

            output_path = output_dir_pth.joinpath(filename)
            output_path = output_path.with_suffix(".mdpa")

            with open(output_path, "w") as _buf:
                _buf.writelines(_out)
        return _out


