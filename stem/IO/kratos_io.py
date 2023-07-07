import json
from copy import deepcopy
from functools import reduce
from typing import List, Dict, Any, Optional

from stem.IO.kratos_boundaries_io import KratosBoundariesIO
from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.IO.kratos_output_io import KratosOutputsIO
from stem.boundary import BoundaryParametersABC
from stem.load import LoadParametersABC
from stem.model import Model
from stem.output import Output
from stem.utils import merge

DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        - ndim (int): The number of dimensions of the problem.
        - model (:class:`stem.model.Model`): The model object with information on parts, node and elements.
        - output (Optional[List[:class:`stem.output.Output`]]): The number of dimensions of the problem.
        - material_io (:class:`stem.io.kratos_loads_io.KratosMaterialIO`): The
            material IO object.
        - loads_io (:class:`stem.io.kratos_loads_io.KratosLoadsIO`): The loads IO
            object.
        - boundaries_io (:class:`stem.io.kratos_loads_io.KratosBoundariesIO`): The
            boundary conditions IO object.
        - outputs_io (:class:`stem.io.kratos_loads_io.KratosOutputsIO`): The outputs IO
            object.
    """

    def __init__(
        self,
        ndim: int,
        model: Model,
        outputs: Optional[List[Output]] = None,
    ):
        """
        Constructor of KratosIO class

        Args:
            - ndim: The number of dimensions of the problem.
        """
        self.model = model
        self.ndim = ndim
        self.outputs = outputs

        self.material_io = KratosMaterialIO(self.ndim)
        self.loads_io = KratosLoadsIO(DOMAIN)
        self.boundaries_io = KratosBoundariesIO(DOMAIN)
        self.outputs_io = KratosOutputsIO(DOMAIN)

    def write_material_parameters_json(self, filename: Optional[str] = None) -> Dict[str,Any]:
        """
        Writes material parameters to json file.

        Args:
            filename (str): name of the material json file for kratos.

        Returns:
            materials_dict (Dict[str,Any]): dictionary of the material properties.
        """
        materials_dict : Dict[str,Any] = {"properties": []}
        for ix, bmp in enumerate(self.model.body_model_parts):
            bmp.id = ix + 1

            materials_dict["properties"].append(
                self.material_io.create_material_dict(
                    part_name=f"{DOMAIN}.{bmp.name}", material=bmp.parameters, material_id=bmp.id
                )
            )

        if filename is not None:
            json.dump(materials_dict, open(filename, "w"), indent=4)

        return materials_dict

    def write_project_parameters_json(
        self,
        filename: Optional[str] = None,
        solver_settings=None,
    ):
        """
        Writes the project parameters to json.

        Args:
            - filename (Optional[str]): string defining the name of the outputfile. Should be a JSON file.
            - solver_settings (Any): list of Solver settings objects TODO: add when ready.

        Returns:
            - processes_dict (Dict[str, Any]): dictionary of Kratos processes containing loads, boundary conditions
                and outputs.
        """
        processes_dict: Dict[str, Any] = {
            "processes": {"constraints_process_list": [], "loads_process_list": []}
        }

        for ix, mp in enumerate(self.model.model_parts):
            # TODO: define the properties for the conditions
            mp.id = ix + 1

            if isinstance(mp.parameters, LoadParametersABC):
                _parameters = self.loads_io.create_load_dict(mp.name, mp.parameters)
                processes_dict["processes"]["loads_process_list"].append(_parameters)

            elif isinstance(mp.parameters, BoundaryParametersABC):
                _parameters = self.boundaries_io.create_boundary_condition_dict(
                    mp.name, mp.parameters
                )
                _key = "loads_process_list"
                if mp.parameters.is_constraint:
                    _key = "constraints_process_list"
                processes_dict["processes"][_key].append(_parameters)

        _dict_merged = deepcopy(processes_dict)

        if self.outputs is not None:
            output_dict = self.outputs_io.create_output_process_dictionary(
                outputs=self.outputs
            )
            # recursive merging
            _dict_merged = reduce(merge, (_dict_merged, output_dict))

        if filename is not None:
            json.dump(_dict_merged, open(filename, "w"), indent=4)

        return processes_dict

    def __write_solver_settings(self):
        pass

    def write_mesh_to_mdpa(self, *args, **kwargs):
        """
        Writes the parts, sets and nodes in the model as mdpa
        format for Kratos and returns a list containing the strings making the mdpa file.

        Args:
            - ind (int): indentation level of the mdpa file. Default is 2.
            - fmt_coord (str): format of the coordinates to be printed. Default is
                `.10f`.
            - fmt_id (str): format of the ids to be printed. Default is `d`.
            - filename (Optional[str]): name of the mdpa file. if suffix is not provided or is
                not mdpa, mdpa is added instead. If `None`, no file is created.
            - output_dir (str): relative of absolute path to the directory where
                the mdpa file is to be stored.
            - linebreak (str): linebreak to provide at the end of each line. If None is
                ignored.

        Returns:
            - List[str]: list containing the string for the mdpa files
        """

        return self.model.write_mdpa_file(*args, **kwargs)
