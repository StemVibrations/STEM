import json
from functools import reduce
from pathlib import Path
from typing import List, Dict, Any, Optional

from stem.IO.kratos_boundaries_io import KratosBoundariesIO
from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.IO.kratos_model_io import KratosModelIO
from stem.IO.kratos_output_io import KratosOutputsIO
from stem.IO.kratos_solver_io import KratosSolverIO
from stem.boundary import BoundaryParametersABC
from stem.load import LoadParametersABC
from stem.model import Model
from stem.output import Output
from stem.utils import Utils

DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        - ndim (int): The number of dimensions of the problem.
        - loads_io (:class:`stem.IO.kratos_loads_io.KratosLoadsIO`): The loads IO object.
        - material_io (:class:`stem.IO.kratos_material_io.KratosMaterialIO`): The material IO object.
    """

    def __init__(self, ndim: int):
        """
        Constructor of KratosIO class

        Args:
            - ndim: The number of dimensions of the problem.
        """

        self.ndim = ndim

        self.material_io = KratosMaterialIO(self.ndim, DOMAIN)
        self.loads_io = KratosLoadsIO(DOMAIN)
        self.boundaries_io = KratosBoundariesIO(DOMAIN)
        self.outputs_io = KratosOutputsIO(DOMAIN)
        self.model_io = KratosModelIO(self.ndim, DOMAIN)
        self.solver_io = KratosSolverIO(self.ndim, DOMAIN)

    def write_mesh_to_mdpa(self, model: Model, mesh_file_name: str, output_folder="./"):
        """
        Saves mesh data to mdpa file.

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info on the \
                materials.
            - mesh_file_name (str): The name of the mesh file to store the mdpa file.
            - output_folder (str): folder to store the project parameters file. Defaults to the working directory.

        Returns:
            - block_text (List[str]): list of strings for the mdpa file. Each element is a line in the mdpa file.
        """
        block_text = self.model_io.write_mdpa_text(model)

        # append EOL to each line
        output_formatted_txt = [line + "\n" for line in block_text]

        output_folder_pth = Path(output_folder)
        output_folder_pth.mkdir(exist_ok=True, parents=True)

        output_path = output_folder_pth.joinpath(mesh_file_name)
        # if no suffix or wrong suffix change to mdpa
        output_path = output_path.with_suffix(".mdpa")

        with open(output_path, "w") as _buf:
            _buf.writelines(output_formatted_txt)

        return output_formatted_txt

    def write_material_parameters_json(
        self,
        model: Model,
        materials_file_name: str = "MaterialParameters.json",
        output_folder: str = "./"
    ):
        """
        Writes the material parameters to json format for Kratos.

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info on the \
                materials.
            - materials_file_name (str): name of the material parameters file. Defaults to `MaterialParameters.json`.
            - output_folder (str): folder to store the material parameters file. Defaults to the working directory.

        Raises:
            - ValueError: if material is not assigned to the body model part
            - ValueError: if material id is not initialised

        Returns:
            - materials_dict Dict([str, Any]): dictionary containing the material parameters' dictionary.
        """

        materials_dict: Dict[str, Any] = {"properties": []}

        # initialise the model ids
        self.model_io.initialise_model_ids(model)

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
        output_folder_pth = Path(output_folder)
        output_folder_pth.mkdir(exist_ok=True, parents=True)
        output_path_file = output_folder_pth.joinpath(materials_file_name)
        json.dump(materials_dict, open(output_path_file, "w"), indent=4)

        return materials_dict

    def __write_solver_settings(
        self, model: Model, mesh_file_name: str, materials_file_name: str
    ):
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
            print("WARNING: Solver settings are undefined in model.")
            return {"output_processes": {}, "processes": {}}

        return self.solver_io.create_settings_dictionary(
            model.project_parameters,
            Path(mesh_file_name).stem,
            materials_file_name,
            model.get_all_model_parts(),
        )

    def __write_output_processes(self, outputs: Optional[List[Output]] = None):
        """
        Creates a dictionary containing the output settings.

        Args:
            - outputs (Optional[List[:class:`stem.output.Output`]]): The list of output processes objects to write \
                in outputs.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters dictionary related to outputs
        """
        if outputs is None or len(outputs) == 0:
            return {"output_processes": {}, "processes": {}}
        else:
            return self.outputs_io.create_output_process_dictionary(outputs=outputs)

    def __write_loads_and_constraints(self, model: Model):
        """
        Creates a dictionary containing the loads and boundary conditions.

        Args:
            - model (:class:`stem.model.Model`): The model object containing the process model parts.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters dictionary related \
                to loads and boundary conditions
        """
        processes_dict: Dict[str, Any] = {
            "processes": {"constraints_process_list": [], "loads_process_list": []}
        }

        # loop on the process model parts
        for mp in model.process_model_parts:

            # add load
            if isinstance(mp.parameters, LoadParametersABC):
                _parameters = self.loads_io.create_load_dict(mp.name, mp.parameters)
                processes_dict["processes"]["loads_process_list"].append(_parameters)

            # add boundary condition
            elif isinstance(mp.parameters, BoundaryParametersABC):
                _parameters = self.boundaries_io.create_boundary_condition_dict(
                    mp.name, mp.parameters
                )
                _key = "loads_process_list"
                if mp.parameters.is_constraint:
                    _key = "constraints_process_list"
                processes_dict["processes"][_key].append(_parameters)

        return processes_dict

    def write_project_parameters_json(
        self,
        model: Model,
        outputs: List[Output],
        mesh_file_name: str,
        materials_file_name: str,
        project_file_name: str = "ProjectParameters.json",
        output_folder: str = "./"
    ):
        """
        Writes project parameters to json file

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info, i.e. \
                body and process model parts, boundary conditions, solver settings and problem data.
            - outputs (List[:class:`stem.output.Output`]): The list of output processes objects to write in outputs.
            - mesh_file_name (str): The name of the mesh file.
            - materials_file_name (str): The name of the materials file.
            - project_file_name (str): name of the project parameters file. Defaults to `ProjectParameters.json`.
            - output_folder (str): folder to store the project parameters file. Defaults to the working directory.

        Returns:
            - project_parameters_dict (Dict[str, Any]): the dictionary containing the project parameters.
        """
        # initialise material, tables and process model part ids
        self.model_io.initialise_model_ids(model)

        # get the solver dictionary
        solver_dict = self.__write_solver_settings(
            model, mesh_file_name, materials_file_name
        )
        # get the output dictionary
        outputs_dict = self.__write_output_processes(outputs=outputs)
        # get the boundary condition dictionary
        loads_and_bc_dict = self.__write_loads_and_constraints(model=model)
        # TODO get the additional_processes dictionary

        # merge dictionaries into one
        project_parameters_dict: Dict[str, Any] = reduce(
            Utils.merge, (solver_dict, outputs_dict, loads_and_bc_dict)
        )
        # write json file
        output_folder_pth = Path(output_folder)
        output_folder_pth.mkdir(exist_ok=True, parents=True)

        output_path_file = output_folder_pth.joinpath(project_file_name)
        json.dump(project_parameters_dict, open(output_path_file, "w"), indent=4)

        return project_parameters_dict

    def write_input_files_for_kratos(
        self,
        model: Model,
        outputs: List[Output],
        mesh_file_name: str,
        materials_file_name: str = "MaterialParameters.json",
        project_file_name: str = "ProjectParameters.json",
        output_folder: str = "./"
    ):
        """
        Writes all required input files for a Kratos simulation, i.e: project parameters json; material parameters json
        and the mdpa mesh file

        Args:
            - model (:class:`stem.model.Model`): The model object containing all the required info.
            - outputs (List[:class:`stem.output.Output`]): The list of output processes objects to write in outputs.
            - mesh_file_name (str): The name of the mesh file.
            - materials_file_name (str): The name of the materials file.
            - project_file_name (str): name of the project parameters file. Defaults to `ProjectParameters.json`.
            - output_folder (str): folder to store the project parameters file. Defaults to the working directory.
        """

        # write materials
        self.write_material_parameters_json(model, materials_file_name, output_folder)

        # write project parameters
        self.write_project_parameters_json(
            model,
            outputs,
            mesh_file_name,
            materials_file_name,
            project_file_name,
            output_folder
        )

        # write mdpa files
        self.write_mesh_to_mdpa(model, mesh_file_name, output_folder)