import os
from copy import deepcopy
from pathlib import Path
from typing import List, Dict
import warnings

import KratosMultiphysics
from KratosMultiphysics.StemApplication.geomechanics_analysis import StemGeoMechanicsAnalysis

from stem.model import Model
from stem.solver import SolutionType
from stem.output import VtkOutputParameters, GiDOutputParameters, JsonOutputParameters
from stem.IO.kratos_io import KratosIO


class Stem:
    """
    Class containing the main calculation.

    Attributes:
        - input_files_dir (str): The directory where the input files are to be written.
        - kratos_io (:class:`stem.IO.kratos_io.KratosIO`): The Kratos IO object.
        - kratos_model (KratosMultiphysics.Model): The Kratos model.
        - __stages (List[:class:`stem.model.Model`]): The calculation stages.
        - __stage_settings_file_names (Dict[int, str]): The file names of the project parameters files for each stage.
        - __last_ran_stage_number (int): The number of the last ran stage.
        - __last_uvec_data (KratosMultiphysics.Parameters): The uvec data from the last ran stage

    """

    def __init__(self, initial_stage: Model, input_files_dir: str):
        """
        Constructor of the main calculation class.

        Args:
            - initial_stage (:class:`stem.model.Model`): The initial stage of the calculation.
            - input_files_dir (str): The directory where the input files are to be written.
        """
        self.input_files_dir = input_files_dir
        self.kratos_io = KratosIO(initial_stage.ndim)
        self.kratos_model = KratosMultiphysics.Model()

        self.__stages: List[Model] = [initial_stage]
        self.__stage_settings_file_names: Dict[int, str] = {}
        self.__last_ran_stage_number: int = 0

        self.__last_uvec_data = KratosMultiphysics.Parameters("""{"u": {},
                                                               "theta": {},
                                                               "loads": {},
                                                               "state": {}}""")

        # perform initial stage setup and mesh generation in this order
        initial_stage.post_setup()
        initial_stage.generate_mesh()

    @property
    def stages(self) -> List[Model]:
        """
        The calculation stages.

        Returns:
            - (List[:class:`stem.model.Model`]): The calculation stages.

        """
        return self.__stages

    def create_new_stage(self, delta_time: float, stage_duration: float) -> Model:
        """
        Create a new stage based on the last stage in the calculation. Note that the stage is not added to the
        calculation.

        Args:
            - delta_time (float): The time step of the new stage.
            - stage_duration (float): The duration of the new stage.

        Returns:
            - :class:`stem.model.Model`: The new stage.

        """
        # create a new stage based on the last stage
        new_stage = deepcopy(self.__stages[-1])

        # check if project parameters are set, both the last stage and the new stage have to be checked for mypy
        if new_stage.project_parameters is None or self.__stages[-1].project_parameters is None:
            raise Exception("Project parameters of the last stage are not set")

        # set the time integration settings of the new stage
        new_stage.project_parameters.settings.time_integration.start_time = (
            self.__stages[-1].project_parameters.settings.time_integration.end_time)
        new_stage.project_parameters.settings.time_integration.end_time = (
            new_stage.project_parameters.settings.time_integration.start_time + stage_duration)
        new_stage.project_parameters.settings.time_integration.delta_time = delta_time

        # set output directory and output name new stage
        self.__set_output_name_new_stage(new_stage, len(self.__stages) + 1)

        return new_stage

    def add_calculation_stage(self, stage: Model):
        """
        Add a calculation stage to the calculation. The geometry and the mesh of the new stage are regenerated.

        Args:
            - stage (:class:`stem.model.Model`): The model of the stage to be added.

        """
        self.__stages.append(stage)

        # check if the mesh is the same in the new stage
        self.validate_latest_stage()

    def validate_latest_stage(self):
        """
        Validate the latest stage. The validation checks if the number of body and process model parts are the same
        between the last two stages. And if the mesh is the same in the new stage.

        Raises:
            - Exception: If the number of body model parts are not the same between stages.
            - Exception: If the number of process model parts are not the same between stages.

        """

        # check if number of model parts is the same
        if len(self.__stages[-2].body_model_parts) != len(self.__stages[-1].body_model_parts):
            raise Exception("Number of body model parts are not the same between stages")

        # todo update kratos such that process model parts can be added
        if len(self.__stages[-2].process_model_parts) != len(self.__stages[-1].process_model_parts):
            raise Exception("Number of process model parts are not the same between stages")

        # check if the mesh is the same in the new stage
        self.__check_if_mesh_between_stages_is_the_same(self.__stages[-2], self.__stages[-1])

        # check solver settings new stage
        self.__check_if_acceleration_should_be_initialised(self.__stages[-2], self.__stages[-1])

    def write_all_input_files(self):
        """
        Write all input files for the calculation.

        """

        for stage_nr, stage in enumerate(self.stages):
            if stage.project_parameters is not None:

                # validate settings
                stage.project_parameters.settings.validate_settings()

                mesh_name = stage.project_parameters.problem_name + f"_stage_{stage_nr+1}.mdpa"
                project_settings_file_name = f"ProjectParameters_stage_{stage_nr+1}.json"
                material_settings_file_name = f"MaterialParameters_stage_{stage_nr+1}.json"
                self.kratos_io.project_folder = self.input_files_dir
                self.kratos_io.write_input_files_for_kratos(stage,
                                                            mesh_name,
                                                            materials_file_name=material_settings_file_name,
                                                            project_file_name=project_settings_file_name)

                self.__stage_settings_file_names[stage_nr + 1] = project_settings_file_name

    def run_stage(self, stage_number: int):
        """
        Runs a single stage of the calculation.

        Args:
            - stage_number (int): The number of the stage to be run.

        """

        # check if the stages are run in order
        if stage_number != self.__last_ran_stage_number + 1:
            raise Exception("Stages should be run in order")

        # change working directory to input files directory
        cwd = os.getcwd()
        os.chdir(self.input_files_dir)

        # get file name of project parameters file for the current stage
        parameters_file_name = self.__stage_settings_file_names[stage_number]

        # read project parameters file
        with open(parameters_file_name, "r") as parameter_file:
            kratos_parameters = KratosMultiphysics.Parameters(parameter_file.read())

        # set uvec state if it is present
        if kratos_parameters["solver_settings"].Has("uvec"):
            kratos_parameters["solver_settings"]["uvec"]["uvec_data"]["state"] = self.__last_uvec_data["state"]

        # run calculation
        simulation = StemGeoMechanicsAnalysis(self.kratos_model, kratos_parameters)
        simulation.Run()

        # save the uvec data for the next stage if it is present
        if hasattr(simulation._GetSolver().solver, 'uvec_data'):
            self.__last_uvec_data = simulation._GetSolver().solver.uvec_data

        # make sure the simulation is deleted, else bad memory allocation may occur when serializing the kratos model
        del simulation

        # update last ran stage number
        self.__last_ran_stage_number = stage_number

        # change working directory back to original working directory
        os.chdir(cwd)

    def finalise(self):
        """
        Finalise the calculation.

        """

        for stage in self.stages:
            stage.finalise(input_folder=self.input_files_dir)

        # if more than 1 stage is run, transfer all vtk results to a shared output directory
        if len(self.stages) > 1:
            self.__transfer_vtk_files_to_main_output_directories()

    def run_calculation(self):
        """
        Run the full calculation.

        """

        # run all stages
        for stage_nr, stage in enumerate(self.stages):
            self.run_stage(stage_nr + 1)

        # finalise the calculation
        self.finalise()

    @staticmethod
    def __check_if_mesh_between_stages_is_the_same(reference_stage: Model, target_stage: Model):
        """
        Check if the mesh between stages is the same. The mesh is checked by checking if the mesh in each body model
        part is the same.

        Args:
            - reference_stage (:class:`stem.model.Model`): The reference stage.
            - target_stage (:class:`stem.model.Model`): The target stage.

        Raises:
            - Exception: If the number of body model parts are not the same between stages.
            - Exception: If the body model part names are not the same between stages.
            - Exception: If the mesh is not the same between stages in a body model part.

        """

        if len(reference_stage.body_model_parts) != len(target_stage.body_model_parts):
            raise Exception("Number of body model parts are not the same between stages")

        # check each body model part
        for ref_body, target_body in zip(reference_stage.body_model_parts, target_stage.body_model_parts):

            # check if the body model part names are the same
            if ref_body.name != target_body.name:
                raise Exception("Body model part names are not the same between stages")

            # check if the mesh is the same
            if ref_body.mesh != target_body.mesh:
                raise Exception(f"Meshes between stages in body model part: {ref_body.name} "
                                f"are not the same between stages")

    def __check_if_vtk_files_are_written_per_stage(self, path: Path, part_name, stage_index) -> bool:
        """
        Check if vtk files are written per stage. This is required as vtk files are always written to a new directory,
        in order to avoid overwriting directories from previous stages. If no vtk files are written in a stage, a
        warning message is shown.

        Args:
            - path (Path): The path to the vtk output directory.
            - part_name (str): The name of the part.
            - stage_index (int): The index of the stage.

        Returns:
            - bool: A boolean indicating if vtk files are written
        """

        if not path.exists() or not os.listdir(path):
            warnings.warn(f"No output vtk files were written for part: '{part_name}' for stage {stage_index+1}. As the "
                          f"output interval is greater than the amount of time steps in the stage.")
            return False
        return True

    def __transfer_vtk_files_to_main_output_directories(self):
        """
        Transfer vtk files from the stage output directory to the main output directory. This is required as vtk files
        are always written to a new directory, in order to avoid overwriting directories from previous stages.

        """

        # initialise dictionary to store main vtk output directories
        main_vtk_output_dirs = {}
        for i, stage in enumerate(self.stages):
            for output_settings in stage.output_settings:
                # only transfer vtk files
                if isinstance(output_settings.output_parameters, VtkOutputParameters):
                    if output_settings.part_name is None:
                        part_name = "full_model"
                    else:
                        part_name = output_settings.part_name
                    # The main output directory is the directory where the first stage writes its output
                    if i == 0:
                        if os.path.isabs(output_settings.output_dir):
                            main_vtk_output_dirs[part_name] = Path(output_settings.output_dir)
                        else:
                            main_vtk_output_dirs[part_name] = Path(self.input_files_dir) / output_settings.output_dir

                        self.__check_if_vtk_files_are_written_per_stage(main_vtk_output_dirs[part_name], part_name, i)

                    else:
                        # create main output directory if it does not exist
                        if not (os.path.isdir(main_vtk_output_dirs[part_name])):
                            os.makedirs(main_vtk_output_dirs[part_name], exist_ok=True)

                        # check if vtk files are written in the stage
                        # if the current stage is not the main stage, move the vtk files to the main output
                        # directory
                        if os.path.isabs(output_settings.output_dir):
                            stage_vtk_output_dir = Path(output_settings.output_dir)
                        else:
                            stage_vtk_output_dir = Path(self.input_files_dir) / output_settings.output_dir

                        if self.__check_if_vtk_files_are_written_per_stage(stage_vtk_output_dir, part_name, i):
                            # move all vtk files in stage vtk output dir to main vtk output dir
                            for file in os.listdir(stage_vtk_output_dir):
                                if file.endswith(".vtk"):
                                    os.rename(stage_vtk_output_dir / file, main_vtk_output_dirs[part_name] / file)

                        # remove the stage vtk output dir if it is empty
                        if not os.listdir(stage_vtk_output_dir):
                            os.rmdir(stage_vtk_output_dir)

    @staticmethod
    def __set_output_name_new_stage(new_stage: Model, stage_nr: int):
        """
        Set the output name of the new stage. The output name is set to the current output name with the
        stage number appended.

        Args:
            - new_stage (:class:`stem.model.Model`): The new stage.
            - stage_nr (int): The number of the stage.

        """

        for output_settings in new_stage.output_settings:

            # set output directory for vtk output
            if isinstance(output_settings.output_parameters, VtkOutputParameters):
                output_settings.output_dir = Path(str(output_settings.output_dir) + f"_stage_{stage_nr}")

            # set output name for gid output
            elif isinstance(output_settings.output_parameters, GiDOutputParameters):
                output_settings.output_name = f"{output_settings.output_name}_stage_{stage_nr}"

            # set output name for json output
            elif isinstance(output_settings.output_parameters, JsonOutputParameters):
                if output_settings.output_name is not None:
                    stage_identifier = f"_stage_{stage_nr}"
                    suffix = Path(output_settings.output_name).suffix

                    base_path = Path(output_settings.output_name).parent / Path(output_settings.output_name).stem
                    output_settings.output_name = str(base_path) + stage_identifier + suffix

    def __check_if_acceleration_should_be_initialised(self, previous_stage: Model, current_stage: Model):
        """
        Check if the acceleration should be initialised in the current stage. Acceleration should be initialised when
        transitioning from quasi static to dynamic.

        Args:
            - previous_stage (:class:`stem.model.Model`): The previous stage.
            - current_stage (:class:`stem.model.Model`): The current stage.
        """
        if (current_stage.project_parameters is not None and previous_stage.project_parameters is not None
                and current_stage.project_parameters.settings is not None
                and previous_stage.project_parameters.settings is not None):
            # generally acceleration should not be initialized
            current_stage.project_parameters.settings._inititalize_acceleration = False
            # acceleration should be initialized when transitioning from quasi static to dynamic
            if (previous_stage.project_parameters.settings.solution_type == SolutionType.QUASI_STATIC
                    and current_stage.project_parameters.settings.solution_type == SolutionType.DYNAMIC):
                current_stage.project_parameters.settings._inititalize_acceleration = True
