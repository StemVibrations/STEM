import os
from copy import deepcopy

import KratosMultiphysics
from KratosMultiphysics.StemApplication.geomechanics_analysis import StemGeoMechanicsAnalysis

from typing import List, Dict
from stem.model import Model
from stem.IO.kratos_io import KratosIO


class Stem:
    """
    Class containing the main calculation.

    Attributes:
        - input_files_dir (str): The directory where the input files are to be written.
        - kratos_io (:class:`stem.IO.kratos_io.KratosIO`): The Kratos IO object.
        - kratos_model (:class:`KratosMultiphysics.Model`): The Kratos model.
        - __stages (List[:class:`stem.model.Model`]): The calculation stages.
        - __stage_settings_file_names (Dict[int, str]): The file names of the project parameters files for each stage.
        - __last_ran_stage_number (int): The number of the last ran stage.

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
        self.kratos_stages = []

        self.__stages: List[Model] = [initial_stage]
        self.__stage_settings_file_names: Dict[int, str] = {}
        self.__last_ran_stage_number: int = 0

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

        # set the time integration settings of the new stage
        new_stage.project_parameters.settings.time_integration.start_time = (
            self.__stages[-1].project_parameters.settings.time_integration.end_time)
        new_stage.project_parameters.settings.time_integration.end_time = (
                new_stage.project_parameters.settings.time_integration.start_time + stage_duration)
        new_stage.project_parameters.settings.time_integration.delta_time = (
            delta_time)

        return new_stage

    def add_calculation_stage(self, stage: Model):
        """
        Add a calculation stage to the calculation. The geometry and the mesh of the new stage are regenerated.

        Args:
            - stage (:class:`stem.model.Model`): The model of the stage to be added.


        """
        self.__stages.append(stage)

        # add the geo data to gmsh
        stage.gmsh_io.generate_geo_from_geo_data()

        # post setup and generate mesh
        stage.post_setup()
        stage.generate_mesh()

        # check if the mesh is the same in the new stage
        self.__check_if_mesh_between_stages_is_the_same(self.__stages[-2], stage)

    def validate_stages(self):
        """
        Validate the stages of the calculation. Currently stages are not validated, but this method is reserved for
        when multi-stage calculations are implemented. In this case, the mesh in all stages should be the same.
        Furthermore, time should be continuous between stages.

        Raises:
            - NotImplementedError: Validation of stages is not implemented yet.

        """

        NotImplementedError("Validation of stages is not implemented yet.")

    def write_all_input_files(self):
        """
        Write all input files for the calculation.

        """

        for stage_nr, stage in enumerate(self.stages):
            if stage.project_parameters is not None:
                mesh_name = stage.project_parameters.problem_name + f"_stage_{stage_nr+1}.mdpa"
                project_settings_file_name = f"ProjectParameters_stage_{stage_nr+1}.json"
                material_settings_file_name = f"MaterialParameters_stage_{stage_nr+1}.json"
                self.kratos_io.project_folder = self.input_files_dir
                self.kratos_io.write_input_files_for_kratos(stage, mesh_name,
                                                            materials_file_name=material_settings_file_name,
                                                            project_file_name=project_settings_file_name)

                self.__stage_settings_file_names[stage_nr+1] = project_settings_file_name

    def run_stage(self, stage_number: int, time_step):
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

        # if stage_number >1:
        #     mdpa_text = "Begin Properties 1\n" + "End Properties"
        #
        #     with open("test_multi_stage_block_stage_2.mdpa", "w") as mdpa_file:
        #         mdpa_file.write(mdpa_text)

        # run calculation
        simulation = StemGeoMechanicsAnalysis(self.kratos_model, kratos_parameters)
        simulation.Initialize()
        simulation._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP] = time_step
        simulation.RunSolutionLoop()
        simulation.Finalize()

        time_step = simulation._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP]

        # simulation.Run()

        # update last ran stage number
        self.__last_ran_stage_number = stage_number

        self.kratos_stages.append(simulation)

        # change working directory back to original working directory
        os.chdir(cwd)

        return time_step

    def run_calculation(self):
        """
        Run the full calculation.

        """
        timestep = 0
        for stage_nr, stage in enumerate(self.stages):
            timestep = self.run_stage(stage_nr + 1, time_step=timestep)

        # time_step = 0
        # cwd = os.getcwd()
        # os.chdir(self.input_files_dir)
        # for stage in self.kratos_stages:
        #     stage.Initialize()
        #     stage._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP] = time_step
        #     stage.RunSolutionLoop()
        #     stage.Finalize()
        #
        #     time_step = stage._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP]
        #
        #
        #     # stage._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP] += 1
        #
        # os.chdir(cwd)

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
