import os

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

    def add_calculation_stage(self, stage: Model):
        """
        Add a calculation stage to the calculation. Currently only one stage is supported. This method is reserved for
        when multi-stage calculations are implemented. In this case, when adding a stage, a deepcopy of the model is to
        be made, where the nodes and elements is not to be written again.

        Args:
            - stage (:class:`stem.model.Model`): The model of the stage to be added.

        Raises:
            - NotImplementedError: Adding calculation stages is not implemented yet.

        """
        NotImplementedError("Adding calculation stages is not implemented yet. Currently only one stage is supported.")

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

        # run calculation
        simulation = StemGeoMechanicsAnalysis(self.kratos_model, kratos_parameters)
        simulation.Run()

        # update last ran stage number
        self.__last_ran_stage_number = stage_number

        # change working directory back to original working directory
        os.chdir(cwd)

    def run_calculation(self):
        """
        Run the full calculation.

        """

        for stage_nr, stage in enumerate(self.stages):
            self.run_stage(stage_nr + 1)


