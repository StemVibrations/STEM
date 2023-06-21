import json
from typing import Dict, Union, Any, List
from copy import deepcopy

from stem.solver import *



class KratosSolverIO:
    def __init__(self, ndim: int, domain:str):
        """
        Class to read and write Kratos solver settings.

        Args:
            ndim (int): The number of dimensions of the problem (2 or 3).
        """
        self.ndim = ndim
        self.domain = domain

    def __create_problem_data_dictionary(self, problem_data: Problem):
        """
        Creates a dictionary containing the problem data

        Args:
            problem_data (Problem): The problem data

        Returns:
            Dict[str, Any]: dictionary containing the problem data
        """
        problem_data_dict = {"problem_name": problem_data.problem_name,
                             "start_time": problem_data.settings.time_integration.start_time,
                             "end_time": problem_data.settings.time_integration.end_time,
                             "echo_level": problem_data.echo_level,
                             "parallel_type": "OpenMP",
                             "number_of_threads": problem_data.number_of_threads}

        return problem_data_dict

    def __create_solver_settings_dictionary(self, solver_settings: SolverSettings, problem_name:str):
        """
        Creates a dictionary containing the solver settings

        Returns:
            Dict[str, Any]: dictionary containing the solver settings
        """
        solver_settings_dict = {"solver_type": "U_Pw",
                                "model_part_name": self.domain,
                                "domain_size": self.ndim,
                                "start_time": solver_settings.time_integration.start_time,
                                "model_import_settings": {
                                    "input_type": "mdpa",
                                    "input_filename": problem_name},
                                "time_stepping": {
                                    "time_step": solver_settings.time_integration.delta_time,
                                    "max_delta_time": solver_settings.time_integration.max_delta_time_factor}


                                }


        return solver_settings_dict


    def create_settings_dictionary(self, problem_data: Problem):
        """
        Creates a dictionary containing the solver settings

        Returns:
            Dict[str, Any]: dictionary containing the solver settings
        """
        pass


