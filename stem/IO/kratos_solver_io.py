from typing import Dict, Any, List, Optional
from copy import deepcopy
import math

from stem.solver import *
from stem.model_part import ModelPart, BodyModelPart
from stem.load import UvecLoad


class KratosSolverIO:
    """
    Class containing methods to write Kratos solver settings

    Attributes:
        - ndim (int): The number of dimensions of the problem (2 or 3).
        - domain (str): The name of the Kratos domain.
    """

    def __init__(self, ndim: int, domain: str):
        """
        Class to read and write Kratos solver settings.

        Args:
            - ndim (int): The number of dimensions of the problem (2 or 3).
            - domain (str): The name of the Kratos domain.
        """
        self.ndim: int = ndim
        self.domain: str = domain

    @staticmethod
    def __create_problem_data_dictionary(problem_data: Problem):
        """
        Creates a dictionary containing the problem data

        Args:
            - problem_data (:class:`stem.solver.Problem`): The problem data

        Returns:
            - Dict[str, Any]: dictionary containing the problem data
        """
        problem_data_dict: Dict[str, Any] = {"problem_name": problem_data.problem_name,
                                             "start_time": problem_data.settings.time_integration.start_time,
                                             "end_time": problem_data.settings.time_integration.end_time,
                                             "echo_level": 1,
                                             "parallel_type": "OpenMP",
                                             "number_of_threads": problem_data.number_of_threads}

        return problem_data_dict

    @staticmethod
    def __create_scheme_dict(scheme: SchemeABC):
        """
        Creates a dictionary containing the scheme parameters

        Args:
            - scheme (:class:`stem.solver.SchemeABC`): The scheme object

        Returns:
            - Dict[str, Any]: dictionary containing the scheme parameters
        """

        scheme_dict: Dict[str, Any] = {"scheme_type": scheme.scheme_type}

        scheme_dict.update(deepcopy(scheme.__dict__))
        return scheme_dict

    @staticmethod
    def __create_strategy_dict(strategy_parameters: StrategyTypeABC):
        """
        Creates a dictionary containing the strategy parameters

        Args:
            - strategy_parameters (:class:`stem.solver.StrategyTypeABC`): The strategy parameters object

        Returns:
            - Dict[str, Any]: dictionary containing the strategy parameters
        """

        strategy_dict: Dict[str, Any] = {"strategy_type": strategy_parameters.strategy_type}
        strategy_dict.update(deepcopy(strategy_parameters.__dict__))
        return strategy_dict

    @staticmethod
    def __create_convergence_criterion_dict(convergence_criterion: ConvergenceCriteriaABC):
        """
        Creates a dictionary containing the convergence criterion parameters

        Args:
            - convergence_criterion (:class:`stem.solver.ConvergenceCriteriaABC`): The convergence criterion object

        Returns:
            - Dict[str, Any]: dictionary containing the convergence criterion parameters
        """

        convergence_criterion_dict: Dict[str, Any] = {
            "convergence_criterion": convergence_criterion.convergence_criterion}
        convergence_criterion_dict.update(deepcopy(convergence_criterion.__dict__))
        return convergence_criterion_dict

    @staticmethod
    def __create_linear_solver_dict(linear_solver: LinearSolverSettingsABC):
        """
        Creates a dictionary containing the linear solver parameters

        Args:
            - linear_solver (:class:`stem.solver.LinearSolverSettingsABC`): The linear solver object

        Returns:
            - Dict[str, Any]: dictionary containing the linear solver parameters
        """

        linear_solver_dict: Dict[str, Any] = {"linear_solver_settings": {"solver_type": linear_solver.solver_type}}
        linear_solver_dict["linear_solver_settings"].update(deepcopy(linear_solver.__dict__))
        return linear_solver_dict

    @staticmethod
    def __create_model_part_name_dict(model_parts: List[ModelPart]):
        """
        Creates a dictionary containing the model part names

        Args:
            - model_parts (List[:class:`stem.model_part.ModelPart`]): The list of model parts

        Returns:
            - Dict[str, Any]: dictionary containing the model part names
        """
        model_parts_dict: Dict[str, Any] = {"problem_domain_sub_model_part_list": [],
                                            "processes_sub_model_part_list": [],
                                            "body_domain_sub_model_part_list": []}

        # loop over model parts and add body model parts and other model parts to the corresponding lists
        for model_part in model_parts:
            if isinstance(model_part, BodyModelPart):
                model_parts_dict["problem_domain_sub_model_part_list"].append(model_part.name)
                model_parts_dict["body_domain_sub_model_part_list"].append(model_part.name)
            else:
                model_parts_dict["processes_sub_model_part_list"].append(model_part.name)

        return model_parts_dict

    @staticmethod
    def __set_solution_type(solver_settings: SolverSettings):
        """
        Sets the solution type name

        Args:
            - solver_settings (:class:`stem.solver.SolverSettings`): The solver settings

        Returns:
            - str: The solution type name
        """
        if solver_settings.solution_type == SolutionType.QUASI_STATIC:
            if solver_settings.stress_initialisation_type == StressInitialisationType.K0_PROCEDURE:
                return "k0_procedure"
            else:
                return "quasi_static"
        elif solver_settings.solution_type == SolutionType.DYNAMIC:
            return "dynamic"

    @staticmethod
    def __set_analysis_type(analysis_type: AnalysisType):
        """
        Sets the analysis type name

        Args:
            - analysis_type (:class:`stem.solver.AnalysisType`): The analysis type

        Returns:
            - str: The analysis type name
        """
        if analysis_type == AnalysisType.MECHANICAL_GROUNDWATER_FLOW:
            return "U_Pw"
        elif analysis_type == AnalysisType.MECHANICAL:
            return "U_Pw"
        elif analysis_type == AnalysisType.GROUNDWATER_FLOW:
            return "Pw"

    @staticmethod
    def __create_uvec_parameters_dictionary(solver_settings_dict: Dict[str, Any], model_parts: List[ModelPart]) \
            -> Optional[Dict[str, Any]]:
        """
        Creates a dictionary containing the uvec parameters

        Args:
            - solver_settings_dict (Dict[str, Any]): The solver settings dictionary
            - model_parts (List[:class:`stem.model_part.ModelPart`]): The list of model parts

        Returns:
            - Optional[Dict[str, Any]]: dictionary containing the UVEC parameters or None if no UVEC load is found

        """

        # loop over model parts to check if there is a UVEC load
        for model_part in model_parts:

            # if there is a UVEC load, create the UVEC parameters dictionary
            if isinstance(model_part.parameters, UvecLoad):

                # set strategy type to newton_raphson_with_uvec
                solver_settings_dict["strategy_type"] = "newton_raphson_with_uvec"

                uvec_dict: Dict[str, Any] = {"uvec_path": model_part.parameters.uvec_file,
                                             "uvec_method": model_part.parameters.uvec_function_name,
                                             "uvec_model_part": model_part.name,
                                             "uvec_data": {"dt": solver_settings_dict["time_stepping"]["time_step"],
                                                           "u": {},
                                                           "theta": {},
                                                           "loads": {},
                                                           "parameters": model_part.parameters.uvec_parameters,
                                                           "state": model_part.parameters.uvec_state_variables}}

                # return the UVEC parameters dictionary
                return uvec_dict

        # return None if no UVEC load is found
        return None

    def __create_solver_settings_dictionary(self, solver_settings: SolverSettings, mesh_file_name: str,
                                            materials_file_name: str, model_parts: List[ModelPart]):
        """
        Creates a dictionary containing the solver settings

        Args:
            - solver_settings (:class:`stem.solver.SolverSettings`): The solver settings
            - mesh_file_name (str): The name of the mesh file
            - materials_file_name (str): The name of the materials file
            - model_parts (List[:class:`stem.model_part.ModelPart`]): The list of model parts

        Returns:
            - Dict[str, Any]: dictionary containing the solver settings
        """
        solver_settings_dict: Dict[str, Any] = {"solver_type": self.__set_analysis_type(solver_settings.analysis_type),
                                                "model_part_name": self.domain,
                                                "domain_size": self.ndim,
                                                "model_import_settings": {
                                                    "input_type": "mdpa",
                                                    "input_filename": mesh_file_name},
                                                "material_import_settings": {
                                                    "materials_filename": materials_file_name},
                                                "time_stepping": {
                                                    "time_step": solver_settings.time_integration.delta_time,
                                                    "max_delta_time_factor": solver_settings.time_integration.max_delta_time_factor},
                                                "reduction_factor": solver_settings.time_integration.reduction_factor,
                                                "increase_factor": solver_settings.time_integration.increase_factor,
                                                "buffer_size": 2,
                                                "echo_level": solver_settings.echo_level,
                                                "clear_storage": False,
                                                "compute_reactions": False,
                                                "move_mesh_flag": False,
                                                "reform_dofs_at_each_step": False,
                                                "nodal_smoothing": solver_settings.calculate_stresses_on_nodes,
                                                "block_builder": True,
                                                "rebuild_level": 0 if solver_settings.is_stiffness_matrix_constant
                                                                   else 2,
                                                "prebuild_dynamics": solver_settings.are_mass_and_damping_constant,
                                                "solution_type": self.__set_solution_type(solver_settings),
                                                "rayleigh_m": solver_settings.rayleigh_m if solver_settings.rayleigh_m
                                                                                            is not None else 0,
                                                "rayleigh_k": solver_settings.rayleigh_k if solver_settings.rayleigh_k
                                                                                            is not None else 0,
                                                "calculate_reactions": True,
                                                "rotation_dofs": True,
                                                "reset_displacements": solver_settings.reset_displacements
                                                }

        # Add the settings of the scheme, strategy, convergence criterion and linear solver
        solver_settings_dict.update(self.__create_scheme_dict(solver_settings.scheme))
        solver_settings_dict.update(self.__create_strategy_dict(solver_settings.strategy_type))
        solver_settings_dict.update(self.__create_convergence_criterion_dict(solver_settings.convergence_criteria))
        solver_settings_dict.update(self.__create_linear_solver_dict(solver_settings.linear_solver_settings))

        # Add the model part names
        solver_settings_dict.update(self.__create_model_part_name_dict(model_parts))

        # set number of cycles to 1 if no time step reduction is used
        if math.isclose(solver_settings_dict["reduction_factor"], 1.0) and "number_cycles" in solver_settings_dict:
            solver_settings_dict["number_cycles"] = 1

        # add  Uvec parameters if present
        uvec_settings = self.__create_uvec_parameters_dictionary(solver_settings_dict, model_parts)
        if uvec_settings is not None:
            solver_settings_dict.update({"uvec": uvec_settings})

        return solver_settings_dict

    def create_settings_dictionary(self, problem_data: Problem, mesh_file_name: str, materials_file_name: str,
                                   model_parts: List[ModelPart]) -> Dict[str, Any]:
        """
        Creates a dictionary containing the solver settings

        Args:
            - problem_data (:class:`stem.solver.Problem`): The problem data
            - mesh_file_name (str): The name of the mesh file
            - materials_file_name (str): The name of the materials file
            - model_parts (List[:class:`stem.model_part.ModelPart`]): The list of model parts

        Returns:
            - Dict[str, Any]: dictionary containing the problem data and the solver settings
        """

        settings_dict: Dict[str, Any] = {"problem_data": self.__create_problem_data_dictionary(problem_data),
                                         "solver_settings":
                                             self.__create_solver_settings_dictionary(problem_data.settings,
                                                                                      mesh_file_name,
                                                                                      materials_file_name,
                                                                                      model_parts)}

        return settings_dict
