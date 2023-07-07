from typing import Dict, Any, List, Union


def merge(a: Dict[str, Any], b: Dict[str, Any], path: Union[List[str], Any] = None):
    """
    merges dictionary b into dictionary a. if existing keywords conflict it assumes
    they are concatenated in a list

    Args:
        a (Dict[str,Any]): first dictionary
        b (Dict[str,Any]): second dictionary
        path (List[str]): object to help navigate the deeper layers of the dictionary.
            Always place it as None

    Returns:
        a (Dict[str,Any]): updated dictionary with the additional dictionary `b`

    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            elif isinstance(a[key], list) and isinstance(b[key], list):
                a[key] = a[key] + b[key]
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def add_solver_settings_to_project_parameters(
        project_parameters:Dict[str, Any], fname:str, materials_fname:str
):
    """"""
    extra_params = {
        "problem_data": {
            "problem_name": fname,
            "start_time": 0.0,
            "end_time": 1.0,
            "echo_level": 1,
            "parallel_type": "OpenMP",
            "number_of_threads": 1
        },
        "solver_settings": {
            "solver_type": "U_Pw",
            "model_part_name": "PorousDomain",
            "domain_size": 2,
            "start_time": 0.0,
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": fname
            },
            "material_import_settings": {
                "materials_filename": materials_fname
            },
            "time_stepping": {
                "time_step": 0.01,
                "max_delta_time_factor": 1000
            },
            "buffer_size": 2,
            "echo_level": 1,
            "rebuild_level": 1,
            "clear_storage": False,
            "compute_reactions": False,
            "move_mesh_flag": False,
            "reform_dofs_at_each_step": False,
            "nodal_smoothing": False,
            "block_builder": True,
            "solution_type": "Dynamic",
            "scheme_type": "Newmark",
            "reset_displacements": True,
            "newmark_beta": 0.25,
            "newmark_gamma": 0.5,
            "newmark_theta": 0.5,
            "rayleigh_m": 0.0,
            "rayleigh_k": 0.0,
            "strategy_type": "newton_raphson",
            "convergence_criterion": "displacement_criterion",
            "displacement_relative_tolerance": 1.0E-4,
            "displacement_absolute_tolerance": 1.0E-9,
            "residual_relative_tolerance": 1.0E-4,
            "residual_absolute_tolerance": 1.0E-9,
            "water_pressure_relative_tolerance": 1.0E-4,
            "water_pressure_absolute_tolerance": 1.0E-9,
            "min_iterations": 6,
            "max_iterations": 15,
            "number_cycles": 100,
            "reduction_factor": 1,
            "increase_factor": 1,
            "desired_iterations": 4,
            "max_radius_factor": 10.0,
            "min_radius_factor": 0.1,
            "calculate_reactions": True,
            "max_line_search_iterations": 5,
            "first_alpha_value": 0.5,
            "second_alpha_value": 1.0,
            "min_alpha": 0.1,
            "max_alpha": 2.0,
            "line_search_tolerance": 0.5,
            "rotation_dofs": True,
            "linear_solver_settings": {
                "solver_type": "amgcl",
                "tolerance": 1.0e-6,
                "max_iteration": 1000,
                "scaling": False
            },
            "problem_domain_sub_model_part_list": ["Soil_drained-auto-1"],
            "processes_sub_model_part_list": ["Solid_Displacement-auto-1", "Line_Load-auto-1"],
            "body_domain_sub_model_part_list": ["Soil_drained-auto-1"]
        }
    }
    
    return merge(project_parameters, extra_params)
