from typing import List, Dict, Any, Tuple, Union, NewType

from abc import ABC

import numpy as np

from stem.boundary import (
    Boundary,
    AbsorbingBoundary,
    DisplacementConstraint,
    RotationConstraint,
)
from stem.load import Load, PointLoad, MovingLoad
from stem.output import (
    OutputProcess,
    GiDOutputParameters,
    VtkOutputParameters,
    JsonOutputParameters,
)

DOMAIN = "PorousDomain"

ConstraintType = NewType(
    "Constraint", Union[DisplacementConstraint, RotationConstraint]
)
BoundaryLoadType = NewType("BoundaryLoad", Union[AbsorbingBoundary])


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        -

    """

    def __init__(self):
        pass

    def write_mesh_to_mdpa(self, nodes, elements, filename):
        """
        Saves mesh data to mdpa file

        Args:
            nodes (np.array): node id followed by node coordinates in an array
            elements (np.array): element id followed by connectivities in an array
            filename (str): filename of mdpa file

        Returns:
            -

        """

        # todo improve this such that nodes and elements are written in the same mdpa file, where the elements are split per physical group

        np.savetxt(
            "0.nodes.mdpa", nodes, fmt=["%.f", "%.10f", "%.10f", "%.10f"], delimiter=" "
        )
        # np.savetxt('1.lines.mdpa', lines, delimiter=' ')
        # np.savetxt('2.surfaces.mdpa', surfaces, delimiter=' ')
        # np.savetxt('3.volumes.mdpa', volumes, delimiter=' ')

    def __write_problem_data(self):
        pass

    def __write_solver_settings(self):
        pass

    def __write_output_processes(self):
        pass

    def __write_input_processes(self):
        pass

    def __write_constraints(self):
        pass

    def __write_loads(self):
        pass

    # --------------------------- Loads -----------------------------------------------

    @staticmethod
    def __create_point_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the point load parameters

        Args:
            load (Load): point load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": load.load_parameters.__dict__,
        }

        load_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{load.name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    @staticmethod
    def __create_moving_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the absorbing boundary parameters

        Args:
            load (Load): moving load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # initialize load dictionary
        load_dict: Dict[str, Any] = {
            "python_module": "set_moving_load_process",
            "kratos_module": "StructuralMechanicsApplication",
            "process_name": "SetMovingLoadProcess",
            "Parameters": load.load_parameters.__dict__,
        }

        load_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{load.name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"

        return load_dict

    @staticmethod
    def __create_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the load parameters

        Args:
            load (Load): load object

        Returns:
            Dict[str, Any]: dictionary containing the load parameters
        """

        # add load parameters to dictionary based on load type.
        if isinstance(load.load_parameters, PointLoad):
            return KratosIO.__create_point_load_dict(load=load)
        elif isinstance(load.load_parameters, MovingLoad):
            return KratosIO.__create_moving_load_dict(load=load)
        else:
            raise NotImplementedError

    def create_loads_process_dictionary(self, loads: List[Load]) -> Dict[str, Any]:
        """
        Creates a dictionary containing the load_process_list (list of
        dictionaries to specify the loads for the model)

        Args:
            loads (List[Load]): list of load objects

        Returns:
            loads_dict (Dict): dictionary of a list containing the load properties
        """

        loads_dict: Dict[str, Any] = {"loads_process_list": []}

        for load in loads:
            loads_dict["loads_process_list"].append(self.__create_load_dict(load))

        return loads_dict

    # --------------------- boundary conditions ----------------------------------------

    @staticmethod
    def __create_displacement_constraint_dict(boundary: Boundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the displacement constraint parameters

        Args:
            boundary (Boundary): displacement constraint object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": boundary.boundary_parameters.__dict__,
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{boundary.name}"
        boundary_dict["Parameters"]["variable_name"] = "DISPLACEMENT"
        boundary_dict["Parameters"]["table"] = [0, 0, 0]

        return boundary_dict

    @staticmethod
    def __create_rotation_constraint_dict(boundary: Boundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the rotation constraint parameters

        Args:
            boundary (Boundary): rotation constraint object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "apply_vector_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyVectorConstraintTableProcess",
            "Parameters": boundary.boundary_parameters.__dict__,
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{boundary.name}"
        boundary_dict["Parameters"]["variable_name"] = "ROTATION"
        boundary_dict["Parameters"]["table"] = [0, 0, 0]

        return boundary_dict

    @staticmethod
    def __create_absorbing_boundary_dict(boundary: Boundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the absorbing boundary parameters

        Args:
            boundary (Boundary): absorbing boundary object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # initialize boundary dictionary
        boundary_dict: Dict[str, Any] = {
            "python_module": "set_absorbing_boundary_parameters_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "SetAbsorbingBoundaryParametersProcess",
            "Parameters": boundary.boundary_parameters.__dict__,
        }

        boundary_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{boundary.name}"

        return boundary_dict

    @staticmethod
    def __create_boundary_dict(boundary: Boundary) -> Dict[str, Any]:
        """
        Creates a dictionary containing the boundary parameters

        Args:
            boundary (Load): boundary object

        Returns:
            Dict[str, Any]: dictionary containing the boundary parameters
        """

        # add boundary parameters to dictionary based on boundary type.

        if isinstance(boundary.boundary_parameters, DisplacementConstraint):
            return KratosIO.__create_displacement_constraint_dict(boundary=boundary)
        elif isinstance(boundary.boundary_parameters, RotationConstraint):
            return KratosIO.__create_rotation_constraint_dict(boundary=boundary)
        elif isinstance(boundary.boundary_parameters, AbsorbingBoundary):
            return KratosIO.__create_absorbing_boundary_dict(boundary=boundary)
        else:
            raise NotImplementedError

    def create_dictionaries_for_boundaries(
        self, boundaries: List[Boundary]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Creates a dictionary containing the `constraint_process_list` (list of
        dictionaries to specify the constraints for the model) and a list of
        dictionaries for the absorbing boundaries to be given to `load_process_list`

        Args:
            boundaries (List[Boundary]): list of load objects

        Returns:
            constraints_dict (Dict[str, Any]): dictionary of a list containing the
                constraints acting on the model
            absorbing_boundaries_list (List[Dict[str, Any]]): dictionary of a list
            containing
                the absorbing boundaries of the model
        """

        constraints_dict: Dict[str, Any] = {"constraints_process_list": []}
        absorbing_boundaries_list: List[Dict[str, Any]] = []

        for boundary in boundaries:
            boundary_dict = self.__create_boundary_dict(boundary)
            if boundary.boundary_parameters.is_constraint():
                constraints_dict["constraints_process_list"].append(boundary_dict)
            else:
                absorbing_boundaries_list.append(boundary_dict)

        return constraints_dict, absorbing_boundaries_list

    # ------------------------- outputs ------------------------------------------------

    @staticmethod
    def __create_gid_output_dict(output: OutputProcess) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in GiD
        format. To visualize the outputs, the software GiD is required.

        Args:
            output (OutputProcess): output process object

        Returns:
            Dict[str, Any]: dictionary containing the output parameters
        """

        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "gid_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "GiDOutputProcess",
            "Parameters": output.output_parameters.assemble_parameters(),
        }

        output_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{output.part_name}"
        output_dict["Parameters"]["output_name"] = f"{output.output_name}"

        return output_dict

    @staticmethod
    def __create_vtk_output_dict(output: OutputProcess) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in vtk
        format. The format can be visualized e.g., using Paraview.

        Args:
            output (OutputProcess): output process object

        Returns:
            Dict[str, Any]: dictionary containing the output parameters
        """

        # initialize load dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "vtk_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "VtkOutputProcess",
            "Parameters": output.output_parameters.assemble_parameters(),
        }

        output_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{output.part_name}"
        output_dict["Parameters"]["output_path"] = f"{output.output_name}"
        return output_dict

    @staticmethod
    def __create_json_output_dict(output: OutputProcess) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in
        JSON format.

        Args:
            output (OutputProcess): output process object

        Returns:
            output_dict (Dict[str, Any]): dictionary containing the output parameters
        """

        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "json_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "JsonOutputProcess",
            "Parameters": output.output_parameters.assemble_parameters(),
        }

        output_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{output.part_name}"
        output_dict["Parameters"]["output_file_name"] = f"{output.output_name}"

        return output_dict

    def __create_output_dict(self, output: OutputProcess) -> Tuple[str, Dict[str, Any]]:
        """
        Creates a dictionary containing the output parameters for the desired format.
        Allowed format are GiD, VTK and JSON.

        Args:
            output (OutputProcess): output process object

        Returns:
            str: string specifying the format of the output
            Dict[str, Any]: dictionary containing the output parameters
        """
        # add output keys and parameters to dictionary based on output process type.
        # TODO: check that keys are correct for VTK and JSON
        if isinstance(output.output_parameters, GiDOutputParameters):
            return "gid_output", KratosIO.__create_gid_output_dict(output=output)
        elif isinstance(output.output_parameters, VtkOutputParameters):
            return "vtk_output", KratosIO.__create_vtk_output_dict(output=output)
        elif isinstance(output.output_parameters, JsonOutputParameters):
            return "json_output", KratosIO.__create_json_output_dict(output=output)
        else:
            raise NotImplementedError

    def create_output_process_dictionary(
        self, outputs: List[OutputProcess]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Creates a dictionary containing the output_processes, that specifies
        which output to request Kratos and the type of output ('GiD', 'VTK',
        'JSON')

        Args:
            outputs (List[OutputProcess]): list of output process objects

        Returns:
            Tuple[Dict[str, Any]]: Tuple of two dictionaries containing the output
                properties.
                - the first containing the "output_process" dictionary. This is a
                  separate dictionary.
                - the second containing the "json_output" dictionary. This is to be
                  placed under "processes".
        """
        output_dict: Dict[str, Any] = {"output_processes": {}}
        json_dict: Dict[str, Any] = {"json_output": []}

        for output in outputs:
            output.output_parameters.validate()
            key_output, _parameters_output = self.__create_output_dict(output=output)
            if isinstance(output.output_parameters, GiDOutputParameters) or isinstance(
                output.output_parameters, VtkOutputParameters
            ):
                output_dict["output_processes"][key_output] = [_parameters_output]
            elif isinstance(output.output_parameters, JsonOutputParameters):
                json_dict[key_output] = [_parameters_output]

        return output_dict, json_dict

    def write_project_parameters_json(self, filename):
        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()
        self.__write_constraints()
        self.__write_loads()
        # todo write Projectparameters.json
        pass

    def write_material_parameters_json(self, materials, filename):
        pass
