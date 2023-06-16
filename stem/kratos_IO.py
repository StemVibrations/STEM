import json
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any

import numpy as np

from stem.boundary import (
    Boundary,
    AbsorbingBoundary,
    DisplacementConstraint,
    RotationConstraint,
)
from stem.load import Load, PointLoad, MovingLoad
from stem.output import (
    Output,
    GiDOutputParameters,
    VtkOutputParameters,
    JsonOutputParameters,
    OutputParametersABC,
)
from stem.soil_material import *
from stem.structural_material import *

DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        ndim (int): The number of dimensions of the problem.

    """

    # TODO:
    #  add project folder to the attributes for relative output paths

    def __init__(self, ndim: int):
        self.ndim = ndim

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
            "Parameters": deepcopy(load.load_parameters.__dict__),
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
            "Parameters": deepcopy(load.load_parameters.__dict__),
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

        boundary_dict["Parameters"][
            "model_part_name"
        ] = f"{DOMAIN}.{boundary.part_name}"
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

        boundary_dict["Parameters"][
            "model_part_name"
        ] = f"{DOMAIN}.{boundary.part_name}"
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

        boundary_dict["Parameters"][
            "model_part_name"
        ] = f"{DOMAIN}.{boundary.part_name}"

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
                containing the absorbing boundaries of the model
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

    @staticmethod
    def __create_gid_output_dict(
        part_name: str, output_path: Path, output_parameters: GiDOutputParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in GiD
        format. To visualize the outputs, the software GiD is required.

        Args:
            part_name (str): name of the model part
            output_path (Path): output path for the GiD output
            output_parameters (GiDOutputParameters): class containing GiD output
                parameters

        Returns:
            Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """
        _output_path_gid = str(output_path).replace("\\", "/")

        parameters_dict = {
            "model_part_name": f"{DOMAIN}.{part_name}",
            "output_name": _output_path_gid,
            "postprocess_parameters": {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "WriteDeformedMeshFlag": "WriteUndeformed",
                        "WriteConditionsFlag": "WriteElementsOnly",
                        "GiDPostMode": output_parameters.gid_post_mode,
                        "MultiFileFlag": "SingleFile",
                    },
                    "file_label": output_parameters.file_label,
                    "output_control_type": output_parameters.output_control_type,
                    "output_interval": output_parameters.output_interval,
                    "body_output": output_parameters.body_output,
                    "node_output": output_parameters.node_output,
                    "skin_output": output_parameters.skin_output,
                    "plane_output": output_parameters.plane_output,
                    "nodal_results": output_parameters.nodal_results,
                    "gauss_point_results": output_parameters.gauss_point_results,
                },
                "point_data_configuration": output_parameters.point_data_configuration,
            },
        }
        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "gid_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "GiDOutputProcess",
            "Parameters": parameters_dict,
        }

        return output_dict

    @staticmethod
    def __create_vtk_output_dict(
        part_name: str, output_path: Path, output_parameters: VtkOutputParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in vtk
        format. The format can be visualized e.g., using Paraview.

        Args:
            part_name (str): name of the model part
            output_path (Path): output path for the GiD output
            output_parameters (VtkOutputParameters): class containing VTK output
                parameters

        Returns:
            Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        _output_path_vtk = str(output_path).replace("\\", "/")
        parameters_dict = {
            "model_part_name": f"{DOMAIN}.{part_name}",
            "output_path": _output_path_vtk,
            "file_format": output_parameters.file_format,
            "output_precision": output_parameters.output_precision,
            "output_control_type": output_parameters.output_control_type,
            "output_interval": output_parameters.output_interval,
            "nodal_solution_step_data_variables": output_parameters.nodal_results,
            "gauss_point_variables_in_elements": output_parameters.gauss_point_results,
        }

        # initialize load dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "vtk_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "VtkOutputProcess",
            "Parameters": parameters_dict,
        }

        return output_dict

    @staticmethod
    def __create_json_output_dict(
        part_name, output_path: Path, output_parameters: JsonOutputParameters
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output parameters to produce outputs in
        JSON format.

        Args:
            part_name (str): name of the model part
            output_path (Path): output path for the GiD output
            output_parameters (JsonOutputParameters): class containing JSON output
                parameters

        Returns:
            Dict[str, Any]: dictionary containing the output parameters in Kratos format
        """

        _output_path = deepcopy(output_path)
        if _output_path.suffix == "":
            # assume is a folder
            _output_path = _output_path.joinpath(f"{part_name}" + ".json")
        elif _output_path.suffix == "":
            _output_path = _output_path.with_suffix(".json")

        _output_path_json = (str(_output_path)).replace("\\", "/")
        # initialize output dictionary
        output_dict: Dict[str, Any] = {
            "python_module": "json_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "JsonOutputProcess",
            "Parameters": {
                "model_part_name": f"{DOMAIN}.{part_name}",
                "output_file_name": _output_path_json,
                "output_variables": output_parameters.nodal_results,
                "gauss_points_output_variables": output_parameters.gauss_point_results,
                "sub_model_part_name": output_parameters.sub_model_part_name,
            },
        }

        return output_dict

    def __create_output_dict(self, output: Output) -> Tuple[str, Dict[str, Any]]:
        """
        Creates a dictionary containing the output parameters for the desired format.
        Allowed format are GiD, VTK and JSON.

        Args:
            output (Output): output process object

        Returns:
            str: string specifying the format of the output
            Dict[str, Any]: dictionary containing the output parameters
        """
        # add output keys and parameters to dictionary based on output process type.
        if isinstance(output.output_parameters, GiDOutputParameters):
            return "gid_output", KratosIO.__create_gid_output_dict(**output.__dict__)
        elif isinstance(output.output_parameters, VtkOutputParameters):
            return "vtk_output", KratosIO.__create_vtk_output_dict(**output.__dict__)
        elif isinstance(output.output_parameters, JsonOutputParameters):
            return "json_output", KratosIO.__create_json_output_dict(**output.__dict__)
        else:
            raise NotImplementedError

    @staticmethod
    def __get_process_type(output_parameters: OutputParametersABC) -> str:
        """
        Creates a dictionary containing the output parameters to produce outputs in vtk
        format. The format can be visualized e.g., using Paraview.

        Args:
            output_parameters (Output): class containing output parameters

        Returns:
            str
        """

        if isinstance(
            output_parameters, (VtkOutputParameters, GiDOutputParameters)
        ):
            return "output_processes"
        return "processes"

    def create_output_process_dictionary(
        self, outputs: List[Output]
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the output_processes, that specifies which
        output to request Kratos and the type of output ('GiD', 'VTK', 'JSON')

        Args:
            outputs (List[Output]): list of output process objects

        Returns:
            output_dict (Dict[str, Any]): dictionary containing two other dictionary
                for output properties:
                - the first containing the "output_process" dictionary.
                - the second containing the "processes" dictionary, which includes JSON
                  outputs.
        """
        output_dict: Dict[str, Any] = {"output_processes": {}, "processes": {}}

        for output in outputs:
            output.output_parameters.validate()
            key_output, _parameters_output = self.__create_output_dict(output=output)
            key_process = KratosIO.__get_process_type(output.output_parameters)
            if key_output in output_dict[key_process].keys():
                output_dict[key_process][key_output].append(_parameters_output)
            else:
                output_dict[key_process][key_output] = [_parameters_output]
        return output_dict

    def write_project_parameters_json(self, filename):
        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()
        self.__write_constraints()
        self.__write_loads()
        # todo write Projectparameters.json
        pass

    @staticmethod
    def __create_umat_material_dict(material: SoilConstitutiveLawABC) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UMAT material. The UMAT_NAME and IS_FORTRAN_UMAT
        keys are moved to the UDSM_NAME and IS_FORTRAN_UDSM keys, as this can be recognized by Kratos.

        Args:
            material (SoilConstitutiveLawABC): soil constitutive law object containing the material parameters for UMAT

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """
        material_dict: Dict[str, Any] = deepcopy(material.__dict__)

        material_dict["UDSM_NAME"] = material_dict.pop("UMAT_NAME")
        material_dict["IS_FORTRAN_UDSM"] = material_dict.pop("IS_FORTRAN_UMAT")
        material_dict["NUMBER_OF_UMAT_PARAMETERS"] = len(
            material_dict["UMAT_PARAMETERS"]
        )

        return material_dict

    @staticmethod
    def __create_udsm_material_dict(material: SoilConstitutiveLawABC) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UDSM material. The UDSM parameters are moved to
        the UMAT_PARAMETERS key, as this can be recognized by Kratos.

        Args:
            material (SoilConstitutiveLawABC): soil constitutive law object containing the material parameters for UDSM

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """
        material_dict: Dict[str, Any] = deepcopy(material.__dict__)

        material_dict["UMAT_PARAMETERS"] = material_dict.pop("UDSM_PARAMETERS")

        return material_dict

    @staticmethod
    def __create_elastic_spring_damper_dict(
        material: StructuralParametersABC,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for an elastic spring damper material. The
        NODAL_DAMPING_COEFFICIENT and NODAL_ROTATIONAL_DAMPING_COEFFICIENT keys are moved to the NODAL_DAMPING_RATIO
        and NODAL_ROTATIONAL_DAMPING_RATIO keys, as this can be recognized by Kratos.

        Args:
            material (StructuralParametersABC): material object containing the material parameters for an elastic
                spring damper material

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        material_dict: Dict[str, Any] = deepcopy(material.__dict__)

        # Change naming of coefficient to ratio as this is the (incorrect) naming in Kratos
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop(
            "NODAL_DAMPING_COEFFICIENT"
        )
        material_dict["NODAL_ROTATIONAL_DAMPING_RATIO"] = material_dict.pop(
            "NODAL_ROTATIONAL_DAMPING_COEFFICIENT"
        )
        return material_dict

    @staticmethod
    def __create_nodal_concentrated_dict(
        material: StructuralParametersABC,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a nodal concentrated material. The
        NODAL_DAMPING_COEFFICIENT key is moved to the NODAL_DAMPING_RATIO key, as this can be recognized by Kratos.

        Args:
            material (StructuralParametersABC): material object containing the material parameters for the nodal
                concentrated material.

        Returns:
            Dict[str, Any]: dictionary containing the material parameters
        """
        material_dict: Dict[str, Any] = deepcopy(material.__dict__)

        # Change naming of coefficient to ratio as this is the (incorrect) naming in Kratos
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop(
            "NODAL_DAMPING_COEFFICIENT"
        )
        return material_dict

    def __create_linear_elastic_soil_dict(
        self, material: SoilConstitutiveLawABC
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a linear elastic soil material. The constitutive law
        is set to the correct law for the dimension of the problem.

        Args:
            material (SoilConstitutiveLawABC): soil constitutive law object containing the material parameters for a
                linear elastic soil material

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {
            "constitutive_law": {"name": ""},
            "Variables": deepcopy(material.__dict__),
        }
        if self.ndim == 2:
            material_dict["constitutive_law"][
                "name"
            ] = "GeoLinearElasticPlaneStrain2DLaw"

        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "GeoLinearElastic3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __create_umat_soil_dict(
        self, material: SoilConstitutiveLawABC
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UMAT soil material. The constitutive law is set to
        the correct law for the dimension of the problem.

        Args:
            material (SoilConstitutiveLawABC): soil constitutive law object containing the material parameters for a
                UMAT soil material

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {
            "constitutive_law": {"name": ""},
            "Variables": self.__create_umat_material_dict(material),
        }

        if self.ndim == 2:
            material_dict["constitutive_law"][
                "name"
            ] = "SmallStrainUMAT2DPlaneStrainLaw"
        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "SmallStrainUMAT3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __create_udsm_soil_dict(
        self, material: SoilConstitutiveLawABC
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UDSM soil material. The constitutive law is set to
        the correct law for the dimension of the problem.

        Args:
            material (SoilConstitutiveLawABC): soil constitutive law object containing the material parameters for a
                UDSM soil material

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {
            "constitutive_law": {"name": ""},
            "Variables": self.__create_udsm_material_dict(material),
        }

        if self.ndim == 2:
            material_dict["constitutive_law"][
                "name"
            ] = "SmallStrainUDSM2DPlaneStrainLaw"
        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "SmallStrainUDSM3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __add_soil_formulation_parameters_to_material_dict(
        self, variables_dict: Dict[str, Any], soil_parameters: SoilMaterial
    ):
        """
        Adds the soil type parameters to the material dictionary. The soil type parameters are different for one phase
        and two phase soil. The correct parameters are added to the material dictionary based on the soil type.

        Args:
            variables_dict (Dict[str, Any]): dictionary containing the material parameters
            soil_parameters (SoilMaterial): soil material object

        Returns:
            None

        """

        if isinstance(soil_parameters.soil_formulation, OnePhaseSoil):
            one_phase_soil_parameters_dict = (
                self.__create_one_phase_soil_parameters_dict(
                    soil_parameters.soil_formulation
                )
            )
            variables_dict.update(one_phase_soil_parameters_dict)
        elif isinstance(soil_parameters.soil_formulation, TwoPhaseSoil):
            two_phase_soil_parameters_dict = (
                self.__create_two_phase_soil_parameters_dict(
                    soil_parameters.soil_formulation
                )
            )
            variables_dict.update(two_phase_soil_parameters_dict)

        # Remove ndim as this is not a material parameter
        if "ndim" in variables_dict:
            variables_dict.pop("ndim")

    @staticmethod
    def __create_one_phase_soil_parameters_dict(
        soil_parameters: OnePhaseSoil,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the single phase soil parameters. For one phase soil, the permeability is set to
        near zero. Biot coefficient is added if it is not None.

        Args:
            soil_parameters (OnePhaseSoil): one phase soil parameters object.

        Returns:
            Dict[str, Any]: dictionary containing the one phase soil parameters

        """

        soil_parameters_dict = deepcopy(soil_parameters.__dict__)
        soil_parameters_dict["IGNORE_UNDRAINED"] = soil_parameters_dict.pop(
            "IS_DRAINED"
        )
        soil_parameters_dict["PERMEABILITY_XX"] = 1e-30
        soil_parameters_dict["PERMEABILITY_YY"] = 1e-30
        soil_parameters_dict["PERMEABILITY_ZZ"] = 1e-30
        soil_parameters_dict["PERMEABILITY_XY"] = 0
        soil_parameters_dict["PERMEABILITY_YZ"] = 0
        soil_parameters_dict["PERMEABILITY_ZX"] = 0

        # Create a new dictionary without None values
        soil_parameters_dict = {
            k: v for k, v in soil_parameters_dict.items() if v is not None
        }

        return soil_parameters_dict

    @staticmethod
    def __create_two_phase_soil_parameters_dict(
        two_phase_soil_parameters: TwoPhaseSoil,
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the two phase soil parameters. For two phase soil, permeability is taken into
        account and undrained behaviour is taken into account. Biot coefficient is added if it is not None.

        Args:
            two_phase_soil_parameters (TwoPhaseSoil): Two phase soil parameters object.

        Returns:
            Dict[str, Any]: dictionary containing the two phase soil parameters

        """

        two_phase_soil_parameters_dict = deepcopy(two_phase_soil_parameters.__dict__)
        two_phase_soil_parameters_dict["IGNORE_UNDRAINED"] = False

        # Create a new dictionary without None values
        two_phase_soil_parameters_dict = {
            k: v for k, v in two_phase_soil_parameters_dict.items() if v is not None
        }

        return two_phase_soil_parameters_dict

    def __create_soil_material_dict(self, material: SoilMaterial) -> Dict[str, Any]:
        """
        Creates a dictionary containing the soil material parameters. The soil material parameters are based on the
        material type. The material parameters are added to the dictionary and the retention parameters are added to
        the dictionary, lastly the fluid parameters are added to the dictionary.

        Args:
            material (SoilMaterial): Material object.

        Returns:
            Dict[str, Any]: dictionary containing the soil material parameters

        """

        soil_material_dict: Dict[str, Any] = {
            "constitutive_law": {"name": ""},
            "Variables": {},
        }

        # add material parameters to dictionary based on material type.
        if isinstance(material.constitutive_law, LinearElasticSoil):
            soil_material_dict.update(
                self.__create_linear_elastic_soil_dict(material.constitutive_law)
            )
        elif isinstance(material.constitutive_law, SmallStrainUmatLaw):
            soil_material_dict.update(
                self.__create_umat_soil_dict(material.constitutive_law)
            )
        elif isinstance(material.constitutive_law, SmallStrainUdsmLaw):
            soil_material_dict.update(
                self.__create_udsm_soil_dict(material.constitutive_law)
            )

        self.__add_soil_formulation_parameters_to_material_dict(
            soil_material_dict["Variables"], material
        )

        # get retention parameters
        retention_law = material.retention_parameters.__class__.__name__
        retention_parameters: Dict[str, Any] = deepcopy(
            material.retention_parameters.__dict__
        )

        # add retention parameters to dictionary
        soil_material_dict["Variables"]["RETENTION_LAW"] = retention_law
        soil_material_dict["Variables"].update(retention_parameters)

        # add fluid parameters to dictionary
        fluid_parameters: Dict[str, Any] = deepcopy(material.fluid_properties.__dict__)
        fluid_parameters["DENSITY_WATER"] = fluid_parameters.pop("DENSITY_FLUID")

        soil_material_dict["Variables"].update(fluid_parameters)

        return soil_material_dict

    def __create_euler_beam_dict(
        self, material_parameters: StructuralParametersABC
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the euler beam parameters

        Args:
            material: Material object containing the material parameters

        Returns:
            euler_beam_parameters_dict: Dictionary containing the euler beam parameters

        """

        material_parameters_dict = deepcopy(material_parameters.__dict__)

        # Create a new dictionary without None values
        material_parameters_dict = {
            k: v for k, v in material_parameters_dict.items() if v is not None
        }

        # remove ndim from dictionary
        if "ndim" in material_parameters_dict.keys():
            material_parameters_dict.pop("ndim")

        # initialize material dictionary
        euler_beam_parameters_dict: Dict[str, Any] = {
            "constitutive_law": {"name": ""},
            "Variables": material_parameters_dict,
        }

        # add constitutive law name to dictionary based on dimension
        if self.ndim == 2:
            euler_beam_parameters_dict["constitutive_law"][
                "name"
            ] = "LinearElastic2DBeamLaw"
        elif self.ndim == 3:
            euler_beam_parameters_dict["constitutive_law"][
                "name"
            ] = "KratosMultiphysics.StructuralMechanicsApplication.BeamConstitutiveLaw"

        return euler_beam_parameters_dict

    def __create_structural_material_dict(
        self, material: StructuralMaterial
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the structural material parameters. The structural material parameters are based
        on the material type. The material parameters are added to the dictionary.

        Args:
            material (StructuralMaterial): Material object.

        Returns:
            Dict[str, Any]: dictionary containing the structural material parameters

        """

        structural_material_dict: Dict[str, Any] = {
            "constitutive_law": {"name": ""},
            "Variables": {},
        }

        # add material parameters to dictionary based on material type.
        if isinstance(material.material_parameters, EulerBeam):
            structural_material_dict.update(
                self.__create_euler_beam_dict(material.material_parameters)
            )
        elif isinstance(material.material_parameters, ElasticSpringDamper):
            structural_material_dict[
                "Variables"
            ] = self.__create_elastic_spring_damper_dict(material.material_parameters)
        elif isinstance(material.material_parameters, NodalConcentrated):
            structural_material_dict[
                "Variables"
            ] = self.__create_nodal_concentrated_dict(material.material_parameters)

        return structural_material_dict

    def __create_material_dict(
        self, material: Union[SoilMaterial, StructuralMaterial], material_id: int
    ) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters

        Args:
            material (Union[SoilMaterial, StructuralMaterial]): material object
            material_id (int): material id

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {
            "model_part_name": material.name,
            "properties_id": material_id,
            "Material": {"constitutive_law": {"name": ""}, "Variables": {}},
            "Tables": {},
        }

        # add material parameters to dictionary based on material type.
        if isinstance(material, SoilMaterial):
            material_dict["Material"].update(self.__create_soil_material_dict(material))
        elif isinstance(material, StructuralMaterial):
            material_dict["Material"].update(
                self.__create_structural_material_dict(material)
            )

        return material_dict

    def write_material_parameters_json(
        self, materials: List[Union[SoilMaterial, StructuralMaterial]], filename: str
    ):
        """
        Writes the material parameters to a json file

        Args:
            materials (List[Union[SoilMaterial, StructuralMaterial]): list of material objects
            filename (str): filename of the output json file

        """

        materials_dict: Dict[str, Any] = {"properties": []}

        # create material dictionary for each material and assign a unique material id
        material_id = 1
        for material in materials:
            materials_dict["properties"].append(
                self.__create_material_dict(material, material_id)
            )
            material_id += 1

        # write material dictionary to json file
        json.dump(materials_dict, open(filename, "w"), indent=4)
