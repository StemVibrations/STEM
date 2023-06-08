import json
from typing import Dict, List, Any, Union

import numpy as np

from stem.material import (Material,
                           SoilParametersABC,
                           StructuralParametersABC,
                           DrainedSoil,
                           UndrainedSoil,
                           TwoPhaseSoil2D,
                           TwoPhaseSoil3D,
                           LinearElasticSoil,
                           SmallStrainUdsmLaw,
                           SmallStrainUmatLaw,
                           EulerBeam2D,
                           EulerBeam3D,
                           ElasticSpringDamper,
                           NodalConcentrated)

from stem.load import Load, PointLoad, MovingLoad


DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        ndim (int): The number of dimensions of the problem.

    """

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
            "Parameters": load.load_parameters.__dict__,
        }

        load_dict["Parameters"]["model_part_name"] = f"{DOMAIN}.{load.name}"
        load_dict["Parameters"]["variable_name"] = "POINT_LOAD"
        load_dict["Parameters"]["table"] = [0, 0, 0]

        return load_dict

    @staticmethod
    def __create_moving_load_dict(load: Load) -> Dict[str, Any]:
        """
        Creates a dictionary containing the moving load parameters

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

    def write_project_parameters_json(self, filename):

        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()
        self.__write_constraints()
        self.__write_loads()
        # todo write Projectparameters.json
        pass


    def __create_umat_material_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UMAT material. The UMAT_NAME and IS_FORTRAN_UMAT
        keys are moved to the UDSM_NAME and IS_FORTRAN_UDSM keys, as this can be recognized by Kratos.

        Args:
            material (Material): material object

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """
        material_dict: Dict[str, Any] = material.material_parameters.__dict__

        material_dict["UDSM_NAME"] = material_dict.pop("UMAT_NAME")
        material_dict["IS_FORTRAN_UDSM"] = material_dict.pop("IS_FORTRAN_UMAT")
        material_dict["NUMBER_OF_UMAT_PARAMETERS"] = len(material_dict["UMAT_PARAMETERS"])

        return material_dict

    def __create_udsm_material_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UDSM material. The UDSM parameters are moved to
        the UMAT_PARAMETERS key, as this can be recognized by Kratos.

        Args:
            material (Material): material object

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """
        material_dict: Dict[str, Any] = material.material_parameters.__dict__

        material_dict["UMAT_PARAMETERS"] = material_dict.pop("UDSM_PARAMETERS")

        return material_dict

    def __create_elastic_spring_damper_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for an elastic spring damper material. The
        NODAL_DAMPING_COEFFICIENT and NODAL_ROTATIONAL_DAMPING_COEFFICIENT keys are moved to the NODAL_DAMPING_RATIO
        and NODAL_ROTATIONAL_DAMPING_RATIO keys, as this can be recognized by Kratos.

        Args:
            material (Material): material object

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        material_dict: Dict[str, Any] = material.material_parameters.__dict__

        # Change naming of coefficient to ratio as this is the (incorrect) naming in Kratos
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop("NODAL_DAMPING_COEFFICIENT")
        material_dict["NODAL_ROTATIONAL_DAMPING_RATIO"] = material_dict.pop("NODAL_ROTATIONAL_DAMPING_COEFFICIENT")
        return material_dict

    def __create_nodal_concentrated_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a nodal concentrated material. The
        NODAL_DAMPING_COEFFICIENT key is moved to the NODAL_DAMPING_RATIO key, as this can be recognized by Kratos.

        Args:
            material (Material): material object containing the material parameters for the nodal concentrated material.

        Returns:
            Dict[str, Any]: dictionary containing the material parameters
        """
        material_dict: Dict[str, Any] = material.material_parameters.__dict__

        # Change naming of coefficient to ratio as this is the (incorrect) naming in Kratos
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop("NODAL_DAMPING_COEFFICIENT")
        return material_dict

    def __create_linear_elastic_soil_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a linear elastic soil material. The constitutive law
        is set to the correct law for the dimension of the problem.

        Args:
            material (Material): material object containing the material parameters for the linear elastic soil

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                         "Variables": material.material_parameters.__dict__}
        if self.ndim == 2:
            material_dict["constitutive_law"]["name"] = "GeoLinearElasticPlaneStrain2DLaw"

        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "GeoLinearElastic3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __create_umat_soil_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UMAT soil material. The constitutive law is set to
        the correct law for the dimension of the problem.

        Args:
            material (Material): material object containing the material parameters for the UMAT soil

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                         "Variables": self.__create_umat_material_dict(material)}

        if self.ndim == 2:
            material_dict["constitutive_law"]["name"] = "SmallStrainUMAT2DPlaneStrainLaw"
        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "SmallStrainUMAT3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __create_udsm_soil_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a UDSM soil material. The constitutive law is set to
        the correct law for the dimension of the problem.

        Args:
            material (Material): material object containing the material parameters for the UDSM soil

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                         "Variables": self.__create_udsm_material_dict(material)}

        if self.ndim == 2:
            material_dict["constitutive_law"]["name"] = "SmallStrainUDSM2DPlaneStrainLaw"
        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "SmallStrainUDSM3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __add_soil_type_parameters_to_material_dict(self, variables_dict: Dict[str, Any],
                                                    soil_type_parameters: SoilParametersABC):
        """
        Adds the soil type parameters to the material dictionary. The soil type parameters are different for drained
        , undrained and two phase soil. The correct parameters are added to the material dictionary based on the soil
        type.

        Args:
            variables_dict (Dict[str, Any]): dictionary containing the material parameters
            soil_type_parameters (SoilParametersABC): soil parameters object containing the soil type

        Returns:
            None

        """

        if isinstance(soil_type_parameters.SOIL_TYPE, DrainedSoil):
            drained_soil_parameters_dict = self.__create_drained_soil_parameters_dict(
                soil_type_parameters.SOIL_TYPE)
            variables_dict.update(drained_soil_parameters_dict)
        elif isinstance(soil_type_parameters.SOIL_TYPE, UndrainedSoil):
            undrained_soil_parameters_dict = self.__create_undrained_soil_parameters_dict(
                soil_type_parameters.SOIL_TYPE)
            variables_dict.update(undrained_soil_parameters_dict)
        elif isinstance(soil_type_parameters.SOIL_TYPE, TwoPhaseSoil2D) or isinstance(
                soil_type_parameters.SOIL_TYPE, TwoPhaseSoil3D):
            two_phase_soil_parameters_dict = self.__create_two_phase_soil_parameters_dict(
                soil_type_parameters.SOIL_TYPE)
            variables_dict.update(two_phase_soil_parameters_dict)

        variables_dict.pop("SOIL_TYPE")

    def __create_drained_soil_parameters_dict(self, drained_soil_parameters: DrainedSoil) -> Dict[str, Any]:
        """
        Creates a dictionary containing the drained soil parameters. For drained soil, the permeability is set to near
        zero and undrained behaviour is ignored. Biot coefficient is added if it is not None.

        Args:
            drained_soil_parameters (DrainedSoil): Drained soil parameters object.

        Returns:
            Dict[str, Any]: dictionary containing the drained soil parameters

        """

        drained_soil_parameters_dict = drained_soil_parameters.__dict__
        drained_soil_parameters_dict["IGNORE_UNDRAINED"] = True
        drained_soil_parameters_dict["PERMEABILITY_XX"] = 1e-30
        drained_soil_parameters_dict["PERMEABILITY_YY"] = 1e-30
        drained_soil_parameters_dict["PERMEABILITY_ZZ"] = 1e-30
        drained_soil_parameters_dict["PERMEABILITY_XY"] = 0
        drained_soil_parameters_dict["PERMEABILITY_YZ"] = 0
        drained_soil_parameters_dict["PERMEABILITY_ZX"] = 0

        if drained_soil_parameters_dict["BIOT_COEFFICIENT"] is None:
            drained_soil_parameters_dict.pop("BIOT_COEFFICIENT")

        return drained_soil_parameters_dict

    def __create_undrained_soil_parameters_dict(self, undrained_soil_parameters: UndrainedSoil) -> Dict[str, Any]:
        """
        Creates a dictionary containing the undrained soil parameters. For undrained soil, the permeability is set to
        near zero and undrained behaviour is taken into account. Biot coefficient is added if it is not None.

        Args:
            undrained_soil_parameters (UndrainedSoil): Undrained soil parameters object.

        Returns:
            Dict[str, Any]: dictionary containing the undrained soil parameters

        """

        undrained_soil_parameters_dict = undrained_soil_parameters.__dict__
        undrained_soil_parameters_dict["IGNORE_UNDRAINED"] = False
        undrained_soil_parameters_dict["PERMEABILITY_XX"] = 1e-30
        undrained_soil_parameters_dict["PERMEABILITY_YY"] = 1e-30
        undrained_soil_parameters_dict["PERMEABILITY_ZZ"] = 1e-30
        undrained_soil_parameters_dict["PERMEABILITY_XY"] = 0
        undrained_soil_parameters_dict["PERMEABILITY_YZ"] = 0
        undrained_soil_parameters_dict["PERMEABILITY_ZX"] = 0

        if undrained_soil_parameters_dict["BIOT_COEFFICIENT"] is None:
            undrained_soil_parameters_dict.pop("BIOT_COEFFICIENT")

        return undrained_soil_parameters_dict

    def __create_two_phase_soil_parameters_dict(self, two_phase_soil_parameters: Union[TwoPhaseSoil3D, TwoPhaseSoil2D]) -> Dict[str, Any]:
        """
        Creates a dictionary containing the two phase soil parameters. For two phase soil, permeability is taken into
        account and undrained behaviour is taken into account. Biot coefficient is added if it is not None.

        Args:
            two_phase_soil_parameters (Union[TwoPhaseSoil3D, TwoPhaseSoil2D]): Two phase soil parameters object.

        Returns:
            Dict[str, Any]: dictionary containing the two phase soil parameters

        """

        two_phase_soil_parameters_dict = two_phase_soil_parameters.__dict__
        two_phase_soil_parameters_dict["IGNORE_UNDRAINED"] = False

        if two_phase_soil_parameters_dict["BIOT_COEFFICIENT"] is None:
            two_phase_soil_parameters_dict.pop("BIOT_COEFFICIENT")

        return two_phase_soil_parameters_dict

    def __create_soil_material_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the soil material parameters. The soil material parameters are based on the
        material type. The material parameters are added to the dictionary and the retention parameters are added to
        the dictionary, lastly the fluid parameters are added to the dictionary.

        Args:
            material (Material): Material object.

        Returns:
            Dict[str, Any]: dictionary containing the soil material parameters

        """

        soil_material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                              "Variables": {}}

        # add material parameters to dictionary based on material type.
        if isinstance(material.material_parameters, LinearElasticSoil):
            soil_material_dict.update(self.__create_linear_elastic_soil_dict(material))
        elif isinstance(material.material_parameters, SmallStrainUmatLaw):
            soil_material_dict.update(self.__create_umat_soil_dict(material))
        elif isinstance(material.material_parameters, SmallStrainUdsmLaw):
            soil_material_dict.update(self.__create_udsm_soil_dict(material))

        self.__add_soil_type_parameters_to_material_dict(soil_material_dict["Variables"],
                                                         material.material_parameters)

        # get retention parameters
        retention_law = material.material_parameters.RETENTION_PARAMETERS.__class__.__name__
        retention_parameters: Dict[str, Any] = material.material_parameters.RETENTION_PARAMETERS.__dict__

        soil_material_dict["Variables"].pop("RETENTION_PARAMETERS")

        # add retention parameters to dictionary
        soil_material_dict["Variables"]["RETENTION_LAW"] = retention_law
        soil_material_dict["Variables"].update(retention_parameters)

        # add fluid parameters to dictionary
        fluid_parameters: Dict[str, Any] = material.fluid_properties.__dict__
        soil_material_dict["Variables"].update(fluid_parameters)

        return soil_material_dict

    def __create_euler_beam_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the euler beam parameters

        Args:
            material: Material object containing the material parameters

        Returns:
            euler_beam_parameters_dict: Dictionary containing the euler beam parameters

        """

        # initialize material dictionary
        euler_beam_parameters_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                                      "Variables": material.material_parameters.__dict__}

        if self.ndim == 2:
            euler_beam_parameters_dict["constitutive_law"]["name"] = "LinearElastic2DBeamLaw"
        elif self.ndim == 3:
            euler_beam_parameters_dict["constitutive_law"]["name"] = \
                "KratosMultiphysics.StructuralMechanicsApplication.BeamConstitutiveLaw"
        else:
            raise ValueError("Dimension not supported")

        return euler_beam_parameters_dict

    def __create_structural_material_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the structural material parameters. The structural material parameters are based
        on the material type. The material parameters are added to the dictionary.

        Args:
            material (Material): Material object.

        Returns:
            Dict[str, Any]: dictionary containing the structural material parameters

        """

        structural_material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                                    "Variables": {}}

        # add material parameters to dictionary based on material type.
        if isinstance(material.material_parameters, EulerBeam2D) or isinstance(material.material_parameters,
                                                                               EulerBeam3D):
            structural_material_dict.update(self.__create_euler_beam_dict(material))
        elif isinstance(material.material_parameters, ElasticSpringDamper):
            structural_material_dict["Variables"] = self.__create_elastic_spring_damper_dict(material)
        elif isinstance(material.material_parameters, NodalConcentrated):
            structural_material_dict["Variables"] = self.__create_nodal_concentrated_dict(material)

        return structural_material_dict

    def __create_material_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters

        Args:
            material (Material): material object

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {"model_part_name": material.name,
                                         "properties_id": material.id,
                                         "Material": {"constitutive_law": {"name": ""},
                                                      "Variables": {}},
                                         "Tables": {}
                                        }

        # add material parameters to dictionary based on material type.
        if isinstance(material.material_parameters, SoilParametersABC):
            material_dict["Material"].update(self.__create_soil_material_dict(material))
        elif isinstance(material.material_parameters, StructuralParametersABC):
            material_dict["Material"].update(self.__create_structural_material_dict(material))

        return material_dict

    def write_material_parameters_json(self, materials: List[Material], filename: str):
        """
        Writes the material parameters to a json file

        Args:
            materials (List[Material]): list of material objects
            filename (str): filename of the output json file

        """

        materials_dict: Dict[str, Any] = {"properties": []}

        for material in materials:
            materials_dict["properties"].append(self.__create_material_dict(material))

        json.dump(materials_dict, open(filename, 'w'), indent=4)

