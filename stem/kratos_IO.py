import json
from typing import Dict, List, Any, Union
from copy import deepcopy

import numpy as np

from stem.soil_material import *
from stem.structural_material import *


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
            "Parameters": deepcopy(load.load_parameters.__dict__),
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
        material_dict["NUMBER_OF_UMAT_PARAMETERS"] = len(material_dict["UMAT_PARAMETERS"])

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
    def __create_elastic_spring_damper_dict(material: StructuralParametersABC) -> Dict[str, Any]:
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
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop("NODAL_DAMPING_COEFFICIENT")
        material_dict["NODAL_ROTATIONAL_DAMPING_RATIO"] = material_dict.pop("NODAL_ROTATIONAL_DAMPING_COEFFICIENT")
        return material_dict

    @staticmethod
    def __create_nodal_concentrated_dict(material: StructuralParametersABC) -> Dict[str, Any]:
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
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop("NODAL_DAMPING_COEFFICIENT")
        return material_dict

    def __create_linear_elastic_soil_dict(self, material: SoilConstitutiveLawABC) -> Dict[str, Any]:
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
        material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                         "Variables": deepcopy(material.__dict__)}
        if self.ndim == 2:
            material_dict["constitutive_law"]["name"] = "GeoLinearElasticPlaneStrain2DLaw"

        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "GeoLinearElastic3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __create_umat_soil_dict(self, material: SoilConstitutiveLawABC) -> Dict[str, Any]:
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
        material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                         "Variables": self.__create_umat_material_dict(material)}

        if self.ndim == 2:
            material_dict["constitutive_law"]["name"] = "SmallStrainUMAT2DPlaneStrainLaw"
        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "SmallStrainUMAT3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __create_udsm_soil_dict(self, material: SoilConstitutiveLawABC) -> Dict[str, Any]:
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
        material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                         "Variables": self.__create_udsm_material_dict(material)}

        if self.ndim == 2:
            material_dict["constitutive_law"]["name"] = "SmallStrainUDSM2DPlaneStrainLaw"
        elif self.ndim == 3:
            material_dict["constitutive_law"]["name"] = "SmallStrainUDSM3DLaw"
        else:
            raise ValueError("Dimension not supported")

        return material_dict

    def __add_soil_formulation_parameters_to_material_dict(self, variables_dict: Dict[str, Any],
                                                    soil_parameters: SoilMaterial):
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
            one_phase_soil_parameters_dict = self.__create_one_phase_soil_parameters_dict(
                soil_parameters.soil_formulation)
            variables_dict.update(one_phase_soil_parameters_dict)
        elif isinstance(soil_parameters.soil_formulation, TwoPhaseSoil) :
            two_phase_soil_parameters_dict = self.__create_two_phase_soil_parameters_dict(
                soil_parameters.soil_formulation)
            variables_dict.update(two_phase_soil_parameters_dict)

        # Remove ndim as this is not a material parameter
        if "ndim" in variables_dict:
            variables_dict.pop("ndim")

    @staticmethod
    def __create_one_phase_soil_parameters_dict(soil_parameters: OnePhaseSoil) -> Dict[str, Any]:
        """
        Creates a dictionary containing the single phase soil parameters. For one phase soil, the permeability is set to
        near zero. Biot coefficient is added if it is not None.

        Args:
            soil_parameters (OnePhaseSoil): one phase soil parameters object.

        Returns:
            Dict[str, Any]: dictionary containing the one phase soil parameters

        """

        soil_parameters_dict = deepcopy(soil_parameters.__dict__)
        soil_parameters_dict["IGNORE_UNDRAINED"] = soil_parameters_dict.pop("IS_DRAINED")
        soil_parameters_dict["PERMEABILITY_XX"] = 1e-30
        soil_parameters_dict["PERMEABILITY_YY"] = 1e-30
        soil_parameters_dict["PERMEABILITY_ZZ"] = 1e-30
        soil_parameters_dict["PERMEABILITY_XY"] = 0
        soil_parameters_dict["PERMEABILITY_YZ"] = 0
        soil_parameters_dict["PERMEABILITY_ZX"] = 0

        # Create a new dictionary without None values
        soil_parameters_dict = {k: v for k, v in soil_parameters_dict.items() if v is not None}

        return soil_parameters_dict

    @staticmethod
    def __create_two_phase_soil_parameters_dict(two_phase_soil_parameters: TwoPhaseSoil) \
            -> Dict[str, Any]:
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
        two_phase_soil_parameters_dict = {k: v for k, v in two_phase_soil_parameters_dict.items() if v is not None}

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

        soil_material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                              "Variables": {}}

        # add material parameters to dictionary based on material type.
        if isinstance(material.constitutive_law, LinearElasticSoil):
            soil_material_dict.update(self.__create_linear_elastic_soil_dict(material.constitutive_law))
        elif isinstance(material.constitutive_law, SmallStrainUmatLaw):
            soil_material_dict.update(self.__create_umat_soil_dict(material.constitutive_law))
        elif isinstance(material.constitutive_law, SmallStrainUdsmLaw):
            soil_material_dict.update(self.__create_udsm_soil_dict(material.constitutive_law))

        self.__add_soil_formulation_parameters_to_material_dict(soil_material_dict["Variables"], material)

        # get retention parameters
        retention_law = material.retention_parameters.__class__.__name__
        retention_parameters: Dict[str, Any] = deepcopy(material.retention_parameters.__dict__)

        # add retention parameters to dictionary
        soil_material_dict["Variables"]["RETENTION_LAW"] = retention_law
        soil_material_dict["Variables"].update(retention_parameters)

        # add fluid parameters to dictionary
        fluid_parameters: Dict[str, Any] = deepcopy(material.fluid_properties.__dict__)
        fluid_parameters["DENSITY_WATER"] = fluid_parameters.pop("DENSITY_FLUID")

        soil_material_dict["Variables"].update(fluid_parameters)

        return soil_material_dict

    def __create_euler_beam_dict(self, material_parameters: StructuralParametersABC) -> Dict[str, Any]:
        """
        Creates a dictionary containing the euler beam parameters

        Args:
            material: Material object containing the material parameters

        Returns:
            euler_beam_parameters_dict: Dictionary containing the euler beam parameters

        """

        material_parameters_dict = deepcopy(material_parameters.__dict__)

        # Create a new dictionary without None values
        material_parameters_dict = {k: v for k, v in material_parameters_dict.items() if v is not None}

        # remove ndim from dictionary
        if "ndim" in material_parameters_dict.keys():
            material_parameters_dict.pop("ndim")

        # initialize material dictionary
        euler_beam_parameters_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                                      "Variables": material_parameters_dict}

        # add constitutive law name to dictionary based on dimension
        if self.ndim == 2:
            euler_beam_parameters_dict["constitutive_law"]["name"] = "LinearElastic2DBeamLaw"
        elif self.ndim == 3:
            euler_beam_parameters_dict["constitutive_law"]["name"] = \
                "KratosMultiphysics.StructuralMechanicsApplication.BeamConstitutiveLaw"

        return euler_beam_parameters_dict

    def __create_structural_material_dict(self, material: StructuralMaterial) -> Dict[str, Any]:
        """
        Creates a dictionary containing the structural material parameters. The structural material parameters are based
        on the material type. The material parameters are added to the dictionary.

        Args:
            material (StructuralMaterial): Material object.

        Returns:
            Dict[str, Any]: dictionary containing the structural material parameters

        """

        structural_material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                                    "Variables": {}}

        # add material parameters to dictionary based on material type.
        if isinstance(material.material_parameters, EulerBeam):
            structural_material_dict.update(self.__create_euler_beam_dict(material.material_parameters))
        elif isinstance(material.material_parameters, ElasticSpringDamper):
            structural_material_dict["Variables"] = self.__create_elastic_spring_damper_dict(material.material_parameters)
        elif isinstance(material.material_parameters, NodalConcentrated):
            structural_material_dict["Variables"] = self.__create_nodal_concentrated_dict(material.material_parameters)

        return structural_material_dict

    def __create_material_dict(self, material: Union[SoilMaterial, StructuralMaterial], material_id: int) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters

        Args:
            material (Union[SoilMaterial, StructuralMaterial]): material object
            material_id (int): material id

        Returns:
            Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {"model_part_name": material.name,
                                         "properties_id": material_id,
                                         "Material": {"constitutive_law": {"name": ""},
                                                      "Variables": {}},
                                         "Tables": {}
                                        }

        # add material parameters to dictionary based on material type.
        if isinstance(material, SoilMaterial):
            material_dict["Material"].update(self.__create_soil_material_dict(material))
        elif isinstance(material, StructuralMaterial):
            material_dict["Material"].update(self.__create_structural_material_dict(material))

        return material_dict

    def write_material_parameters_json(self, materials: List[Union[SoilMaterial, StructuralMaterial]], filename: str):
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
            materials_dict["properties"].append(self.__create_material_dict(material, material_id))
            material_id += 1

        # write material dictionary to json file
        json.dump(materials_dict, open(filename, 'w'), indent=4)

