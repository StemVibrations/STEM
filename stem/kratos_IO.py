import json
from typing import Dict,List, Any,Union

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

        #todo improve this such that nodes and elements are written in the same mdpa file, where the elements are split per physical group

        np.savetxt('0.nodes.mdpa', nodes, fmt=['%.f', '%.10f', '%.10f', '%.10f'], delimiter=' ')
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

    def write_project_parameters_json(self, filename):

        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()

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

    def __create_spring_damper_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters for a spring damper material. The
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
            material (Material): material object

        Returns:
            Dict[str, Any]: dictionary containing the material parameters
        """
        material_dict: Dict[str, Any] = material.material_parameters.__dict__

        # Change naming of coefficient to ratio as this is the (incorrect) naming in Kratos
        material_dict["NODAL_DAMPING_RATIO"] = material_dict.pop("NODAL_DAMPING_COEFFICIENT")
        return material_dict

    def __create_linear_elastic_soil_dict(self, material: Material) -> Dict[str, Any]:

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
                                                    material_parameters: SoilParametersABC):

        if isinstance(material_parameters.SOIL_TYPE, DrainedSoil):
            drained_soil_parameters_dict = self.__create_drained_soil_parameters_dict(
                material_parameters.SOIL_TYPE)
            variables_dict.update(drained_soil_parameters_dict)
        elif isinstance(material_parameters.SOIL_TYPE, UndrainedSoil):
            undrained_soil_parameters_dict = self.__create_undrained_soil_parameters_dict(
                material_parameters.SOIL_TYPE)
            variables_dict.update(undrained_soil_parameters_dict)
        elif isinstance(material_parameters.SOIL_TYPE, TwoPhaseSoil2D) or isinstance(
                material_parameters.SOIL_TYPE, TwoPhaseSoil3D):
            two_phase_soil_parameters_dict = self.__create_two_phase_soil_parameters_dict(
                material_parameters.SOIL_TYPE)
            variables_dict.update(two_phase_soil_parameters_dict)

        variables_dict.pop("SOIL_TYPE")

    def __create_drained_soil_parameters_dict(self, drained_soil_parameters: DrainedSoil) -> Dict[str, Any]:
        """
        Creates a dictionary containing the drained soil parameters

        Args:
            drained_soil_parameters:

        Returns:


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
        Creates a dictionary containing the undrained soil parameters

        Args:
            undrained_soil_parameters:

        Returns:


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

    def __create_two_phase_soil_parameters_dict(self, two_phase_soil_parameters: TwoPhaseSoil3D) -> Dict[str, Any]:
        """
        Creates a dictionary containing the two phase soil parameters

        Args:
            two_phase_soil_parameters:

        Returns:

        """

        two_phase_soil_parameters_dict = two_phase_soil_parameters.__dict__
        two_phase_soil_parameters_dict["IGNORE_UNDRAINED"] = False

        if two_phase_soil_parameters_dict["BIOT_COEFFICIENT"] is None:
            two_phase_soil_parameters_dict.pop("BIOT_COEFFICIENT")

        return two_phase_soil_parameters_dict

    def __create_soil_material_dict(self, material: Material) -> Dict[str, Any]:

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

        # add retention parameters to dictionary
        soil_material_dict["Variables"]["RETENTION_LAW"] = retention_law
        soil_material_dict["Variables"].update(retention_parameters)

        return soil_material_dict

    def __create_euler_beam_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the euler beam parameters

        Args:
            material:

        Returns:

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

            structural_material_dict: Dict[str, Any] = {"constitutive_law": {"name": ""},
                                                        "Variables": {}}

            # add material parameters to dictionary based on material type.
            if isinstance(material.material_parameters, EulerBeam2D) or isinstance(material.material_parameters, EulerBeam3D):
                structural_material_dict.update(self.__create_euler_beam_dict(material))

            #     structural_material_dict.update(self.__create_linear_elastic_dict(material))
            # elif isinstance(material.material_parameters, NonLinearElastic):
            #     structural_material_dict.update(self.__create_non_linear_elastic_dict(material))
            elif isinstance(material.material_parameters, SmallStrainUmatLaw):
                structural_material_dict.update(self.__create_umat_dict(material))
            elif isinstance(material.material_parameters, SmallStrainUdsmLaw):
                structural_material_dict.update(self.__create_udsm_dict(material))

            return structural_material_dict

    def __create_material_dict(self, material: Material) -> Dict[str, Any]:
        """
        Creates a dictionary containing the material parameters

        Args:
            material (Material): material object

        Returns:  Dict[str, Any]: dictionary containing the material parameters

        """

        # initialize material dictionary
        material_dict: Dict[str, Any] = {"model_part_name": material.name,
                                         "properties_id": material.id,
                                         "Material": {"constitutive_law": {"name": ""},
                                                      "Variables": {}},
                                         "Tables": {}
                                        }

        if isinstance(material.material_parameters, SoilParametersABC):
            material_dict["Material"].update(self.__create_soil_material_dict(material))
        elif isinstance(material.material_parameters, StructuralParametersABC):
            pass

        #
        # elif isinstance(material.material_parameters, BeamLaw):
        #     material_dict["Material"]["constitutive_law"]["name"] = \
        #         "KratosMultiphysics.StructuralMechanicsApplication.BeamConstitutiveLaw"
        #     material_dict["Material"]["Variables"] = material.material_parameters.__dict__
        # elif isinstance(material.material_parameters, SpringDamperLaw):
        #     material_dict.update(self.__create_spring_damper_dict(material))
        # elif isinstance(material.material_parameters, NodalConcentratedLaw):
        #     material_dict.update(self.__create_nodal_concentrated_dict(material))


        return material_dict

    def write_material_parameters_json(self, materials: List[Material], filename: str):
        """
        Writes the material parameters to a json file

        Args:
            materials (List[Material]): list of material objects
            filename: filename of the output json file

        """

        materials_dict: Dict[str, Any] = {"properties": []}

        for material in materials:
            materials_dict["properties"].append(self.__create_material_dict(material))

        json.dump(materials_dict, open(filename, 'w'), indent=4)


