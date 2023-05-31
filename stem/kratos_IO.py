import json
from typing import Dict,List, Any

import numpy as np

from stem.material import (Material,
                           SoilMaterial2D,
                           SoilMaterial3D,
                           LinearElastic2D,
                           LinearElastic3D,
                           SmallStrainUmat2DLaw,
                           SmallStrainUmat3DLaw,
                           SmallStrainUdsm2DLaw,
                           SmallStrainUdsm3DLaw,
                           BeamLaw,
                           SpringDamperLaw,
                           NodalConcentratedLaw)


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

        # add material parameters to dictionary based on material type.
        if isinstance(material.material_parameters, LinearElastic2D):
            material_dict["Material"]["constitutive_law"]["name"] = "GeoLinearElasticPlaneStrain2DLaw"
            material_dict["Material"]["Variables"] = material.material_parameters.__dict__
        elif isinstance(material.material_parameters, LinearElastic3D):
            material_dict["Material"]["constitutive_law"]["name"] = "GeoLinearElastic3DLaw"
            material_dict["Material"]["Variables"] = material.material_parameters.__dict__
        elif isinstance(material.material_parameters, SmallStrainUmat2DLaw):
            material_dict["Material"]["constitutive_law"]["name"] = "SmallStrainUMAT2DPlaneStrainLaw"
            material_dict["Material"]["Variables"] = self.__create_umat_material_dict(material)
        elif isinstance(material.material_parameters, SmallStrainUmat3DLaw):
            material_dict["Material"]["constitutive_law"]["name"] = "SmallStrainUMAT3DLaw"
            material_dict["Material"]["Variables"] = self.__create_umat_material_dict(material)
        elif isinstance(material.material_parameters, SmallStrainUdsm2DLaw):
            material_dict["Material"]["constitutive_law"]["name"] = "SmallStrainUDSM2DPlaneStrainLaw"
            material_dict["Material"]["Variables"] = self.__create_udsm_material_dict(material)
        elif isinstance(material.material_parameters, SmallStrainUdsm3DLaw):
            material_dict["Material"]["constitutive_law"]["name"] = "SmallStrainUDSM3DLaw"
            material_dict["Material"]["Variables"] = self.__create_udsm_material_dict(material)
        elif isinstance(material.material_parameters, BeamLaw):
            material_dict["Material"]["constitutive_law"]["name"] = \
                "KratosMultiphysics.StructuralMechanicsApplication.BeamConstitutiveLaw"
            material_dict["Material"]["Variables"] = material.material_parameters.__dict__
        elif isinstance(material.material_parameters, SpringDamperLaw):
            material_dict.update(self.__create_spring_damper_dict(material))
        elif isinstance(material.material_parameters, NodalConcentratedLaw):
            material_dict.update(self.__create_nodal_concentrated_dict(material))

        # add retention parameters to dictionary if material is a soil material
        if (isinstance(material.material_parameters, SoilMaterial2D) or
                isinstance(material.material_parameters, SoilMaterial3D)):

            # get retention parameters
            retention_law = material.retention_parameters.__class__.__name__
            retention_parameters: Dict[str, Any] = material.retention_parameters.__dict__

            # add retention parameters to dictionary
            material_dict["Material"]["Variables"]["RETENTION_LAW"] = retention_law
            material_dict["Material"]["Variables"].update(retention_parameters)

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


