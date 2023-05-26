import json

import numpy as np

from stem.material import (Material,
                           LinearElastic2D,
                           LinearElastic3D,
                           SmallStrainUmat2DLaw,
                           SmallStrainUmat3DLaw,
                           SmallStrainUdsm2DLaw,
                           SmallStrainUdsm3DLaw)


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


    def __create_umat_material_dict(self, material):
        material_dict = material.__dict__

        material_dict["UDSM_NAME"] = material_dict["UMAT_NAME"].pop()
        material_dict["IS_FORTRAN_UDSM"] = material_dict["IS_FORTRAN_UMAT"].pop()

        return material_dict

    def __create_udsm_material_dict(self, material):
        material_dict = material.__dict__

        material_dict["UMAT_PARAMETERS"] = material_dict["UDSM_PARAMETERS"].pop()

        return material_dict


    def __create_material_dict(self, material):

        material_dict = {"model_part_name": material.name,
                         "properties_id": material.id,
                         "Material": {"constitutive_law": {"name": ""},
                                      "Variables": {}},
                         "Tables": {}
                        }

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

        # get retention parameters
        retention_law = material.retention_parameters.__name__
        retention_parameters = material.retention_parameters.__dict__

        material_dict["Material"]["Variables"]["RETENTION_LAW"] = retention_law
        material_dict["Material"]["Variables"].update(retention_parameters)

        return material_dict

    def write_material_parameters_json(self, materials, filename):

        materials_dict = {"properties": []}

        for material in materials:
            materials_dict["properties"].append(self.__create_material_dict(material))

        json.dump(materials_dict, open(filename, 'w'), indent=4)


