import numpy as np


class KratosIO:

    def __init__(self):
        pass

    def write_mesh_to_mdpa(self, nodes, elements, filename):
        """
        saves mesh data to mdpa file

        todo improve this, such that it can be read by Kratos, also do checks if lines, surfaces, volumes are
        available. Don't write if they are not

        :param nodes: node tag followed by node coordinates in an array
        :param elements: list of all elements per element type # todo, should be per physical group

        :return: -
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

    def write_material_parameters_json(self, materials, filename):
        pass

