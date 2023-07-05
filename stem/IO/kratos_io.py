from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.IO.kratos_material_io import KratosMaterialIO


DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        - ndim (int): The number of dimensions of the problem.
        - loads_io (:class:`stem.IO.kratos_loads_io.KratosLoadsIO`): The loads IO object.
        - material_io (:class:`stem.IO.kratos_material_io.KratosMaterialIO`): The material IO object.
    """

    def __init__(self, ndim: int):
        """
        Constructor of KratosIO class

        Args:
            - ndim: The number of dimensions of the problem.
        """

        self.ndim = ndim
        self.loads_io = KratosLoadsIO(DOMAIN)
        self.material_io = KratosMaterialIO(self.ndim)

    def write_mesh_to_mdpa(self, filename):
        """
        Saves mesh data to mdpa file

        Args:
            - filename (str): filename of mdpa file

        Returns:
        """
        pass

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

    def write_project_parameters_json(self, filename: str):
        """
        Writes project parameters to json file

        Args:
            - filename (str): filename of json file

        Returns:
        """
        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()
        self.__write_constraints()
        self.__write_loads()
        # todo write Projectparameters.json
