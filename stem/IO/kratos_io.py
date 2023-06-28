import json
from functools import reduce
from typing import List

from stem.IO.kratos_boundaries_io import KratosBoundariesIO
from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.IO.kratos_output_io import KratosOutputsIO
from stem.boundary import Boundary
from stem.load import Load
from stem.output import Output
from stem.utils import merge

DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        ndim (int): The number of dimensions of the problem.
        material_io (:class:`stem.io.kratos_loads_io.KratosMaterialIO`): The material IO
            object.
        loads_io (:class:`stem.io.kratos_loads_io.KratosLoadsIO`): The loads IO object.
        boundaries_io (:class:`stem.io.kratos_loads_io.KratosBoundariesIO`): The
            boundary conditions IO object.
        outputs_io (:class:`stem.io.kratos_loads_io.KratosOutputsIO`): The outputs IO
            object.

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
        self.material_io = KratosMaterialIO(self.ndim)
        self.loads_io = KratosLoadsIO(DOMAIN)
        self.boundaries_io = KratosBoundariesIO(DOMAIN)
        self.outputs_io = KratosOutputsIO(DOMAIN)

    def write_material_parameters_json(self, *args, **kwargs):
        """
        Writes material parameters to json file.
        """
        self.material_io.write_material_parameters_json(*args, **kwargs)

    def write_project_parameters_json(
        self,
        loads: List[Load],
        boundaries: List[Boundary],
        outputs: List[Output],
        filename: str,
        solver_settings=None,
    ):
        """
        Writes the project parameters to json.

        Args:
            loads (:class:`stem.load.Load`): list of Load objects
            boundaries (:class:`stem.boundary.Boundary`): list of Boundary objects
            outputs (:class:`stem.output.Output`): list of Load objects
            filename (str):
            solver_settings (Any): list of Solver settings objects TODO: add when ready.

        Returns:
            None
        """
        # now dictionaries have common keys and need merge with an helping function
        _dictionaries = (
            self.outputs_io.create_output_process_dictionary(outputs=outputs),
            self.loads_io.create_loads_process_dict(loads=loads),
            self.boundaries_io.create_dictionaries_for_boundaries(
                boundaries=boundaries
            ),
        )
        # recursive merging
        _dict_merged = reduce(merge, _dictionaries)

        json.dump(_dict_merged, open(filename, "w"), indent=4)

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
            filename (str): filename of json file

        """

        self.__write_problem_data()
        self.__write_solver_settings()
        self.__write_output_processes()
        self.__write_input_processes()
        self.__write_constraints()
        self.__write_loads()
        # todo write Projectparameters.json
