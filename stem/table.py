from dataclasses import dataclass
from typing import Optional, Union, Sequence
import numpy.typing as npty
import numpy as np


@dataclass
class Table:

    """
    Class to write time-dependent functions for imposed load and constraints.
    If analysis runs outside the specified time-steps, the function is extrapolated.
    If load/constraint is required to remain constant, please specify the same load/constraint value for the last
    two point of the sequence/array.

    Attributes:
        - name (str): label of the table object.
        - values (Union[Sequence[float], npty.NDArray[np.float64]]): alues of the load/constraint.
        - times (Union[Sequence[Union[int,float]], npty.NDArray[Union[np.float64, np.int_]]]): time [s]
            corresponding to the values specified.
        - step_type (str): step type specified for the step variable. Select either `step` if simulation step id \
            (integers) are provided or `time`, if time steps in seconds are provided. \
             Currently only `time` is supported.
        - __id (Optional[int]): unique identifier for the table.
    """

    name: str
    values: Union[Sequence[Union[int, float]], npty.NDArray[Union[np.float64, np.int_]]]
    times: Union[Sequence[float], npty.NDArray[np.float64]]
    __id: Optional[int] = None

    @property
    def id(self) -> int:
        """
        Getter for the id of the table.

        Returns:
            - int: The id of the table.

        """
        return self.__id

    @id.setter
    def id(self, value: int):
        """
        Setter for the id of the table.

        Args:
            - value (int): The id of the table.

        """
        self.__id = value

    def __post_init__(self):
        """
        Post-initialisation method to validate table attributes.

        Raises:
            - ValueError: if time and values have different number of elements.
        """

        if len(self.times) != len(self.values):
            raise ValueError(f"Dimension mismatch between times and values in table: {self.name}\n"
                             f" - times: {len(self.times)}\n"
                             f" - values: {len(self.values)}\n")
