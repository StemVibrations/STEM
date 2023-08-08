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

    Use example:


    Attributes:
        - steps (Union[Sequence[float], npty.NDArray[np._int]]): step number
        - time (Union[Sequence[float], npty.NDArray[np.float64]]): time steps
        - amplitude (Union[Sequence[float], npty.NDArray[np.float64]]): amplitude values
        - id (Optional[int]): id related to the table. Not specified by the user.
    """

    steps: Union[Sequence[int], npty.NDArray[np.int_]]
    time: Union[Sequence[float], npty.NDArray[np.float64]]
    amplitude: Union[Sequence[float], npty.NDArray[np.float64]]
    id: Optional[int] = None

