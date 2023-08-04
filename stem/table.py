from dataclasses import dataclass
from typing import Optional, Union, Sequence
import numpy.typing as npty
import numpy as np

@dataclass
class Table:

    """
    Class to write time-dependent functions for imposed load and constraints

    Attributes:
        - steps (Union[Sequence[float], npty.NDArray[np._int]]): step number
        - time (Union[Sequence[float], npty.NDArray[np.float64]]): time steps
        - amplitude (Union[Sequence[float], npty.NDArray[np.float64]]): amplitude values
        - id (Optional[int]): id related to the table
    """

    steps: Union[Sequence[int], npty.NDArray[np.int_]]
    time: Union[Sequence[float], npty.NDArray[np.float64]]
    amplitude: Union[Sequence[float], npty.NDArray[np.float64]]
    id: Optional[int] = None

