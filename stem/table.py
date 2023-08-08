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
        - amplitude (Union[Sequence[float], npty.NDArray[np.float64]]): amplitude values of the load/constraint.
        - step (Union[Sequence[Union[int,float]], npty.NDArray[Union[np.float64, np.int_]]]): time [s] or
            simulation steps of the corresponding amplitudes.
        - step_type (str): step type specified for the step variable. Select either `step` if simulation step id \
            (integers) are provided or `time`, if time steps in seconds are provided. \
             Currently only `time` is supported.
        - id (Optional[int]): id related to the table. Not specified by the user.
    """

    name: str
    amplitude: Union[Sequence[Union[int,float]], npty.NDArray[Union[np.float64, np.int_]]]
    step: Union[Sequence[float], npty.NDArray[np.float64]]
    step_type: str = "time"
    id: Optional[int] = None

    def __post_init__(self):
        """
        Post-initialisation method to validate table attributes.

        Raises:
            - ValueError
        """

        # lower case to avoid case sensitivity
        self.step_type = self.step_type.lower()

        if self.id is not None:
            raise ValueError(f"id attribute should not be specified by the user in table: {self.name}")

        if self.step_type not in ["step", "time"]:
            raise ValueError(f"Specified step is not understood: {self.step_type}.\n"
                             f"Please specify one `step` or `time` for table: {self.name}")

        if len(self.step) != len(self.amplitude):
            raise ValueError(f"Dimension mismatch between time/step and amplitudes in table: {self.name}")