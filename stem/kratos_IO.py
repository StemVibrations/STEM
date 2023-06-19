import json
from typing import Dict, List, Any, Union
from copy import deepcopy

import numpy as np

DOMAIN = "PorousDomain"


class KratosIO:
    """
    Class containing methods to write mesh and problem data to Kratos

    Attributes:
        ndim (int): The number of dimensions of the problem.

    """

    def __init__(self, ndim: int):
        self.ndim = ndim






