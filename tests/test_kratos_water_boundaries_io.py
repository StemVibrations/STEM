import json

import numpy as np
import pytest

from stem.IO.kratos_water_processes_io import KratosWaterProcessesIO
from stem.model import Model
from stem.water_processes import *
from stem.load import PointLoad
from tests.utils import TestUtils


class TestKratosLoadsIO:
    def test_create_load_process_dict_no_tables(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file excluding tables.
        """

        # set water boundary parameters
        water_boundary_parameters = UniformWaterPressure(water_pressure=2.0, is_fixed=False)

        # initialise model
        kratos_water_boundary_io = KratosWaterProcessesIO(domain="PorousDomain")
        uniform_water_boundary_dict = kratos_water_boundary_io.create_water_boundary_condition_dict(
            "test_water_boundary", water_boundary_parameters)

        # set expected water boundary dictionary
        expected_water_boundary_dict = {
            "python_module": "apply_scalar_constraint_table_process",
            "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
            "process_name": "ApplyScalarConstraintTableProcess",
            "Parameters": {"model_part_name": "PorousDomain.test_water_boundary",
                           "variable_name": "WATER_PRESSURE",
                           "table": 0,
                           "value": 2.0,
                           "is_fixed": False,
                           "fluid_pressure_type": "Uniform"}
        }

        # create non water boundary parameters
        non_water_boundary = PointLoad(active=[True, False, True], value=[1000, 0, 0])

        # expected not implemented error
        with pytest.raises(NotImplementedError, match="Water boundary type: PointLoad not implemented."):
            non_water_boundary_dict = kratos_water_boundary_io.create_water_boundary_condition_dict(
                "test_non_water_boundary", non_water_boundary)


