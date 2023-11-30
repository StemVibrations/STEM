import pytest

from stem.IO.kratos_water_processes_io import KratosWaterProcessesIO
from stem.water_processes import *
from stem.load import PointLoad
from tests.utils import TestUtils


class TestWaterProcessesIO:
    """
    Test class for the KratosWaterProcessesIO class.
    """

    def test_create_water_pressure_dict(self):
        """
        Test the creation of the  water process dictionary for the
        ProjectParameters.json file.
        """

        # set water boundary parameters
        water_boundary_parameters = UniformWaterPressure(water_pressure=2.0, is_fixed=False)

        # initialise model
        kratos_water_boundary_io = KratosWaterProcessesIO(domain="PorousDomain")
        uniform_water_boundary_dict = kratos_water_boundary_io.create_water_process_dict(
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

        # compare dictionaries
        TestUtils.assert_dictionary_almost_equal(expected_water_boundary_dict, uniform_water_boundary_dict)

        # create non water boundary parameters
        non_water_boundary = PointLoad(active=[True, False, True], value=[1000, 0, 0])

        # expected not implemented error
        with pytest.raises(NotImplementedError, match="Water boundary type: PointLoad not implemented."):
            non_water_boundary_dict = kratos_water_boundary_io.create_water_process_dict(
                "test_non_water_boundary", non_water_boundary)


