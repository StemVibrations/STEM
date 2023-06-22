
import json

from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.load import *

from tests.utils import TestUtils


class TestKratosLoadsIO:
    def test_create_load_process_dict(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file
        """
        # define load(s) parameters
        # point load
        point_load_parameters = PointLoad(
            active=[True, False, True], value=[1000, 0, 0]
        )
        # line load
        line_load_parameters = LineLoad(
            active=[False, True, False], value=[0, -300, 0]
        )
        # surface load
        surface_load_parameters = SurfaceLoad(
            active=[False, False, True], value=[0, 0, 500]
        )

        # moving (point) load
        moving_point_load_parameters = MovingLoad(
            origin=[0.0, 1.0, 2.0],
            load=[0.0, -10.0, 0.0],
            direction=[1.0, 0.0, -1.0],
            velocity=5.0,
            offset=3.0
        )

        # create Load objects and store in the list
        point_load = Load(part_name="test_name", load_parameters=point_load_parameters)

        line_load = Load(part_name="test_name_line",
                         load_parameters=line_load_parameters)
        surface_load = Load(part_name="test_name_surface",
                            load_parameters=surface_load_parameters)
        moving_point_load = Load(part_name="test_name_moving",
                                 load_parameters=moving_point_load_parameters)

        all_loads = [point_load, line_load, surface_load, moving_point_load]

        # write dictionary for the load(s)
        kratos_io = KratosLoadsIO(domain="PorousDomain")
        test_dictionary = kratos_io.create_loads_process_dict(all_loads)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )