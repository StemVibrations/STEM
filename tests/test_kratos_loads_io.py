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
        line_load_parameters = LineLoad(active=[False, True, False], value=[0, -300, 0])
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
            offset=3.0,
        )

        # collect the part names and parameters into a dictionary
        # TODO: change later when model part is implemented
        all_boundary_parameters = {
            "test_point_load": point_load_parameters,
            "test_line_load": line_load_parameters,
            "test_surface_load": surface_load_parameters,
            "test_moving_load": moving_point_load_parameters,
        }

        # initialize process dictionary
        test_dictionary: Dict[str, Any] = {
            "processes": {"constraints_process_list": [], "loads_process_list": []}
        }

        # write dictionary for the load(s)
        # TODO: when model part are implemented, generate file through kratos_io
        boundaries_io = KratosLoadsIO(domain="PorousDomain")

        for part_name, part_parameters in all_boundary_parameters.items():
            _parameters = boundaries_io.create_load_dict(
                part_name=part_name, parameters=part_parameters
            )
            test_dictionary["processes"]["loads_process_list"].append(_parameters)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )
