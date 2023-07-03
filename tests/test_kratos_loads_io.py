import json

from stem.IO.kratos_io import KratosIO
from stem.load import *
from stem.mesh import Node
from stem.model import Model
from stem.model_part import ModelPart
from tests.utils import TestUtils


class TestKratosLoadsIO:
    def test_create_load_process_dict(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file
        """

        # dummy node
        node = Node(id=1, coordinates=(0, 0, 0))

        # define model parts

        # define load(s) parameters
        # point load
        mp_point_load = ModelPart(
            name="test_name_point",
            nodes=[node],
            parameters=PointLoad(active=[True, False, True], value=[1000, 0, 0]),
        )
        # line load
        mp_line_load = ModelPart(
            name="test_name_line",
            nodes=[node],
            parameters=LineLoad(active=[False, True, False], value=[0, -300, 0]),
        )
        # surface load
        mp_surface_load = ModelPart(
            name="test_name_surface",
            nodes=[node],
            parameters=SurfaceLoad(active=[False, False, True], value=[0, 0, 500]),
        )

        # moving (point) load
        mp_moving_point_load = ModelPart(
            name="test_name_moving",
            nodes=[node],
            parameters=MovingLoad(
                origin=[0.0, 1.0, 2.0],
                load=[0.0, -10.0, 0.0],
                direction=[1.0, 0.0, -1.0],
                velocity=5.0,
                offset=3.0,
            ),
        )

        # collect model parts together
        model_parts = [
            mp_point_load,
            mp_line_load,
            mp_surface_load,
            mp_moving_point_load,
        ]

        # write dictionary for the load(s)
        kratos_io = KratosIO(ndim=2, model=Model(ndim=2, model_parts=model_parts))
        test_dictionary = kratos_io.write_project_parameters_json()

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )
