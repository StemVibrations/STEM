import json

import numpy as np

from stem.IO.kratos_io import KratosIO
from stem.model import Model
from stem.load import *
from tests.utils import TestUtils


class TestKratosLoadsIO:
    def test_create_load_process_dict_no_tables(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file excluding tables.
        """

        # initialise model
        model = Model(ndim=3)
        # define points of the load geometry
        point_load_coords = [(0, 0, 0)]
        line_load_coords = [(0, 0, 0), (0, 1, 0)]
        moving_load_coords = [(1, 0, 0), (1, 0.4, 0)]
        surface_load_coords = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]

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
            origin=[1.0, 0.1, 0.0],
            load=[0.0, -10.0, 0.0],
            direction=[1.0, 0.0, -1.0],
            velocity=5.0,
            offset=3.0,
        )

        # add loads to process model parts:
        model.add_load_by_coordinates(point_load_coords, point_load_parameters, 'test_point_load')
        model.add_load_by_coordinates(line_load_coords, line_load_parameters, 'test_line_load')
        model.add_load_by_coordinates(surface_load_coords, surface_load_parameters, 'test_surface_load')
        model.add_load_by_coordinates(moving_load_coords, moving_point_load_parameters, 'test_moving_load')
        model.synchronise_geometry()

        # write dictionary for the load(s)
        kratos_io = KratosIO(ndim=model.ndim)

        test_dictionary = kratos_io.write_project_parameters_json(
            model=model,
            outputs=[],
            mesh_file_name="test_load_parameters.json",
            materials_file_name=""
        )

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters_no_table.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            expected_load_parameters_json, test_dictionary
        )

    def test_create_load_process_dict_with_tables(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file including tables.
        """

        # initialise model
        model = Model(ndim=3)
        # define points of the load geometry
        point_load_coords = [(0, 0, 0)]
        line_load_coords = [(0, 0, 0), (0, 1, 0)]
        moving_load_coords = [(1, 0, 0), (1, 0.4, 0)]
        surface_load_coords = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]

        # define tables
        _time = np.array([0, 1, 2, 3, 4, 5])

        _amplitude1 = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(steps=np.arange(len(_time))+1, time=_time, amplitude=_amplitude1)

        _amplitude2 = np.array([0, -5, 5, -5, 0, 0])
        table2 = Table(steps=np.arange(len(_time))+1, time=_time, amplitude=_amplitude2)

        # define load(s) parameters
        # point load
        point_load_parameters = PointLoad(
            active=[True, False, True], value=[table1, -20, 0]
        )
        # line load
        line_load_parameters = LineLoad(active=[False, True, False], value=[-10, table2, 30])
        # surface load
        surface_load_parameters = SurfaceLoad(
            active=[False, False, True], value=[0, 0, -200]
        )
        # moving (point) load
        moving_point_load_parameters = MovingLoad(
            origin=[1.0, 0.1, 0.0],
            load=[0.0, -10.0, 0.0],
            direction=[1.0, 0.0, -1.0],
            velocity=5.0,
            offset=3.0,
        )

        # add loads to process model parts:
        model.add_load_by_coordinates(point_load_coords, point_load_parameters, 'test_point_load')
        model.add_load_by_coordinates(line_load_coords, line_load_parameters, 'test_line_load')
        model.add_load_by_coordinates(surface_load_coords, surface_load_parameters, 'test_surface_load')
        model.add_load_by_coordinates(moving_load_coords, moving_point_load_parameters, 'test_moving_load')
        model.synchronise_geometry()

        # write dictionary for the load(s)
        kratos_io = KratosIO(ndim=model.ndim)

        test_dictionary = kratos_io.write_project_parameters_json(
            model=model,
            outputs=[],
            mesh_file_name="test_load_parameters.mdpa",
            materials_file_name=""
        )

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters_with_table.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            expected_load_parameters_json, test_dictionary
        )
