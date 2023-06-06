
import json

from stem.kratos_IO import KratosIO
from stem.load import (PointLoad, Load)

from tests.utils import TestUtils


class TestKratosIO:

    def test_create_load_process_dictionary(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file
        """
        # define load(s) parameters
        point_load_parameters = PointLoad(active=[True, True, True],
                                          value=[1000, 0, 0] )

        # create Load objects and store in the list
        point_load = Load(name="test_name", load_parameters=point_load_parameters)

        all_loads = [point_load]

        # write dictionary for the load(s)
        kratos_io = KratosIO()
        test_dictionary = kratos_io.create_loads_process_dictionary(
            all_loads
        )

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(test_dictionary,
                                                 expected_load_parameters_json)


