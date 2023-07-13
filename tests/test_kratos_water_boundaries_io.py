from tests.utils import TestUtils
import json

from stem.IO.kratos_water_boundaries_io import KratosWaterBoundariesIO
from stem.water_boundaries import WaterBoundary, InterpolateLineBoundary, PhreaticMultiLineBoundary, PhreaticLine


class TestKratosWaterBoundariesIO:

    def test_create_water_boundary_process_dict(self):
        """

        Test the creation of the water boundary process dictionary for the
        ProjectParameters.json file

        """
        multi_line_boundary = PhreaticMultiLineBoundary(
            is_fixed=True,
            gravity_direction=1,
            out_of_plane_direction=2,
            water_pressure=0,
            x_coordinates=[-40.0, -11.4, 0.0, 9.0, 21.5, 95.0],
            y_coordinates=[0.44, 0.44, 3.0, 3.0, -0.5, -0.5],
            surfaces_assigment=["domain a", "domain b", "domain c"],
            specific_weight=10000.0,
        )
        water_boundary = WaterBoundary(multi_line_boundary, name="water_soils_1")
        # use the kratos io to create the dictionary
        kratos_io = KratosWaterBoundariesIO(domain="PorousDomain")
        # set the interpolation type
        interpolation_type = InterpolateLineBoundary(
            surfaces_assigment=["domain d"],
            is_fixed=True,
            gravity_direction=1,
            out_of_plane_direction=2,
        )
        water_boundary_interpolate = WaterBoundary(interpolation_type, name="water_soils_2")
        # check phreatic line
        phreatic_line = PhreaticLine(
            is_fixed=True,
            gravity_direction=1,
            out_of_plane_direction=2,
            value=0,
            first_reference_coordinate=[0.0,1.0,0.0],
            second_reference_coordinate=[1.0,0.5,0.0],
            specific_weight=9.81,
            surfaces_assigment=["domain e"]
        )
        water_boundary_phreatic_line = WaterBoundary(phreatic_line, name="water_soils_3")

        # check the dictionary
        # read the expected dictionary from the json
        with open("tests/test_data/expected_water_lines.json") as json_file:
            expected_water_boundary_json = json.load(json_file)
        # compare the dictionaries
        TestUtils.assert_dictionary_almost_equal(expected_water_boundary_json['test'][0],
                                                 kratos_io.create_water_boundary_dict(
                                                     water_boundary
                                                 ))
        TestUtils.assert_dictionary_almost_equal(expected_water_boundary_json['test'][1],
                                                 kratos_io.create_water_boundary_dict(water_boundary_interpolate))
        TestUtils.assert_dictionary_almost_equal(expected_water_boundary_json['test'][2],
                                                    kratos_io.create_water_boundary_dict(water_boundary_phreatic_line))
