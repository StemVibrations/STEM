import json

import pprint
from stem.kratos_IO import KratosIO
from stem.load import PointLoad, MovingLoad, Load
from stem.output import (
    OutputProcess,
    GiDOutputParameters,
    VtkOutputParameters,
    JsonOutputParameters,
)
from tests.utils import TestUtils


class TestKratosIO:
    def test_create_load_process_dictionary(self):
        """
        Test the creation of the load process dictionary for the
        ProjectParameters.json file
        """
        # define load(s) parameters
        # point load
        point_load_parameters = PointLoad(
            active=[True, False, True], value=[1000, 0, 0]
        )

        # moving (point) load
        moving_point_load_parameters = MovingLoad(
            origin=[0.0, 1.0, 2.0],
            load=[0.0, -10.0, 0.0],
            direction=[1.0, 0.0, -1.0],
            velocity=5.0,
            offset=3.0,
        )

        # create Load objects and store in the list
        point_load = Load(name="test_name", load_parameters=point_load_parameters)

        moving_point_load = Load(
            name="test_name_moving", load_parameters=moving_point_load_parameters
        )

        all_loads = [point_load, moving_point_load]

        # write dictionary for the load(s)
        kratos_io = KratosIO()
        test_dictionary = kratos_io.create_loads_process_dictionary(all_loads)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_load_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )

    def test_create_output_process_dictionary(self):
        """
        Test the creation of the output process dictionary for the
        ProjectParameters.json file
        """
        # Nodal results
        nodal_results1 = ["DISPLACEMENT", "TOTAL_DISPLACEMENT"]
        nodal_results2 = [
            "WATER_PRESSURE",
            "VOLUME_ACCELERATION",
        ]
        # gauss point results
        gauss_point_results1 = [
            "VON_MISES_STRESS",
            "FLUID_FLUX_VECTOR",
            "HYDRAULIC_HEAD",
        ]
        gauss_point_results2 = [
            "GREEN_LAGRANGE_STRAIN_TENSOR",
            "ENGINEERING_STRAIN_TENSOR",
            "CAUCHY_STRESS_TENSOR",
            "TOTAL_STRESS_TENSOR",
        ]
        # define output parameters
        # 1. GiD
        gid_output_parameters1 = GiDOutputParameters(
            output_interval=100,
            body_output=True,
            node_output=False,
            skin_output=False,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )

        gid_output_parameters2 = GiDOutputParameters(
            output_interval=100,
            body_output=True,
            node_output=False,
            skin_output=False,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )

        # 2. Vtk (Paraview)
        vtk_output_parameters1 = VtkOutputParameters(
            file_format="binary",
            output_precision=8,
            output_control_type="step",
            output_interval=100.0,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )
        vtk_output_parameters2 = VtkOutputParameters(
            file_format="binary",
            output_precision=8,
            output_control_type="step",
            output_interval=100.0,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )
        # 3. Json
        json_output_parameters1 = JsonOutputParameters(
            time_frequency=0.002,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )
        json_output_parameters2 = JsonOutputParameters(
            time_frequency=0.002,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )

        # create Load objects and store in the list
        gid_output_process1 = OutputProcess(
            part_name="test_gid_output",
            output_path=r"dir_test/test_gid1",
            output_parameters=gid_output_parameters1,
        )
        gid_output_process2 = OutputProcess(
            part_name="test_gid_output",
            output_path=r"dir_test\test_gid2",
            output_parameters=gid_output_parameters2,
        )
        vtk_output_process1 = OutputProcess(
            part_name="test_vtk_output",
            output_path=r"dir_test/test_vtk1",
            output_parameters=vtk_output_parameters1,
        )
        vtk_output_process2 = OutputProcess(
            part_name="test_vtk_output",
            output_path=r"dir_test\test_vtk2",
            output_parameters=vtk_output_parameters2,
        )

        json_output_process1 = OutputProcess(
            part_name="test_json_output1",
            output_path=r"dir_test",
            output_parameters=json_output_parameters1,
        )

        json_output_process2 = OutputProcess(
            part_name="test_json_output2",
            output_path=r"dir_test",
            output_parameters=json_output_parameters2,
        )
        all_outputs = [
            gid_output_process1,
            vtk_output_process1,
            json_output_process1,
            gid_output_process2,
            vtk_output_process2,
            json_output_process2
        ]

        # write dictionary for the output(s)
        kratos_io = KratosIO()
        test_dictionary, test_json = kratos_io.create_output_process_dictionary(
            all_outputs
        )

        # nest the json into the process dictionary, as it should!
        test_dictionary["processes"] = test_json

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_output_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )
