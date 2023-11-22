import json
import re

import pytest

from stem.IO.kratos_output_io import KratosOutputsIO
from stem.output import *
from tests.utils import TestUtils


class TestKratosOutputsIO:


    def test_check_validation_of_requested_output(self):
        """
        Test the creation of the output process dictionary for the
        ProjectParameters.json file
        """

        # Nodal results
        nodal_results1 = [NodalOutput.DISPLACEMENT, "DISPLACEMENT"]

        nodal_results2 = ["DISPLLLACEMENT"]

        # Gauss point results
        gauss_point_results1 = [
            GaussPointOutput.GREEN_LAGRANGE_STRAIN_TENSOR,
            GaussPointOutput.YOUNG_MODULUS,
            "YOUNG_MODULUS"
        ]
        gauss_point_results2 = [
            "YOUNGS_MODULUS"
        ]
        # define output parameters
        # 1. GiD
        GiDOutputParameters(
            file_format="binary",
            output_interval=100,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )

        # incorrect definition nodal outputs
        msg = ("Incorrect requested output for Nodal outputs:\n"
               "DISPLLLACEMENT. Check the available nodal outputs "
               "in the Enum NodalOutput.")
        with pytest.raises(ValueError, match=msg):
            GiDOutputParameters(
                file_format="ascii",
                output_interval=100,
                nodal_results=nodal_results2,
                gauss_point_results=[],
            )

        # incorrect definition gauss outputs
        msg = ("Incorrect requested output for Gauss point outputs:\n"
               "YOUNGS_MODULUS. Check the available gauss point outputs "
               "in the Enum GaussPointOutput.")
        with pytest.raises(ValueError, match=msg):
            GiDOutputParameters(
                file_format="ascii",
                output_interval=100,
                nodal_results=[],
                gauss_point_results=gauss_point_results2,
            )

        # 2. Vtk (Paraview)
        VtkOutputParameters(
            file_format="binary",
            output_precision=8,
            output_interval=100.0,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )

        # incorrect definition nodal outputs
        msg = ("Incorrect requested output for Nodal outputs:\n"
               "DISPLLLACEMENT. Check the available nodal outputs "
               "in the Enum NodalOutput.")
        with pytest.raises(ValueError, match=msg):
            VtkOutputParameters(
                file_format="ascii",
                output_precision=8,
                output_control_type="step",
                output_interval=100.0,
                nodal_results=nodal_results2,
                gauss_point_results=[],
            )

        # incorrect definition gauss outputs
        msg = ("Incorrect requested output for Gauss point outputs:\n"
               "YOUNGS_MODULUS. Check the available gauss point outputs "
               "in the Enum GaussPointOutput.")
        with pytest.raises(ValueError, match=msg):
            VtkOutputParameters(
                file_format="ascii",
                output_precision=8,
                output_control_type="step",
                output_interval=100.0,
                nodal_results=[],
                gauss_point_results=gauss_point_results2,
            )

        # 3. Json
        JsonOutputParameters(
            output_interval=0.002,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )

        msg = ("Incorrect requested output for Nodal outputs:\n"
               "DISPLLLACEMENT. Check the available nodal outputs "
               "in the Enum NodalOutput.")
        with pytest.raises(ValueError, match=msg):
            JsonOutputParameters(
                output_interval=0.002,
                nodal_results=nodal_results2,
                gauss_point_results=[],
            )

        # incorrect definition gauss outputs
        msg = ("Incorrect requested output for Gauss point outputs:\n"
               "YOUNGS_MODULUS. Check the available gauss point outputs "
               "in the Enum GaussPointOutput.")
        with pytest.raises(ValueError, match=msg):
            JsonOutputParameters(
                output_interval=0.002,
                nodal_results=[],
                gauss_point_results=gauss_point_results2,
            )


    def test_create_output_process_dictionary(self):
        """
        Test the creation of the output process dictionary for the
        ProjectParameters.json file
        """

        # Nodal results
        nodal_results1 = [NodalOutput.DISPLACEMENT, NodalOutput.TOTAL_DISPLACEMENT]
        nodal_results2 = [NodalOutput.WATER_PRESSURE, NodalOutput.VOLUME_ACCELERATION]

        # Gauss point results
        gauss_point_results1 = [
            GaussPointOutput.VON_MISES_STRESS,
            GaussPointOutput.FLUID_FLUX_VECTOR,
            GaussPointOutput.HYDRAULIC_HEAD,
        ]
        gauss_point_results2 = [
            GaussPointOutput.GREEN_LAGRANGE_STRAIN_TENSOR,
            GaussPointOutput.ENGINEERING_STRAIN_TENSOR,
            GaussPointOutput.CAUCHY_STRESS_TENSOR,
            GaussPointOutput.TOTAL_STRESS_TENSOR,
            GaussPointOutput.GREEN_LAGRANGE_STRAIN_VECTOR
        ]
        # define output parameters
        # 1. GiD
        gid_output_parameters1 = GiDOutputParameters(
            file_format="binary",
            output_interval=100,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )

        gid_output_parameters2 = GiDOutputParameters(
            file_format="ascii",
            output_interval=100,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )

        gid_output_parameters3 = GiDOutputParameters(
            file_format="hdf5",
            output_interval=100,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )

        # 2. Vtk (Paraview)
        vtk_output_parameters1 = VtkOutputParameters(
            file_format="binary",
            output_precision=8,
            output_interval=100.0,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )
        vtk_output_parameters2 = VtkOutputParameters(
            file_format="ascii",
            output_precision=8,
            output_control_type="step",
            output_interval=100.0,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )
        # 3. Json
        json_output_parameters1 = JsonOutputParameters(
            output_interval=0.002,
            nodal_results=nodal_results1,
            gauss_point_results=gauss_point_results1,
        )
        json_output_parameters2 = JsonOutputParameters(
            output_interval=0.002,
            nodal_results=nodal_results2,
            gauss_point_results=gauss_point_results2,
        )

        # create Load objects and store in the list
        gid_output_process1 = Output(
            part_name="test_gid_output",
            output_name=r"test_gid1",
            output_parameters=gid_output_parameters1,
        )
        gid_output_process2 = Output(
            part_name="test_gid_output",
            output_dir=r"dir_test",
            output_name=r"test_gid2",
            output_parameters=gid_output_parameters2,
        )
        gid_output_process3 = Output(
            part_name="test_gid_output",
            output_dir=r"dir_test",
            output_name=r"test_gid3",
            output_parameters=gid_output_parameters3,
        )
        vtk_output_process1 = Output(
            part_name="test_vtk_output",
            output_parameters=vtk_output_parameters1,
        )
        vtk_output_process2 = Output(
            part_name="test_vtk_output",
            output_dir=r"test_vtk1",
            output_parameters=vtk_output_parameters2,
        )

        json_output_process1 = Output(
            part_name="test_json_output1",
            output_name="test_json_output1",
            output_parameters=json_output_parameters1,
        )

        json_output_process2 = Output(
            part_name="test_json_output2",
            output_name="test_json_output2",
            output_dir="dir_test",
            output_parameters=json_output_parameters2,
        )
        all_outputs = [
            gid_output_process1,
            vtk_output_process1,
            json_output_process1,
            gid_output_process2,
            vtk_output_process2,
            json_output_process2,
            gid_output_process3
        ]

        # write dictionary for the output(s)
        kratos_outputs_io = KratosOutputsIO(domain="PorousDomain")
        test_output = kratos_outputs_io.create_output_process_dictionary(all_outputs)

        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_output_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            expected_load_parameters_json, test_output
        )
