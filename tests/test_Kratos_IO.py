import json

import pprint

from stem.boundary import (
    DisplacementConstraint,
    RotationConstraint,
    AbsorbingBoundary,
    Boundary,
)
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
        nodal_results = [
            "DISPLACEMENT",
            "TOTAL_DISPLACEMENT",
            "WATER_PRESSURE",
            "VOLUME_ACCELERATION",
        ]
        # gauss point results
        gauss_point_results = [
            "GREEN_LAGRANGE_STRAIN_TENSOR",
            "ENGINEERING_STRAIN_TENSOR",
            "CAUCHY_STRESS_TENSOR",
            "TOTAL_STRESS_TENSOR",
            "VON_MISES_STRESS",
            "FLUID_FLUX_VECTOR",
            "HYDRAULIC_HEAD",
        ]
        # define output parameters
        # 1. GiD
        gid_output_parameters = GiDOutputParameters(
            output_interval=100,
            body_output=True,
            node_output=False,
            skin_output=False,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
        )

        # 2. Vtk (Paraview)
        vtk_output_parameters = VtkOutputParameters(
            file_format="binary",
            output_precision=8,
            output_control_type="step",
            output_interval=100.0,
            nodal_solution_step_data_variables=nodal_results,
            gauss_point_variables_in_elements=gauss_point_results,
        )

        # 3. Json
        json_output_parameters = JsonOutputParameters(
            time_frequency=0.002,
            output_variables=nodal_results,
            gauss_points_output_variables=gauss_point_results,
        )

        # create Load objects and store in the list
        gid_output_process = OutputProcess(
            part_name="test_gid_output",
            output_name="test_gid",
            output_parameters=gid_output_parameters,
        )
        vtk_output_process = OutputProcess(
            part_name="test_vtk_output",
            output_name="test_vtk",
            output_parameters=vtk_output_parameters,
        )

        json_output_process = OutputProcess(
            part_name="test_json_output",
            output_name="test_json",
            output_parameters=json_output_parameters,
        )
        all_outputs = [gid_output_process, vtk_output_process, json_output_process]

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

    def test_create_boundary_condition_dictionaries(self):
        """
        Test the creation of the boundary condition dictionaries for the
        ProjectParameters.json file
        """
        # define constraints

        # Displacements
        fix_displacements_parameters = DisplacementConstraint(
            active=[True, True, False],
            is_fixed=[True, True, False],
            value=[0.0, 0.0, 0.0],
        )

        # Rotations
        fix_rotations_parameters = RotationConstraint(
            active=[False, False, True],
            is_fixed=[False, False, True],
            value=[0.0, 0.0, 0.0],
        )

        # Boundary conditions
        absorbing_boundaries_parameters = AbsorbingBoundary(
            absorbing_factors=[1, 1], virtual_thickness=1000
        )

        # create Load objects and store in the list
        displacement_boundary_condition = Boundary(
            part_name="test_displacement_constraint",
            boundary_parameters=fix_displacements_parameters,
        )
        rotation_boundary_condition = Boundary(
            part_name="test_rotation_constraint",
            boundary_parameters=fix_rotations_parameters,
        )

        absorbing_boundary = Boundary(
            part_name="abs",
            boundary_parameters=absorbing_boundaries_parameters,
        )
        all_outputs = [
            displacement_boundary_condition,
            rotation_boundary_condition,
            absorbing_boundary,
        ]

        # write dictionary for the output(s)
        kratos_io = KratosIO()
        (
            test_constraint_dictionary,
            test_absorbing_bound_list
        ) = kratos_io.create_dictionaries_for_boundaries(all_outputs)

        # nest the json into the process dictionary, as it should!
        test_constraint_dictionary["loads_process_list"] = test_absorbing_bound_list,
        test_dictionary = {"processes": test_constraint_dictionary}
        # load expected dictionary from the json
        expected_load_parameters_json = json.load(
            open("tests/test_data/expected_boundary_conditions_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            test_dictionary, expected_load_parameters_json
        )
