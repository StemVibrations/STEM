import json

import numpy as np

from stem.IO.kratos_io import KratosIO
from stem.boundary import *
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from tests.utils import TestUtils


class TestKratosBoundariesIO:

    def test_create_boundary_condition_dictionaries(self):
        """
        Test the creation of the boundary condition dictionaries for the
        ProjectParameters.json file
        """

        # initialise model
        model = Model(ndim=2)
        # define a simple square

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        # define soil material
        soil_formulation = OnePhaseSoil(2, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        # define tables
        _time = np.array([0, 1, 2, 3, 4, 5])

        _value1 = np.array([0, 5, 10, 5, 0, 0])
        table1 = Table(times=_time, values=_value1)

        _value2 = np.array([0, -5, 5, -5, 0, 0])
        table2 = Table(times=_time, values=_value2)

        # Displacements
        fix_displacements_parameters = DisplacementConstraint(
            active=[True, True, False],
            is_fixed=[True, True, False],
            value=[0.0, table1, 0.0],
        )

        # Rotations
        fix_rotations_parameters = RotationConstraint(
            active=[False, False, True],
            is_fixed=[False, False, True],
            value=[table2, 0.0, 0.0],
        )

        # Absorbing boundaries
        absorbing_boundaries_parameters = AbsorbingBoundary(
            absorbing_factors=[1.0, 1.0], virtual_thickness=1000.0
        )

        model.project_parameters = TestUtils.create_default_solver_settings()

        # add dummy soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "dummy_soil")
        # add loads to process model parts:
        model.add_boundary_condition_by_geometry_ids(1, [1], fix_displacements_parameters,
                                                     'test_displacement_constraint')
        model.add_boundary_condition_by_geometry_ids(1, [2], fix_rotations_parameters, 'test_rotation_constraint')
        model.add_boundary_condition_by_geometry_ids(1, [3], absorbing_boundaries_parameters,
                                                     'test_absorbing_boundaries')
        model.synchronise_geometry()
        # write dictionary for the load(s)
        kratos_io = KratosIO(ndim=model.ndim)

        test_dictionary = kratos_io._KratosIO__write_project_parameters_json(
            model=model,
            mesh_file_name="test_load_parameters.mdpa",
            materials_file_name=""
        )

        # load expected dictionary from the json
        expected_boundary_parameters_json = json.load(
            open("tests/test_data/expected_boundary_conditions_parameters.json")
        )

        # assert the objects to be equal
        TestUtils.assert_dictionary_almost_equal(
            expected_boundary_parameters_json, test_dictionary
        )