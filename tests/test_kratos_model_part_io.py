import json
import numpy.testing as npt
from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.load import LineLoad
from stem.model import Model
from stem.model_part import *
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from gmsh_utils import gmsh_IO

import pytest

from tests.utils import TestUtils
from stem.IO.kratos_model_part_io import KratosModelPartIO


class TestKratosModelPartIO:

    @pytest.fixture(autouse=True)
    def close_gmsh(self):
        """
        Initializer to close gmsh if it was not closed before. In case a test fails, the destroyer method is not called
        on the Model object and gmsh keeps on running. Therefore, nodes, lines, surfaces and volumes ids are not
        reset to one. This causes also the next test after the failed one to fail as well, which has nothing to do
        the test itself.

        Returns:
            - None

        """
        gmsh_IO.GmshIO().finalize_gmsh()


    @pytest.fixture
    def create_default_2d_model_and_mesh(self):
        """
        Sets expected geometry data for a 3D geometry group. The group is a geometry of a cube.

        Returns:
            - :class:`stem.geometry.Geometry`: geometry of a 3D cube
        """
        ndim=2
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        load_coordinates = [layer_coordinates[2], layer_coordinates[3]]
        # define soil material
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())

        # define load properties
        line_load = LineLoad(active=[False, True, False], value=[0, -20, 0])

        # create model
        model = Model(ndim)

        # add soil layer and line load and mesh them
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")
        model.add_load_by_coordinates(load_coordinates, line_load, "load1")
        model.synchronise_geometry()

        model.set_mesh_size(1)
        model.generate_mesh()

        return model

    def test_create_submodelpart_text(self, create_default_2d_model_and_mesh:Model):
        """
        Test the creation of the mdpa text of a model part

        Args:
            - create_default_2d_model_and_mesh (:class:`stem.model.Model`): the default model to use in testing
        """
        # load the default 2D model
        model = create_default_2d_model_and_mesh

        body_model_part_to_write = model.body_model_parts[0]
        process_model_part_to_write = model.process_model_parts[0]

        # IO object
        model_part_io = KratosModelPartIO(ndim=model.ndim, domain="PorousDomain")

        # generate text block body model part: soil1
        actual_text_body = model_part_io.write_submodelpart_body_model_part(
            body_model_part=body_model_part_to_write
        )
        # define expected block text
        expected_text_body = ['', 'Begin SubModelPart soil1', '  Begin SubModelPartTables', '  End SubModelPartTables',
                              '  Begin SubModelPartNodes', '  1', '  2', '  3', '  4', '  5', '  End SubModelPartNodes',
                              '  Begin SubModelPartElements', '  2', '  3', '  4', '  5',
                              '  End SubModelPartElements', '', 'End SubModelPart', '']
        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_body, desired=expected_text_body)

        # generate text block body model part: soil1
        actual_text_load = model_part_io.write_submodelpart_process_model_part(
            process_model_part=process_model_part_to_write
        )
        # define expected block text
        expected_text_load = ['', 'Begin SubModelPart load1', '  Begin SubModelPartTables',
                              '  End SubModelPartTables', '  Begin SubModelPartNodes', '  3', '  4',
                              '  End SubModelPartNodes', '  Begin SubModelPartConditions', '  1',
                              '  End SubModelPartConditions', '', 'Begin SubModelPart', '']

        # assert the objects to be equal
        npt.assert_equal(actual=actual_text_load, desired=expected_text_load)

