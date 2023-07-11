from typing import Tuple

import pytest
from gmsh_utils.gmsh_IO import GmshIO

from stem.model import *
from stem.geometry import *


class TestGeometry:

    @pytest.fixture
    def expected_geo_data_0D(self):
        """
        Expected geometry data for a 0D geometry group. The group is a geometry of a point

        Returns:
            - expected_geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io
        """
        expected_points = {1: [0, 0, 0], 2: [0.5, 0, 0]}
        return {"points": expected_points}

    @pytest.fixture
    def expected_geometry_single_layer_2D(self):

        geometry= Geometry()

        geometry.points = [Point.create([0,0,0], 1),
                           Point.create([1,0,0], 2),
                           Point.create([1,1,0], 3),
                           Point.create([0,1,0], 4)]

        geometry.lines = [Line.create([1,2], 1),
                          Line.create([2,3], 2),
                          Line.create([3,4], 3),
                          Line.create([4,1], 4)]

        geometry.surfaces = [Surface.create([1,2,3,4], 1)]

        geometry.volumes = []

        return geometry


    @pytest.fixture
    def expected_geometry_two_layers_2D(self):

        geometry_1 = Geometry()

        geometry_1.points = [Point.create([0, 0, 0], 1),
                             Point.create([1, 0, 0], 2),
                             Point.create([1, 1, 0], 3),
                             Point.create([0, 1, 0], 4)]

        geometry_1.lines = [Line.create([1, 2], 1),
                            Line.create([2, 3], 2),
                            Line.create([3, 4], 3),
                            Line.create([4, 1], 4)]

        geometry_1.surfaces = [Surface.create([1, 2, 3, 4], 1)]

        geometry_1.volumes = []

        geometry_2 = Geometry()
        geometry_2.points = [Point.create([1, 1, 0], 3),
                             Point.create([0, 1, 0], 4),
                             Point.create([0, 2, 0], 5),
                             Point.create([1, 2, 0], 6)]

        geometry_2.lines = [Line.create([3, 4], 3),
                            Line.create([4, 5], 5),
                            Line.create([5, 6], 6),
                            Line.create([6, 3], 7)]

        geometry_2.surfaces = [Surface.create([3, 5, 6, 7], 2)]

        geometry_2.volumes = []

        return geometry_1, geometry_2

    @pytest.fixture
    def create_default_2d_soil_material(self):
        # define soil material
        ndim=2
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        return soil_material

    def test_add_single_soil_layer_2D(self, expected_geometry_single_layer_2D: Geometry,
                                      create_default_2d_soil_material: SoilMaterial):
        """
        Test if a single soil layer is added correctly to the model in a 2D space. A single soil layer is generated
        and a single soil material is created and added to the model.

        Args:
            - expected_geometry_single_layer_2D (Geometry): expected geometry of the model
            - create_default_2d_soil_material (SoilMaterial): default soil material

        """

        ndim = 2

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        # define soil material
        soil_material = create_default_2d_soil_material

        # create model
        model = Model()
        model.ndim = ndim

        # add soil layer
        model.add_soil_layer(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material

        # check if geometry is added correctly
        generated_geometry = model.body_model_parts[0].geometry
        expected_geometry = expected_geometry_single_layer_2D

        # check if points are added correctly
        for generated_point, expected_point in zip(generated_geometry.points, expected_geometry.points):
            assert generated_point.id == expected_point.id
            assert pytest.approx(generated_point.coordinates) == expected_point.coordinates

        # check if lines are added correctly
        for generated_line, expected_line in zip(generated_geometry.lines, expected_geometry.lines):
            assert generated_line.id == expected_line.id
            assert generated_line.point_ids == expected_line.point_ids

        # check if surfaces are added correctly
        for generated_surface, expected_surface in zip(generated_geometry.surfaces, expected_geometry.surfaces):
            assert generated_surface.id == expected_surface.id
            assert generated_surface.line_ids == expected_surface.line_ids


    def test_add_multiple_soil_layers_2D(self, expected_geometry_two_layers_2D: Tuple[Geometry, Geometry],
                                         create_default_2d_soil_material: SoilMaterial):
        """
        Test if multiple soil layers are added correctly to the model in a 2D space. Multiple soil layers are generated
        and multiple soil materials are created and added to the model.

        """

        ndim = 2

        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_2d_soil_material
        soil_material1.name = "soil1"

        soil_material2 = create_default_2d_soil_material
        soil_material2.name = "soil2"

        # create model
        model = Model()
        model.ndim = ndim

        # add soil layers
        model.add_soil_layer(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer(layer2_coordinates, soil_material2, "layer2")

        # check if layers are added correctly
        assert len(model.body_model_parts) == 2
        assert model.body_model_parts[0].name == "layer1"
        assert model.body_model_parts[0].material == soil_material1
        assert model.body_model_parts[1].name == "layer2"
        assert model.body_model_parts[1].material == soil_material2

        # check if geometry is added correctly for each layer
        for i in range(len(model.body_model_parts)):
            generated_geometry = model.body_model_parts[i].geometry
            expected_geometry = expected_geometry_two_layers_2D[i]

            # check if points are added correctly
            for generated_point, expected_point in zip(generated_geometry.points, expected_geometry.points):
                assert generated_point.id == expected_point.id
                assert pytest.approx(generated_point.coordinates) == expected_point.coordinates

            # check if lines are added correctly
            for generated_line, expected_line in zip(generated_geometry.lines, expected_geometry.lines):
                assert generated_line.id == expected_line.id
                assert generated_line.point_ids == expected_line.point_ids

            # check if surfaces are added correctly
            for generated_surface, expected_surface in zip(generated_geometry.surfaces, expected_geometry.surfaces):
                assert generated_surface.id == expected_surface.id
                assert generated_surface.line_ids == expected_surface.line_ids




