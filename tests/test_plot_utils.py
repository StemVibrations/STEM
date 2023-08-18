import pytest

from stem.model import Model
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.plot_utils import PlotUtils
from stem.geometry import Point, Line

# plots are not shown in the CI, but can be enabled for local testing
SHOW_PLOTS = False


class TestPlotUtils:
    """
    Test class for the :class:`stem.plot_utils.PlotUtils` class.

    """

    @pytest.fixture
    def create_default_2d_soil_material(self):
        """
        Create a default soil material for a 2D geometry.

        Returns:
            - :class:`stem.soil_material.SoilMaterial`: default soil material

        """
        # define soil material
        ndim = 2
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        return soil_material

    @pytest.fixture
    def create_default_3d_soil_material(self):
        """
        Create a default soil material for a 3D geometry.

        Returns:
            - :class:`stem.soil_material.SoilMaterial`: default soil material

        """
        # define soil material
        ndim = 3
        soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
        soil_material = SoilMaterial(name="soil", soil_formulation=soil_formulation, constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        return soil_material

    @staticmethod
    def create_geometry(ndim: int, material: SoilMaterial):
        """
        Create a geometry and plots it.

        Args:
            - ndim (int): dimension of the geometry
            - material (:class:`stem.soil_material.SoilMaterial`): soil material

        """
        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = material
        soil_material1.name = "soil1"

        soil_material2 = material
        soil_material2.name = "soil2"

        # create model
        model = Model(ndim)
        model.extrusion_length = [0, 0, 1]

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        # synchronise geometry
        model.synchronise_geometry()

        return model

    @pytest.mark.skipif(not SHOW_PLOTS, reason="Plotting is disabled")
    def test_plot_geometry_2D(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test the plot of a 2D geometry.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        # create geometry and plot it
        model = self.create_geometry(2, create_default_2d_soil_material)

        # show geometry
        PlotUtils.show_geometry(model.ndim, model.geometry, True, True, True, True)

    @pytest.mark.skipif(not SHOW_PLOTS, reason="Plotting is disabled")
    def test_plot_geometry_with_loose_lines_2D(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test the plot of a 2D geometry, including loose lines.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        new_points = {10: Point.create([0, 3, 0],10),
                        11: Point.create([1, 3, 0],11)}
        new_lines = {10: Line.create([10, 11],10)}

        # create geometry and plot it
        model = self.create_geometry(2, create_default_2d_soil_material)

        # add loose line to geometry
        model.geometry.points.update(new_points)
        model.geometry.lines.update(new_lines)

        # show geometry
        PlotUtils.show_geometry(model.ndim, model.geometry, True, True, True, True)

    @pytest.mark.skipif(not SHOW_PLOTS, reason="Plotting is disabled")
    def test_plot_geometry_3D(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test the plot of a 3D geometry.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        # create geometry
        model = self.create_geometry(3, create_default_3d_soil_material)

        # show geometry
        PlotUtils.show_geometry(model.ndim, model.geometry, True, True, True, True)

    @pytest.mark.skipif(not SHOW_PLOTS, reason="Plotting is disabled")
    def test_plot_geometry_with_loose_lines_3D(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test the plot of a 2D geometry, including loose lines.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        new_points = {100: Point.create([0, 3, 0], 100),
                      110: Point.create([1, 3, 0], 110)}
        new_lines = {100: Line.create([100, 110], 100)}

        # create geometry and plot it
        model = self.create_geometry(3, create_default_3d_soil_material)

        # add loose line to geometry
        model.geometry.points.update(new_points)
        model.geometry.lines.update(new_lines)

        # show geometry
        PlotUtils.show_geometry(model.ndim, model.geometry, True, True,
                                True, True)
