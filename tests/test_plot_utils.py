import pytest

from stem.model import Model
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.plot_utils import PlotUtils

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

    def create_geometry_and_plot(self, ndim, material):
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

        # show geometry in 3D
        PlotUtils.show_geometry(model.ndim, model.geometry, True, True, True, True)

    @pytest.mark.skipif(not SHOW_PLOTS, reason="Plotting is disabled")
    def test_plot_geometry_2D(self, create_default_2d_soil_material):
        """
        Test the plot of a 2D geometry.

        Args:
            create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        Returns:

        """

        # create geometry and plot it
        self.create_geometry_and_plot(2, create_default_2d_soil_material)

    @pytest.mark.skipif(not SHOW_PLOTS, reason="Plotting is disabled")
    def test_plot_geometry_3D(self, create_default_3d_soil_material):
        """
        Test the plot of a 3D geometry.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        # create geometry and plot it
        self.create_geometry_and_plot(3, create_default_3d_soil_material)