import pytest
from pathlib import Path
import codecs
from bs4 import BeautifulSoup

from stem.model import Model
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw
from stem.plot_utils import PlotUtils

from gmsh_utils import gmsh_IO


class TestPlotUtils:
    """
    Test class for the :class:`stem.plot_utils.PlotUtils` class.

    """

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
    def create_default_2d_soil_material(self) -> SoilMaterial:
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
    def create_default_3d_soil_material(self) -> SoilMaterial:
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
    def create_geometry_plot_and_assert(ndim: int, material: SoilMaterial) -> None:
        """
        Create a geometry and plots it.

        Args:
            - ndim (int): dimension of the model
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
        model.extrusion_length = 1

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        # synchronise geometry
        model.synchronise_geometry()

        # create figure
        fig = PlotUtils.create_geometry_figure(model.ndim, model.geometry, True, True, True, True)

        # save to html for testing
        fig.write_html(f"tests/generated_geometry_{ndim}D.html")

        with codecs.open(f"tests/generated_geometry_{ndim}D.html", "r", encoding="utf-8") as f:
            generated_geometry = BeautifulSoup(f.read()).prettify()
            generated_geometry = generated_geometry.splitlines()

            # only compare the actual plotly object within the html file
            generated_geometry = generated_geometry[22].split(",")
            generated_geometry.pop(0)

        with codecs.open(f"tests/test_data/expected_geometry_{ndim}D.html", "r", encoding="utf-8") as f:
            expected_geometry = BeautifulSoup(f.read()).prettify()
            expected_geometry = expected_geometry.splitlines()

            # only compare the actual plotly object within the html file
            expected_geometry = expected_geometry[22].split(",")
            expected_geometry.pop(0)

        # compare the actual plotly object within the html file
        for generated_line, expected_line in zip(generated_geometry, expected_geometry):
            assert generated_line == expected_line

        # remove generated file
        Path(f"tests/generated_geometry_{ndim}D.html").unlink()

    def test_plot_geometry_2D(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test the plot of a 2D geometry.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        # create geometry and plot it
        self.create_geometry_plot_and_assert(2, create_default_2d_soil_material)

    def test_plot_geometry_3D(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test the plot of a 3D geometry.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        # create geometry and plot it
        self.create_geometry_plot_and_assert(3, create_default_3d_soil_material)
