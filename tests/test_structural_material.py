import pytest

from stem.structural_material import *


class TestStructuralMaterial:

    def test_raise_errors_for_material_parameters(self):
        """
        Tests that errors are raised when the structural material parameters are not valid.

        """

        # create 3d euler beam without I22, which is not allowed
        pytest.raises(ValueError,
                      EulerBeam,
                      ndim=3,
                      DENSITY=1.0,
                      YOUNG_MODULUS=1.0,
                      POISSON_RATIO=0.2,
                      CROSS_AREA=1.0,
                      I33=1,
                      TORSIONAL_INERTIA=1)

        # create 3d euler beam without torsional inertia, which is not allowed
        pytest.raises(ValueError,
                      EulerBeam,
                      ndim=3,
                      DENSITY=1.0,
                      YOUNG_MODULUS=1.0,
                      POISSON_RATIO=0.2,
                      CROSS_AREA=1.0,
                      I33=1,
                      I22=1)

    def test_get_property_in_structural_material(self):
        """
        Check that properties are correctly returned in soil material properties
        """
        beam_material = StructuralMaterial(
            "euler_beam",
            EulerBeam(ndim=3,
                      DENSITY=1.0,
                      YOUNG_MODULUS=1e5,
                      POISSON_RATIO=0.2,
                      CROSS_AREA=1.0,
                      I33=1,
                      I22=0.1,
                      TORSIONAL_INERTIA=1))

        assert beam_material.get_property_in_material("YOUNG_MODULUS") == 1e5
        assert beam_material.get_property_in_material("POISSON_RATIO") == 0.2
        assert beam_material.get_property_in_material("CROSS_AREA") == 1.0

        msg = "Property YOUNGS_MODULUS is not one of the parameters of the structural material"
        with pytest.raises(ValueError, match=msg):
            beam_material.get_property_in_material("YOUNGS_MODULUS")

    def test_embedded_beam(self):
        """
        Check the default effective shear areas, the validation and the element names of the embedded beam
        """
        beam_2d = EmbeddedBeam(ndim=2, YOUNG_MODULUS=1e5, POISSON_RATIO=0.2, DENSITY=1.0, CROSS_AREA=2.0, I33=1.0)
        assert beam_2d.AREA_EFFECTIVE_Y == 2.0
        assert beam_2d.AREA_EFFECTIVE_Z is None

        beam_3d = EmbeddedBeam(ndim=3, YOUNG_MODULUS=1e5, POISSON_RATIO=0.2, DENSITY=1.0, CROSS_AREA=2.0, I33=1.0,
                               I22=1.0, AREA_EFFECTIVE_Y=1.5)
        assert beam_3d.AREA_EFFECTIVE_Y == 1.5
        assert beam_3d.AREA_EFFECTIVE_Z == 2.0

        with pytest.raises(ValueError, match=r"The second moment of area around the y-axis \(I22\) is not defined."):
            EmbeddedBeam(ndim=3, YOUNG_MODULUS=1e5, POISSON_RATIO=0.2, DENSITY=1.0, CROSS_AREA=2.0, I33=1.0)

        assert EmbeddedBeam.get_element_name(2, 2, AnalysisType.MECHANICAL) == "GeoEmbeddedBeamElement2D2N"
        assert (EmbeddedBeam.get_element_name(3, 2, AnalysisType.MECHANICAL_GROUNDWATER_FLOW) ==
                "GeoEmbeddedBeamElement3D2N")

        with pytest.raises(ValueError, match="Analysis type AnalysisType.GROUNDWATER_FLOW is not implemented for "
                           "embedded beams."):
            EmbeddedBeam.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

    def test_embedded_beam_interface(self):
        """
        Check the validation and the element names of the embedded beam interface
        """
        interface = EmbeddedBeamInterface(INTERFACE_SHEAR_STIFFNESS=1e5, INTERFACE_NORMAL_STIFFNESS=2e5)
        assert not interface.has_shear_strength
        assert EmbeddedBeamInterface(1e5, 2e5, GEO_FRICTION_ANGLE=30).has_shear_strength

        assert (EmbeddedBeamInterface.get_element_name(2, 2, AnalysisType.MECHANICAL) ==
                "GeoEmbeddedBeamInterfaceElement2D2N")
        assert (EmbeddedBeamInterface.get_element_name(3, 2, AnalysisType.MECHANICAL) ==
                "GeoEmbeddedBeamInterfaceElement3D2N")

        with pytest.raises(ValueError, match="The interface shear and normal stiffness of an embedded beam interface "
                           "must be positive."):
            EmbeddedBeamInterface(INTERFACE_SHEAR_STIFFNESS=0, INTERFACE_NORMAL_STIFFNESS=2e5)
        with pytest.raises(ValueError, match="The cohesion of an embedded beam interface must not be negative."):
            EmbeddedBeamInterface(1e5, 2e5, GEO_COHESION=-1)
        with pytest.raises(ValueError, match=r"The friction angle of an embedded beam interface must be in the range "
                           r"\[0, 90\) degrees."):
            EmbeddedBeamInterface(1e5, 2e5, GEO_FRICTION_ANGLE=90)
        with pytest.raises(ValueError, match="The diameter of an embedded beam interface must be positive."):
            EmbeddedBeamInterface(1e5, 2e5, DIAMETER=0)
