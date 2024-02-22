import re

import numpy as np
import pytest

from stem.load import PointLoad, LineLoad, SurfaceLoad, MovingLoad, GravityLoad, UvecLoad
from stem.boundary import DisplacementConstraint, RotationConstraint, AbsorbingBoundary
from stem.structural_material import StructuralMaterial, EulerBeam, ElasticSpringDamper, NodalConcentrated
from stem.model_part import ModelPart, BodyModelPart
from stem.solver import AnalysisType

from tests.utils import TestUtils


class TestModelPart:

    def test_get_element_name_loads(self):
        """
        Test the get_element_name method of the ModelPart class for load

        """

        # point load does not have element names
        point_load = PointLoad([True, True, True], [0, 0, 0])
        point_load_part = ModelPart("point_load_part")
        point_load_part.parameters = point_load

        assert (point_load_part.get_element_name(2, 1, AnalysisType.MECHANICAL) ==
                f"PointLoadCondition2D1N")
        assert (point_load_part.get_element_name(3, 1, AnalysisType.MECHANICAL) ==
                f"PointLoadCondition3D1N")

        # wrong point input
        with pytest.raises(ValueError, match= "Point load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            point_load_part.get_element_name(2, 1, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError, match=re.escape(r'In 2 dimensions, only [1] noded Point load elements are supported. '
                                              r'2 nodes were provided.')):
            point_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL)

        # check line load names
        line_load = LineLoad([True, True, True], [0, 0, 0])
        line_load_part = ModelPart("line_load_part")
        line_load_part.parameters = line_load

        assert line_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "LineLoadCondition2D2N"
        assert line_load_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "LineLoadCondition3D2N"
        assert line_load_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "LineLoadDiffOrderCondition2D3N"
        assert line_load_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "LineLoadCondition3D3N"

        # wrong line_load input
        with pytest.raises(ValueError, match= "Line load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            line_load_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape(r'In 2 dimensions, only [2, 3] noded Line load elements are supported. '
                                           r'4 nodes were provided.')):
            line_load_part.get_element_name(2, 4, AnalysisType.MECHANICAL)

        # check surface load names
        surface_load = SurfaceLoad([True, True, True], [0, 0, 0])
        surface_load_part = ModelPart("surface_load_part")
        surface_load_part.parameters = surface_load

        assert surface_load_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "UPwFaceLoadCondition3D3N"
        assert surface_load_part.get_element_name(3, 4, AnalysisType.MECHANICAL) == "UPwFaceLoadCondition3D4N"
        assert surface_load_part.get_element_name(3, 6, AnalysisType.MECHANICAL) == "SurfaceLoadDiffOrderCondition3D6N"
        assert surface_load_part.get_element_name(3, 8, AnalysisType.MECHANICAL) == "SurfaceLoadDiffOrderCondition3D8N"

        # wrong surface_load input
        with pytest.raises(ValueError, match= "Surface load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            surface_load_part.get_element_name(3, 3, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [3, 4, 6, 8] noded Surface load elements are '
                                           'supported. 9 nodes were provided.')):
            surface_load_part.get_element_name(3, 9, AnalysisType.MECHANICAL)

        # check moving load names
        moving_load = MovingLoad([10, 10, 10], [1, 1, 1], 1, [0, 0, 0])
        moving_load_part = ModelPart("moving_load_part")
        moving_load_part.parameters = moving_load

        assert moving_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "MovingLoadCondition2D2N"
        assert moving_load_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "MovingLoadCondition3D2N"
        assert moving_load_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "MovingLoadCondition2D3N"
        assert moving_load_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "MovingLoadCondition3D3N"

        # wrong moving_load input
        with pytest.raises(ValueError, match= "Moving load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            moving_load_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [2, 3] noded Moving load elements are supported. '
                                           '4 nodes were provided.')):
            moving_load_part.get_element_name(3, 4, AnalysisType.MECHANICAL)

        # check uvec load names
        uvec_parameters = {"load_wheel_1": -10.0, "load_wheel_2": -20.0}
        uvec_load = UvecLoad([10, 10, 10], 5, [0,0,0], [0, 2], r"sample_uvec.py", "uvec_test", uvec_parameters)

        uvec_part = ModelPart("uvec_load_part")
        uvec_part.parameters = uvec_load

        assert uvec_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "MovingLoadCondition2D2N"
        assert uvec_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "MovingLoadCondition3D2N"
        assert uvec_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "MovingLoadCondition2D3N"
        assert uvec_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "MovingLoadCondition3D3N"

        # wrong uvec_load input
        with pytest.raises(ValueError, match="UVEC load can only be applied in mechanical or mechanical groundwater "
                                             "flow analysis"):
            uvec_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [2, 3] noded UVEC load elements are supported. '
                                           '4 nodes were provided.')):
            uvec_part.get_element_name(3, 4, AnalysisType.MECHANICAL)

        # gravity load does not have element names
        gravity_load = GravityLoad([True, True, True], [0, 0, 0])
        gravity_load_part = ModelPart("moving_load_part")
        gravity_load_part.parameters = gravity_load

        assert gravity_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL) is None

    def test_get_element_name_boundaries(self):
        """
        Tests the available element names for the boundary model parts, i.e. the displacement contstraint, rotation
        constraints and absorbing boundaries
        Returns:

        """

        # displacement constraint does not have element names
        displacement_constraint = DisplacementConstraint([True, True, True],[True, True, True], [0, 0, 0])
        displacement_constraint_part = ModelPart("displacement_constraint_part")
        displacement_constraint_part.parameters = displacement_constraint

        assert displacement_constraint_part.get_element_name(2, 2, AnalysisType.MECHANICAL) is None

        # wrong displacement_constraint input
        with pytest.raises(ValueError, match= "Displacement constraint can only be applied in mechanical or mechanical "
                                              "groundwater flow analysis"):
            displacement_constraint_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('Number of dimensions 1 is not supported for Displacement constraint '
                                           'elements. Supported dimensions are [2, 3].')):
            displacement_constraint_part.get_element_name(1, 4, AnalysisType.MECHANICAL)

        # rotation constraint does not have element names
        rotation_constraint = RotationConstraint([True, True, True],[True, True, True], [0, 0, 0])
        rotation_constraint_part = ModelPart("rotation_constraint_part")
        rotation_constraint_part.parameters = rotation_constraint

        assert rotation_constraint_part.get_element_name(2, 2, AnalysisType.MECHANICAL) is None

        # wrong rotation_constraint input
        with pytest.raises(ValueError, match= "Rotation constraint can only be applied in mechanical or mechanical "
                                              "groundwater flow analysis"):
            rotation_constraint_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('Number of dimensions 1 is not supported for Rotation constraint elements. '
                                           'Supported dimensions are [2, 3].')):
            rotation_constraint_part.get_element_name(1, 4, AnalysisType.MECHANICAL)

        # check absorbing boundary names
        absorbing_boundary = AbsorbingBoundary([1, 1], 1)
        absorbing_boundary_part = ModelPart("absorbing_boundary_part")
        absorbing_boundary_part.parameters = absorbing_boundary

        assert (absorbing_boundary_part.get_element_name(2, 2, AnalysisType.MECHANICAL)
                == "UPwLysmerAbsorbingCondition2D2N")
        assert (absorbing_boundary_part.get_element_name(2, 3, AnalysisType.MECHANICAL)
                == "UPwLysmerAbsorbingCondition2D3N")
        assert (absorbing_boundary_part.get_element_name(3, 3, AnalysisType.MECHANICAL)
                == "UPwLysmerAbsorbingCondition3D3N")
        assert (absorbing_boundary_part.get_element_name(3, 4, AnalysisType.MECHANICAL)
                == "UPwLysmerAbsorbingCondition3D4N")

        # wrong absorbing_boundary input
        with pytest.raises(ValueError, match= "Absorbing boundary conditions can only be applied in mechanical or "
                                              "mechanical groundwater flow analysis"):
            absorbing_boundary_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [3, 4] noded Absorbing boundary elements are '
                                           'supported. 6 nodes were provided.')):
            absorbing_boundary_part.get_element_name(3, 6, AnalysisType.MECHANICAL)

    def test_get_element_name_bodies(self):
        """
        Tests the available element names for the body model parts, i.e. the soil and the structure.

        """

        soil = TestUtils.create_default_soil_material(2)
        soil_part = BodyModelPart("soil")
        soil_part.material = soil

        assert soil_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "UPwSmallStrainElement2D3N"
        assert soil_part.get_element_name(2, 4, AnalysisType.MECHANICAL) == "UPwSmallStrainElement2D4N"
        assert soil_part.get_element_name(2, 6, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement2D6N"
        assert soil_part.get_element_name(2, 8, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement2D8N"

        assert soil_part.get_element_name(3, 4, AnalysisType.MECHANICAL) == "UPwSmallStrainElement3D4N"
        assert soil_part.get_element_name(3, 8, AnalysisType.MECHANICAL) == "UPwSmallStrainElement3D8N"
        assert soil_part.get_element_name(3, 10, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement3D10N"
        assert soil_part.get_element_name(3, 20, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement3D20N"

        # wrong soil input
        with pytest.raises(ValueError, match= 'Analysis type AnalysisType.GROUNDWATER_FLOW is not implemented yet for '
                                              'soil material.'):
            soil_part.get_element_name(2, 3, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [4, 8, 10, 20] noded Soil elements are supported. '
                                           '6 nodes were provided.')):
            soil_part.get_element_name(3, 6, AnalysisType.MECHANICAL)

        # check beam element names
        beam = StructuralMaterial(name="beam", material_parameters=EulerBeam(2, 1, 1, 1, 1, 1))
        beam_part = BodyModelPart("beam")
        beam_part.material = beam

        assert beam_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "GeoCrBeamElementLinear2D2N"
        assert beam_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "CrLinearBeamElement3D2N"

        # wrong beam input
        with pytest.raises(ValueError, match= 'Analysis type AnalysisType.GROUNDWATER_FLOW is not implemented '
                                              'for euler beams.'):
            beam_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [2] noded Euler beam elements are supported. '
                                           '6 nodes were provided.')):
            beam_part.get_element_name(3, 6, AnalysisType.MECHANICAL)

        # check ElasticSpringDamper Element names
        spring = StructuralMaterial(name="spring", material_parameters=ElasticSpringDamper([0, 0, 0], [0, 0, 0],
                                                                                           [0, 0, 0], [0, 0, 0]))
        spring_part = BodyModelPart("spring")
        spring_part.material = spring

        assert spring_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "SpringDamperElement2D"
        assert spring_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "SpringDamperElement3D"

        with pytest.raises(ValueError, match= 'Analysis type AnalysisType.GROUNDWATER_FLOW is not implemented '
                                              'for elastic spring dampers.'):
            spring_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [2] noded Elastic spring damper elements are '
                                           'supported. 6 nodes were provided.')):
            spring_part.get_element_name(3, 6, AnalysisType.MECHANICAL)

        # check nodal concentrated element names
        nodal_concentrated = StructuralMaterial(name="nodal_concentrated",
                                                material_parameters=NodalConcentrated([0, 0, 0], 0, [0, 0, 0]))
        nodal_concentrated_part = BodyModelPart("nodal_concentrated")
        nodal_concentrated_part.material = nodal_concentrated

        assert nodal_concentrated_part.get_element_name(2, 1, AnalysisType.MECHANICAL) == "NodalConcentratedElement2D1N"
        assert nodal_concentrated_part.get_element_name(3, 1, AnalysisType.MECHANICAL) == "NodalConcentratedElement3D1N"

        with pytest.raises(ValueError, match= 'Analysis type AnalysisType.GROUNDWATER_FLOW is not implemented for nodal '
                                              'concentrated elements.'):
            nodal_concentrated_part.get_element_name(2, 1, AnalysisType.GROUNDWATER_FLOW)

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [1] noded Nodal concentrated elements are supported. '
                                           '6 nodes were provided.')):
            nodal_concentrated_part.get_element_name(3, 6, AnalysisType.MECHANICAL)

    def test_repr_method_model_part(self):
        """
        Test that the repr method for process and body model part.

        """
        # check soil part repr
        soil = TestUtils.create_default_soil_material(2)
        soil_part = BodyModelPart("soil")
        soil_part.material = soil
        expected_string = "BodyModelPart(name=soil, material=SoilMaterial)"
        assert expected_string == str(soil_part)

        # check line load part repr
        line_load = LineLoad([True, True, True], [0, 0, 0])
        line_load_part = ModelPart("line_load_part")
        line_load_part.parameters = line_load
        expected_string = "ModelPart(name=line_load_part, parameters=LineLoad)"
        assert expected_string == str(line_load_part)
