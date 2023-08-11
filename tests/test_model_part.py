import re

import pytest

from stem.load import PointLoad, LineLoad, SurfaceLoad, MovingLoad, GravityLoad
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

        assert point_load_part.get_element_name(2, 1, AnalysisType.MECHANICAL) is None
        assert point_load_part.get_element_name(3, 1, AnalysisType.MECHANICAL) is None

        # wrong point input
        with pytest.raises(ValueError, match= "Point load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            assert point_load_part.get_element_name(2, 1, AnalysisType.GROUNDWATER_FLOW) is None

        # wrong ndim nnodes combination
        with pytest.raises(ValueError, match=re.escape(r'In 2 dimensions, only [1] noded Point load elements are supported. '
                                              r'2 nodes were provided.')):
            assert point_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL) is None

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
            assert line_load_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW) is None

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape(r'In 2 dimensions, only [2, 3] noded Line load elements are supported. '
                                           r'4 nodes were provided.')):
            assert line_load_part.get_element_name(2, 4, AnalysisType.MECHANICAL) is None

        # check surface load names
        surface_load = SurfaceLoad([True, True, True], [0, 0, 0])
        surface_load_part = ModelPart("surface_load_part")
        surface_load_part.parameters = surface_load

        assert surface_load_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "UPwFaceLoadCondition3D3N"
        assert surface_load_part.get_element_name(3, 4, AnalysisType.MECHANICAL) == "UPwFaceLoadCondition3D4N"
        assert surface_load_part.get_element_name(3, 6, AnalysisType.MECHANICAL) == "SurfaceLoadDiffOrderCondition3D6N"
        assert surface_load_part.get_element_name(3, 8, AnalysisType.MECHANICAL) == "SurfaceLoadDiffOrderCondition3D8N"

        # wrong line_load input
        with pytest.raises(ValueError, match= "Surface load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            assert surface_load_part.get_element_name(3, 3, AnalysisType.GROUNDWATER_FLOW) is None

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [3, 4, 6, 8] noded Surface load elements are '
                                           'supported. 9 nodes were provided.')):
            assert surface_load_part.get_element_name(3, 9, AnalysisType.MECHANICAL) is None

        # check moving load names
        moving_load = MovingLoad([10, 10, 10], [1, 1, 1], 1, [0, 0, 0])
        moving_load_part = ModelPart("moving_load_part")
        moving_load_part.parameters = moving_load

        assert moving_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "MovingLoadCondition2D2N"
        assert moving_load_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "MovingLoadCondition3D2N"
        assert moving_load_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "MovingLoadCondition2D3N"
        assert moving_load_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "MovingLoadCondition3D3N"

        # wrong line_load input
        with pytest.raises(ValueError, match= "Moving load can only be applied in mechanical or mechanical groundwater "
                                              "flow analysis"):
            assert moving_load_part.get_element_name(2, 2, AnalysisType.GROUNDWATER_FLOW) is None

        # wrong ndim nnodes combination
        with pytest.raises(ValueError,
                           match=re.escape('In 3 dimensions, only [2, 3] noded Moving load elements are supported. '
                                           '4 nodes were provided.')):
            assert moving_load_part.get_element_name(3, 4, AnalysisType.MECHANICAL) is None

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

        # rotation constraint does not have element names
        rotation_constraint = RotationConstraint([True, True, True],[True, True, True], [0, 0, 0])
        rotation_constraint_part = ModelPart("rotation_constraint_part")
        rotation_constraint_part.parameters = rotation_constraint

        assert rotation_constraint_part.get_element_name(2, 2, AnalysisType.MECHANICAL) is None

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

    def test_get_element_name_bodies(self):
        """
        Tests the available element names for the body model parts, i.e. the soil and the structure.

        """

        soil = TestUtils.create_default_soil_material(2)
        soil_part = BodyModelPart("soil")
        soil_part.material = soil

        #todo 2n element not allowed
        assert soil_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "UPwSmallStrainElement2D3N"
        assert soil_part.get_element_name(2, 4, AnalysisType.MECHANICAL) == "UPwSmallStrainElement2D4N"
        assert soil_part.get_element_name(2, 6, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement2D6N"
        assert soil_part.get_element_name(2, 8, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement2D8N"

        assert soil_part.get_element_name(3, 4, AnalysisType.MECHANICAL) == "UPwSmallStrainElement3D4N"
        assert soil_part.get_element_name(3, 8, AnalysisType.MECHANICAL) == "UPwSmallStrainElement3D8N"
        assert soil_part.get_element_name(3, 10, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement3D10N"
        assert soil_part.get_element_name(3, 20, AnalysisType.MECHANICAL) == "SmallStrainUPwDiffOrderElement3D20N"

        # check beam element names
        beam = StructuralMaterial(name="beam", material_parameters=EulerBeam(2, 1, 1, 1, 1, 1))
        beam_part = BodyModelPart("beam")
        beam_part.material = beam

        assert beam_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "GeoCrBeamElement2D2N"
        assert beam_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "GeoCrBeamElement3D2N"

        # check ElasticSpringDamper Element names
        spring = StructuralMaterial(name="spring", material_parameters=ElasticSpringDamper([0, 0, 0], [0, 0, 0],
                                                                                           [0, 0, 0], [0, 0, 0]))
        spring_part = BodyModelPart("spring")
        spring_part.material = spring

        assert spring_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "SpringDamperElement2D"
        assert spring_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "SpringDamperElement3D"

        # check nodal concentrated element names
        nodal_concentrated = StructuralMaterial(name="nodal_concentrated",
                                                material_parameters=NodalConcentrated([0, 0, 0], 0, [0, 0, 0]))
        nodal_concentrated_part = BodyModelPart("nodal_concentrated")
        nodal_concentrated_part.material = nodal_concentrated

        assert nodal_concentrated_part.get_element_name(2, 1, AnalysisType.MECHANICAL) == "NodalConcentratedElement2D1N"
        assert nodal_concentrated_part.get_element_name(3, 1, AnalysisType.MECHANICAL) == "NodalConcentratedElement3D1N"



