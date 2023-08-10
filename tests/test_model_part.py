
import pytest


from stem.load import PointLoad, LineLoad, SurfaceLoad, MovingLoad, GravityLoad
from stem.model_part import ModelPart, BodyModelPart

from stem.solver import AnalysisType

class TestModelPart:

    def test_get_element_name(self):

        point_load = PointLoad([True, True, True], [0, 0, 0])
        point_load_part = ModelPart("point_load_part")
        point_load_part.parameters = point_load

        assert point_load_part.get_element_name(2, 1, AnalysisType.MECHANICAL) is None
        assert point_load_part.get_element_name(3, 1, AnalysisType.MECHANICAL) is None


        line_load = LineLoad([True, True, True], [0, 0, 0])
        line_load_part = ModelPart("line_load_part")
        line_load_part.parameters = line_load

        assert line_load_part.get_element_name(2, 2, AnalysisType.MECHANICAL) == "LineLoadCondition2D2N"
        assert line_load_part.get_element_name(3, 2, AnalysisType.MECHANICAL) == "LineLoadCondition3D2N"
        assert line_load_part.get_element_name(2, 3, AnalysisType.MECHANICAL) == "LineLoadDiffOrderCondition2D3N"
        assert line_load_part.get_element_name(3, 3, AnalysisType.MECHANICAL) == "LineLoadCondition3D3N"


        surface_load = SurfaceLoad([True, True, True], [0, 0, 0])

        moving_load = MovingLoad([10,10,10], [1, 1, 1], 1, [0, 0, 0])



        a=1+1