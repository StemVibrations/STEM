from stem.geometry import Geometry, Point, Line, Surface, Volume
from stem.mesh import Mesh, Node, Element, Condition
from stem.model_part import ModelPart, BodyModelPart

from tests.utils import TestUtils

class TestBodyModelPart:

    def test_write_body_model_part(self):


        point_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0, 1.0, 0.0]]

        points = []
        id =1
        for coord in point_coords:
            point = Point()
            point.coordinates = coord
            point.id = id
            id += 1
            points.append(point)

        lines = []
        for i in range(3):
            line = Line()
            line.id = i+1
            line.point_ids = [points[i].id, points[i+1].id]
            lines.append(line)

        # connect last to first point
        line = Line()
        line.id = 4
        line.point_ids = [points[3].id, points[0].id]
        lines.append(line)

        surfaces = []
        surface = Surface()
        surface.id = 1
        surface.line_ids = [1, 2, 3, 4]
        surfaces.append(surface)

        geometry = Geometry()
        geometry.points = points
        geometry.lines = lines
        geometry.surfaces = surfaces


        node_coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        nodes = []

        node_id = 1
        for coord in node_coordinates:
            node = Node(node_id, coord)
            node_id += 1
            nodes.append(node)

        elements = []
        elements.append(Element(1, "TRIANGLE_3N", [1, 2, 4]))
        elements.append(Element(2, "TRIANGLE_3N", [2, 3, 4]))

        mesh = Mesh()
        mesh.nodes = nodes
        mesh.elements = elements

        body_model_part = BodyModelPart()
        body_model_part.name = "test"
        body_model_part.geometry = geometry
        body_model_part.mesh = mesh



        body_model_part.write_body_model_part("test_body_model_part")