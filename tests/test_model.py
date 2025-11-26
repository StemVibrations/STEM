import pickle
from copy import deepcopy
from typing import Tuple
import re
import sys
from pathlib import Path

import numpy.testing as npt
import pytest

from stem.geometry import *
from stem.model import *
from stem.output import NodalOutput, GiDOutputParameters, JsonOutputParameters
from stem.solver import *
from stem.boundary import RotationConstraint, DisplacementConstraint
from tests.utils import TestUtils
from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw

IS_LINUX = sys.platform == "linux"


class TestModel:

    @pytest.fixture
    def model_setup_large_3d_custom(self):
        """
        This fixture creates a 3D model with predefined nodes and elements.
        In this example, we create a simple cubic structure with additional internal nodes.
        Five prism elements are created to form a more complex geometry for testing purposes.
        This larger model tests various functionalities of the Model class.

        Returns:
            - model (:class:`stem.model.Model`): A model object with predefined nodes and elements.
        """

        # --- Nodes ---
        raw_coords = {
            1: [1.0, 2.0, 0.0],
            2: [1.0, 1.0, 0.0],
            3: [1.0, 1.0, 1.0],
            7: [2.0, 1.0, 1.0],
            13: [1.5, 2.0, 0.0],
            15: [1.0, 1.5, 0.5],
            16: [1.0, 1.5, 0.5],
            17: [1.5, 1.0, 0.5102040816],
            18: [1.5, 1.5, 1.0],
            20: [2.0, 1.5, 0.5],
            21: [1.5, 0.0, 0.5],
            22: [2.0, 0.5, 0.5],
            23: [1.0, 0.5, 0.5],
            24: [1.5, 0.5, 1.0],
            25: [1.25, 1.5, 0.0],
            27: [1.0, 1.0, 0.0],
            28: [1.0, 1.0, 1.0],
            30: [2.0, 1.0, 1.0],
            31: [1.5, 1.0, 0.0],
            32: [1.5, 1.0, 0.5102040816]
        }
        nodes = {i: Node(i, coord) for i, coord in raw_coords.items()}

        # --- Elements ---
        # Columns: [elem_id, part_id, n1, n2, n3, n4] for tets
        tet_raw = [(28, 1, 21, 23, 24, 22), (29, 1, 17, 24, 23, 22), (30, 1, 22, 21, 23, 15), (53, 2, 13, 31, 20, 25),
                   (54, 2, 27, 25, 1, 16)]
        # Columns: [elem_id, part_id, n1,n2,n3,n4,n5,n6] for prisms
        prism_raw = [(146, 3, 7, 17, 3, 30, 32, 28), (147, 3, 2, 3, 17, 27, 28, 32)]

        # Build Element objects
        elements_part1 = {
            eid: Element(eid, "TETRAHEDRON_4N", [n1, n2, n3, n4])
            for eid, pid, n1, n2, n3, n4 in tet_raw if pid == 1
        }
        elements_part2 = {
            eid: Element(eid, "TETRAHEDRON_4N", [n1, n2, n3, n4])
            for eid, pid, n1, n2, n3, n4 in tet_raw if pid == 2
        }
        interface_elements = {
            eid: Element(eid, "PRISM_6N", [n1, n2, n3, n4, n5, n6])
            for eid, pid, n1, n2, n3, n4, n5, n6 in prism_raw
        }

        # --- Build Parts ---
        # Part 1
        part_1 = BodyModelPart("part_1")
        part_1.mesh = Mesh(3)
        # nodes referenced by elements of part 1
        part_1_node_ids = {nid for e in elements_part1.values() for nid in e.node_ids}
        part_1.mesh.nodes = {nid: nodes[nid] for nid in part_1_node_ids}
        part_1.mesh.elements = elements_part1

        # Part 2
        part_2 = BodyModelPart("part_2")
        part_2.mesh = Mesh(3)
        part_2_node_ids = {nid for e in elements_part2.values() for nid in e.node_ids}
        part_2.mesh.nodes = {nid: nodes[nid] for nid in part_2_node_ids}
        part_2.mesh.elements = elements_part2

        # Interface Part
        interface_part = BodyModelPart("interface_part")
        interface_part.mesh = Mesh(3)
        iface_node_ids = {nid for e in interface_elements.values() for nid in e.node_ids}
        interface_part.mesh.nodes = {nid: nodes[nid] for nid in iface_node_ids}
        interface_part.mesh.elements = interface_elements

        # --- Assemble Model ---
        model = Model(3)
        model.body_model_parts = [part_1, part_2]

        # gmsh_io
        md = model.gmsh_io.mesh_data
        md["nodes"] = {i: n.coordinates for i, n in nodes.items()}
        md["elements"] = {"TETRAHEDRON_4N": {**elements_part1, **elements_part2}, "PRISM_6N": interface_elements}
        md["physical_groups"] = {
            part_1.name: {
                "node_ids": sorted(part_1_node_ids),
                "element_ids": sorted(elements_part1.keys()),
                "ndim": 3,
                "element_type": "TETRAHEDRON_4N"
            },
            part_2.name: {
                "node_ids": sorted(part_2_node_ids),
                "element_ids": sorted(elements_part2.keys()),
                "ndim": 3,
                "element_type": "TETRAHEDRON_4N"
            },
            interface_part.name: {
                "node_ids": sorted(iface_node_ids),
                "element_ids": sorted(interface_elements.keys()),
                "ndim": 3,
                "element_type": "PRISM_6N"
            },
        }

        return model

    @pytest.fixture
    def model_setup_large_2d(self):
        """
        Set up test data for large 2D model tests
        This fixture creates a model with two body model parts and an interface part.
        The first part contains elements 1, 2, and 3, while the second
        part contains elements 4, 5, and 6. The interface part contains
        elements 7 and 8, which are quadrilateral elements connecting nodes
        from both parts.

        Returns:
            :class:`stem.model.Model`: A dictionary containing the model instance.
        """

        coordinates = [
            [0.0, 0.0, 0.0],  # Node 1
            [0.0, 1.0, 0.0],  # Node 2
            [1.0, 1.0, 0.0],  # Node 3
            [2.0, 0.0, 0.0],  # Node 4
            [2.0, 1.0, 0.0],  # Node 5
            [2.0, 2.0, 0.0],  # Node 6
            [0.0, 2.0, 0.0],  # Node 7
            [0.0, 1.0, 0.0],  # Node 8 - interface node for part 2
            [2.0, 1.0, 0.0],  # Node 9 - interface node for part 2
            [1.0, 1.0, 0.0]  # Node 10 - interface node for part 2
        ]

        # Create nodes
        nodes = {i + 1: Node(i + 1, coordinates[i]) for i in range(len(coordinates))}
        # Create the 6 elements
        elements = {
            1: Element(1, "TRIANGLE_3N", [1, 2, 3]),
            2: Element(2, "TRIANGLE_3N", [4, 5, 3]),
            3: Element(3, "TRIANGLE_3N", [1, 4, 3]),
            4: Element(4, "TRIANGLE_3N", [10, 6, 7]),
            5: Element(5, "TRIANGLE_3N", [8, 10, 7]),
            6: Element(6, "TRIANGLE_3N", [10, 9, 6])
        }
        # QUADRANGLE_4N
        interface_elements = {
            7: Element(7, "QUADRANGLE_4N", [2, 3, 10, 8]),
            8: Element(8, "QUADRANGLE_4N", [3, 5, 9, 10])
        }

        # Create stable part elements 1 2 3
        part_1 = BodyModelPart("part_1")
        part_1.mesh = Mesh(2)
        part_1.mesh.nodes = {1: nodes[1], 2: nodes[2], 3: nodes[3], 4: nodes[4], 5: nodes[5]}
        part_1.mesh.elements = {1: elements[1], 2: elements[2], 3: elements[3]}

        # Create changing part elements 4 5 6
        part_2 = BodyModelPart("part_2")
        part_2.mesh = Mesh(2)
        part_2.mesh.nodes = {6: nodes[6], 7: nodes[7], 8: nodes[8], 9: nodes[9], 10: nodes[10]}
        part_2.mesh.elements = {4: elements[4], 5: elements[5], 6: elements[6]}

        interface_part = BodyModelPart("interface_part")
        interface_part.mesh = Mesh(2)
        interface_part.mesh.nodes = {2: nodes[2], 3: nodes[3], 5: nodes[5], 8: nodes[8], 9: nodes[9], 10: nodes[10]}
        interface_part.mesh.elements = {7: interface_elements[7], 8: interface_elements[8]}

        # model instance for testing
        model = Model(2)
        model.body_model_parts = [part_1, part_2]

        # also the gmsh_io mesh data
        model.gmsh_io.mesh_data["nodes"] = {k: v.coordinates for k, v in nodes.items()}
        model.gmsh_io.mesh_data["elements"] = {"TRIANGLE_3N": {k: v.node_ids for k, v in elements.items()}}
        model.gmsh_io.mesh_data["elements"]["QUADRANGLE_4N"] = {k: v.node_ids for k, v in interface_elements.items()}
        model.gmsh_io.mesh_data["physical_groups"] = {}
        model.gmsh_io.mesh_data["physical_groups"][part_1.name] = {
            "node_ids": list(part_1.mesh.nodes.keys()),
            "element_ids": list(part_1.mesh.elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }
        model.gmsh_io.mesh_data["physical_groups"][part_2.name] = {
            "node_ids": list(part_2.mesh.nodes.keys()),
            "element_ids": list(part_2.mesh.elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }
        model.gmsh_io.mesh_data["physical_groups"][interface_part.name] = {
            "node_ids": list(interface_part.mesh.nodes.keys()),
            "element_ids": list(interface_part.mesh.elements.keys()),
            "ndim": 2,
            "element_type": "QUADRANGLE_4N"
        }
        # Return all needed objects for tests
        return model

    @pytest.fixture
    def model_2d_with_interface(self):
        """
        Creates a comprehensive 2D test model setup with predefined nodes, elements, and model parts.

        This fixture establishes a test environment containing:
        - Four nodes arranged in a linear configuration
        - Two triangular elements connecting these nodes
        - Two body model parts (stable_part and changing_part) with shared nodes
        - Interface material configuration for testing interface functionality
        - Complete mesh data structure for both GMSH and model components

        The setup is specifically designed for testing interface element generation,
        node ID mapping, and model part interactions in 2D scenarios.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - model (:class:`stem.model.Model`): 2D Model instance with two body model parts
                - nodes (Dict[int, Node]): Dictionary of node objects indexed by node ID
                - elements (Dict[int, Element]): Dictionary of triangle elements indexed by element ID
                - coords (List[List[float]]): List of coordinate arrays for each node
                - stable_part (:class:`stem.model.BodyModelPart`): First body model part (stable)
                - changing_part (:class:`stem.model.BodyModelPart`): Second body model part (changing)
                - interface_material (:class:`stem.material.InterfaceMaterial`): Material for interface elements
        """
        # Define test coordinates
        coords = [
            [0.0, 0.0, 0.0],  # Node 1
            [1.0, 0.0, 0.0],  # Node 2
            [1.0, 1.0, 0.0],  # Node 3
            [2.0, 0.0, 0.0]  # Node 4
        ]

        # Create nodes
        nodes = {i + 1: Node(i + 1, coords[i]) for i in range(len(coords))}

        # Create elements
        elements = {
            1: Element(1, "TRIANGLE_3N", [1, 2, 3]),  # Element in stable part
            2: Element(2, "TRIANGLE_3N", [2, 3, 4])  # Element in changing part with nodes 2 and 3 in common
        }

        # Create stable part
        stable_part = BodyModelPart("stable_part")
        stable_part.mesh = Mesh(2)
        stable_part.mesh.nodes = {1: nodes[1], 2: nodes[2], 3: nodes[3]}
        stable_part.mesh.elements = {1: elements[1]}

        # Create changing part
        changing_part = BodyModelPart("changing_part")
        changing_part.mesh = Mesh(2)
        changing_part.mesh.nodes = {2: nodes[2], 3: nodes[3], 4: nodes[4]}
        changing_part.mesh.elements = {2: elements[2]}

        # Create model instance for testing
        model = Model(2)  # Assuming 2D model for this test
        model.body_model_parts = [stable_part, changing_part]

        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=0.2)
        variables = OnePhaseSoilInterface(2,
                                          IS_DRAINED=True,
                                          DENSITY_SOLID=2000,
                                          POROSITY=0.3,
                                          MINIMUM_JOINT_WIDTH=0.001)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        interface_material = InterfaceMaterial(name="interface",
                                               constitutive_law=constitutive_law,
                                               soil_formulation=variables,
                                               retention_parameters=retention_parameters)

        model.gmsh_io.mesh_data["nodes"] = {k: v.coordinates for k, v in nodes.items()}
        model.gmsh_io.mesh_data["elements"] = {"TRIANGLE_3N": {k: v.node_ids for k, v in elements.items()}}
        model.gmsh_io.mesh_data["physical_groups"] = {}
        model.gmsh_io.mesh_data["physical_groups"][stable_part.name] = {
            "node_ids": list(stable_part.mesh.nodes.keys()),
            "element_ids": list(stable_part.mesh.elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }
        model.gmsh_io.mesh_data["physical_groups"][changing_part.name] = {
            "node_ids": list(changing_part.mesh.nodes.keys()),
            "element_ids": list(changing_part.mesh.elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }

        # Return all needed objects for tests
        return {
            "model": model,
            "nodes": nodes,
            "elements": elements,
            "coords": coords,
            "stable_part": stable_part,
            "changing_part": changing_part,
            "interface_material": interface_material
        }

    @pytest.fixture
    def model_setup_3d_with_interface(self):
        """
        Creates a comprehensive 3D test model setup with predefined nodes, elements, and model parts.
        The setup is specifically designed for testing interface element generation,
        node ID mapping, and model part interactions in 3D scenarios. The stable part
        contains nodes 1, 2, 3, 4 while the changing part contains nodes 1, 2, 3, 5,
        creating a shared interface along nodes 1, 2, and 3.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - model (:class:`stem.model.Model`): 3D Model instance with two body model parts
                - nodes (Dict[int, Node]): Dictionary of node objects indexed by node ID
                - elements (Dict[int, Element]): Dictionary of tetrahedral elements indexed by element ID
                - coords (List[List[float]]): List of coordinate arrays for each node
                - stable_part (:class:`stem.model.BodyModelPart`): First body model part (stable)
                - changing_part (:class:`stem.model.BodyModelPart`): Second body model part (changing)
                - interface_material (:class:`stem.material.InterfaceMaterial`): Material for interface elements
        """
        # Define test coordinates for a 3D model
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        # Create nodes
        nodes = {i + 1: Node(i + 1, coords[i]) for i in range(len(coords))}
        # Create elements
        elements = {
            1: Element(1, "TETRAHEDRON_4N", [1, 2, 3, 4]),  # Element in stable part
            2: Element(2, "TETRAHEDRON_4N", [1, 2, 3, 5])  # Element in changing part with nodes 1 and 2 in common
        }
        # Create stable part
        stable_part = BodyModelPart("stable_part")
        stable_part.mesh = Mesh(3)
        stable_part.mesh.nodes = {1: nodes[1], 2: nodes[2], 3: nodes[3], 4: nodes[4]}
        stable_part.mesh.elements = {1: elements[1]}
        # Create changing part
        changing_part = BodyModelPart("changing_part")
        changing_part.mesh = Mesh(3)
        changing_part.mesh.nodes = {1: nodes[1], 2: nodes[2], 3: nodes[3], 5: nodes[5]}
        changing_part.mesh.elements = {2: elements[2]}
        # Create model instance for testing
        model = Model(3)  # Assuming 3D model for this test
        model.body_model_parts = [stable_part, changing_part]
        # also create the gmsh_io mesh data
        model.gmsh_io.mesh_data["nodes"] = {k: v.coordinates for k, v in nodes.items()}
        model.gmsh_io.mesh_data["elements"] = {"TETRAHEDRON_4N": {k: v.node_ids for k, v in elements.items()}}
        model.gmsh_io.mesh_data["physical_groups"] = {}
        model.gmsh_io.mesh_data["physical_groups"][stable_part.name] = {
            "node_ids": list(stable_part.mesh.nodes.keys()),
            "element_ids": list(stable_part.mesh.elements.keys()),
            "ndim": 3,
            "element_type": "TETRAHEDRON_4N"
        }
        model.gmsh_io.mesh_data["physical_groups"][changing_part.name] = {
            "node_ids": list(changing_part.mesh.nodes.keys()),
            "element_ids": list(changing_part.mesh.elements.keys()),
            "ndim": 3,
            "element_type": "TETRAHEDRON_4N"
        }
        constitutive_law = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=0.2)
        variables = OnePhaseSoilInterface(2,
                                          IS_DRAINED=True,
                                          DENSITY_SOLID=2000,
                                          POROSITY=0.3,
                                          MINIMUM_JOINT_WIDTH=0.001)
        retention_parameters = SaturatedBelowPhreaticLevelLaw()
        interface_material = InterfaceMaterial(name="interface",
                                               constitutive_law=constitutive_law,
                                               soil_formulation=variables,
                                               retention_parameters=retention_parameters)

        # Return all needed objects for tests
        return {
            "model": model,
            "nodes": nodes,
            "elements": elements,
            "coords": coords,
            "stable_part": stable_part,
            "interface_material": interface_material,
            "changing_part": changing_part
        }

    @pytest.fixture
    def expected_geo_data_0D(self):
        """
        Expected geometry data for a 0D geometry group. The group is a geometry of a point

        Returns:
            - Dict[str, Any]: dictionary containing the geometry data as generated by the gmsh_io
        """
        expected_points = {1: [0, 0, 0], 2: [0.5, 0, 0]}
        return {"points": expected_points}

    @pytest.fixture
    def expected_geometry_single_layer_2D(self):
        """
        Sets expected geometry data for a 2D geometry group. The group is a geometry of a square.

        Returns:
            - :class:`stem.geometry.Geometry`: geometry of a 2D square
        """

        geometry = Geometry()

        geometry.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([1, 0, 0], 2),
            3: Point.create([1, 1, 0], 3),
            4: Point.create([0, 1, 0], 4)
        }

        geometry.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([2, 3], 2),
            3: Line.create([3, 4], 3),
            4: Line.create([4, 1], 4)
        }

        geometry.surfaces = {1: Surface.create([1, 2, 3, 4], 1)}

        geometry.volumes = {}

        return geometry

    @pytest.fixture
    def expected_geometry_single_layer_3D(self):
        """
        Sets expected geometry data for a 2D geometry group. The group is a geometry of a square.

        Returns:
            - :class:`stem.geometry.Geometry`: geometry of a 2D square
        """

        geometry = Geometry()

        geometry.points = {
            1: Point.create([0, 0, 0], 1),
            5: Point.create([0, 0, 1], 5),
            6: Point.create([1, 0, 1], 6),
            2: Point.create([1, 0, 0], 2),
            7: Point.create([1, 1, 1], 7),
            3: Point.create([1, 1, 0], 3),
            8: Point.create([0, 1, 1], 8),
            4: Point.create([0, 1, 0], 4)
        }

        geometry.lines = {
            5: Line.create([1, 5], 5),
            7: Line.create([5, 6], 7),
            6: Line.create([2, 6], 6),
            1: Line.create([1, 2], 1),
            9: Line.create([6, 7], 9),
            8: Line.create([3, 7], 8),
            2: Line.create([2, 3], 2),
            11: Line.create([7, 8], 11),
            10: Line.create([4, 8], 10),
            3: Line.create([3, 4], 3),
            12: Line.create([8, 5], 12),
            4: Line.create([4, 1], 4)
        }

        geometry.surfaces = {
            2: Surface.create([5, 7, -6, -1], 2),
            3: Surface.create([6, 9, -8, -2], 3),
            4: Surface.create([8, 11, -10, -3], 4),
            5: Surface.create([10, 12, -5, -4], 5),
            1: Surface.create([1, 2, 3, 4], 1),
            6: Surface.create([7, 9, 11, 12], 6)
        }

        # The volumes list converted to a dictionary
        geometry.volumes = {1: Volume.create([-2, -3, -4, -5, -1, 6], 1)}

        return geometry

    @pytest.fixture
    def expected_geometry_two_layers_2D(self):
        """
        Sets expected geometries for 2 attached 2D squares.

        Returns:
            - Tuple[:class:`stem.geometry.Geometry`,:class:`stem.geometry.Geometry`, :class:`stem.geometry.Geometry` ]:\
                geometries of 2 attached 2D squares

        """

        # geometry_1
        geometry_1 = Geometry()
        geometry_1.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([1, 0, 0], 2),
            3: Point.create([1, 1, 0], 3),
            4: Point.create([0, 1, 0], 4)
        }

        geometry_1.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([2, 3], 2),
            3: Line.create([3, 4], 3),
            4: Line.create([4, 1], 4)
        }

        geometry_1.surfaces = {1: Surface.create([1, 2, 3, 4], 1)}

        geometry_1.volumes = {}

        # geometry_2
        geometry_2 = Geometry()

        geometry_2.points = {
            5: Point.create([1, 2, 0], 5),
            6: Point.create([0, 2, 0], 6),
            4: Point.create([0, 1, 0], 4),
            3: Point.create([1, 1, 0], 3)
        }

        geometry_2.lines = {
            5: Line.create([5, 6], 5),
            6: Line.create([6, 4], 6),
            3: Line.create([3, 4], 3),
            7: Line.create([3, 5], 7)
        }

        geometry_2.surfaces = {2: Surface.create([5, 6, -3, 7], 2)}

        geometry_2.volumes = {}

        full_geometry = Geometry()
        full_geometry.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([1, 0, 0], 2),
            3: Point.create([1, 1, 0], 3),
            4: Point.create([0, 1, 0], 4),
            5: Point.create([1, 2, 0], 5),
            6: Point.create([0, 2, 0], 6)
        }

        full_geometry.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([2, 3], 2),
            3: Line.create([3, 4], 3),
            4: Line.create([4, 1], 4),
            5: Line.create([5, 6], 5),
            6: Line.create([6, 4], 6),
            7: Line.create([3, 5], 7)
        }

        full_geometry.surfaces = {1: Surface.create([1, 2, 3, 4], 1), 2: Surface.create([5, 6, -3, 7], 2)}

        full_geometry.volumes = {}

        return geometry_1, geometry_2, full_geometry

    @pytest.fixture
    def expected_geometry_two_layers_2D_after_sync(self):
        """
        Sets expected geometry of two model parts and the whole model after synchronising the geometry.

        Returns:
            - Tuple[:class:`stem.geometry.Geometry`,:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`]: geometries of 2 attached 2D squares and the whole model
        """

        # create expected geometry layer 1
        geometry_1 = Geometry()

        geometry_1.points = {
            8: Point.create([0, 0, 0], 8),
            9: Point.create([1, 0, 0], 9),
            3: Point.create([1, 1, 0], 3),
            7: Point.create([0.5, 1, 0], 7),
            10: Point.create([0, 1, 0], 10)
        }

        geometry_1.lines = {
            9: Line.create([8, 9], 9),
            10: Line.create([9, 3], 10),
            7: Line.create([7, 3], 7),
            11: Line.create([7, 10], 11),
            12: Line.create([10, 8], 12)
        }

        geometry_1.surfaces = {1: Surface.create([9, 10, -7, 11, 12], 1)}

        geometry_2 = Geometry()
        geometry_2.points = {
            5: Point.create([1.0, 2.0, 0.0], 5),
            6: Point.create([0.5, 2.0, 0.0], 6),
            7: Point.create([0.5, 1, 0], 7),
            3: Point.create([1, 1, 0], 3)
        }

        geometry_2.lines = {
            5: Line.create([5, 6], 5),
            6: Line.create([6, 7], 6),
            7: Line.create([7, 3], 7),
            8: Line.create([3, 5], 8)
        }

        geometry_2.surfaces = {2: Surface.create([5, 6, 7, 8], 2)}

        geometry_2.volumes = {}

        # create expected full geometry
        full_geometry = Geometry()
        full_geometry.points = {
            3: Point.create([1, 1, 0], 3),
            5: Point.create([1, 2, 0], 5),
            6: Point.create([0.5, 2, 0], 6),
            7: Point.create([0.5, 1, 0], 7),
            8: Point.create([0, 0, 0], 8),
            9: Point.create([1, 0, 0], 9),
            10: Point.create([0, 1, 0], 10)
        }

        full_geometry.lines = {
            5: Line.create([5, 6], 5),
            6: Line.create([6, 7], 6),
            7: Line.create([7, 3], 7),
            8: Line.create([3, 5], 8),
            9: Line.create([8, 9], 9),
            10: Line.create([9, 3], 10),
            11: Line.create([7, 10], 11),
            12: Line.create([10, 8], 12)
        }

        full_geometry.surfaces = {1: Surface.create([9, 10, -7, 11, 12], 1), 2: Surface.create([5, 6, 7, 8], 2)}

        full_geometry.volumes = {}

        return geometry_1, geometry_2, full_geometry

    @pytest.fixture
    def expected_geometry_line_load(self):
        """
        Sets expected geometry data for a 1D geometry group. The group is a geometry of a multi-line.

        Returns:
            - :class:`stem.geometry.Geometry`: geometry of a 1D multi-line
        """

        geometry = Geometry()

        geometry.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([3, 0, 0], 2),
            3: Point.create([4, -1, 0], 3),
            4: Point.create([10, -1, 0], 4)
        }

        geometry.lines = {1: Line.create([1, 2], 1), 2: Line.create([2, 3], 2), 3: Line.create([3, 4], 3)}

        geometry.surfaces = {}

        geometry.volumes = {}

        return geometry

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
        soil_material = SoilMaterial(name="soil",
                                     soil_formulation=soil_formulation,
                                     constitutive_law=constitutive_law,
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
        soil_material = SoilMaterial(name="soil",
                                     soil_formulation=soil_formulation,
                                     constitutive_law=constitutive_law,
                                     retention_parameters=SaturatedBelowPhreaticLevelLaw())
        return soil_material

    @pytest.fixture
    def create_default_3d_beam(self):
        """
        Create a default beam material for a 3D geometry.
        """
        # Specify beam material model
        YOUNG_MODULUS = 210000000000
        POISSON_RATIO = 0.30000
        DENSITY = 7850
        CROSS_AREA = 0.01
        I22 = 0.0001
        I33 = 0.0001

        TORTIONAL_INERTIA = I22 + I33
        beam_material = EulerBeam(3, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I33, I22, TORTIONAL_INERTIA)
        name = "beam"
        return StructuralMaterial(name, beam_material)

    @pytest.fixture
    def create_default_point_load_parameters(self):
        """
        Create a default point load parameters.

        Returns:
            - :class:`stem.load.PointLoad`: default point load

        """
        # define soil material
        return PointLoad(active=[False, True, False], value=[0, -200, 0])

    @pytest.fixture
    def create_default_line_load_parameters(self):
        """
        Create a default line load parameters.

        Returns:
            - :class:`stem.load.PointLoad`: default point load

        """
        # define soil material
        return LineLoad(active=[False, True, False], value=[0, -20, 0])

    @pytest.fixture
    def create_default_surface_load_parameters(self):
        """
        Create a default surface load properties.

        Returns:
            - :class:`stem.load.SurfaceLoad`: default surface load

        """
        # define soil material
        return SurfaceLoad(active=[False, True, False], value=[0, -2, 0])

    @pytest.fixture
    def create_default_moving_load_parameters(self):
        """
        Create a default surface load properties.

        Returns:
            - :class:`stem.load.SurfaceLoad`: default surface load

        """
        # define soil material
        return MovingLoad(origin=[3.5, -0.5, 0.0],
                          load=[0.0, -10.0, 0.0],
                          velocity=5.0,
                          offset=3.0,
                          direction=[1, 1, 1])

    @pytest.fixture
    def create_default_outputs(self):
        """
        Sets default output parameters.

        Returns:
            - List[:class:`stem.output.Output`]: list of default output processes.
        """
        # Nodal results
        nodal_results = [NodalOutput.ACCELERATION]
        # Gauss point results
        # define output process

        output_process = Output(part_name="nodal_accelerations",
                                output_name="gid_nodal_accelerations_top",
                                output_dir="dir_test",
                                output_parameters=GiDOutputParameters(file_format="binary",
                                                                      output_interval=100,
                                                                      nodal_results=nodal_results))

        return output_process

    @pytest.fixture
    def expected_geometry_two_layers_3D_extruded(self) -> Tuple[Geometry, Geometry]:
        """
        Expected geometry data for a 3D geometry create from 2D extrusion. The geometry is 2 stacked blocks, where the
        top and bottom blocks are in different groups.

        Returns:
            - Tuple[:class:`stem.geometry.Geometry`,:class:`stem.geometry.Geometry`]: expected geometry data
        """

        geometry_1 = Geometry()
        geometry_1.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([0, 0, 1], 2),
            4: Point.create([1, 0, 1], 4),
            3: Point.create([1, 0, 0], 3),
            6: Point.create([1, 1, 1], 6),
            5: Point.create([1, 1, 0], 5),
            8: Point.create([0, 1, 1], 8),
            7: Point.create([0, 1, 0], 7)
        }

        geometry_1.lines = {
            1: Line.create([1, 2], 1),
            4: Line.create([2, 4], 4),
            2: Line.create([3, 4], 2),
            3: Line.create([1, 3], 3),
            7: Line.create([4, 6], 7),
            5: Line.create([5, 6], 5),
            6: Line.create([3, 5], 6),
            10: Line.create([6, 8], 10),
            8: Line.create([7, 8], 8),
            9: Line.create([5, 7], 9),
            12: Line.create([8, 2], 12),
            11: Line.create([7, 1], 11)
        }

        geometry_1.surfaces = {
            1: Surface.create([1, 4, -2, -3], 1),
            2: Surface.create([2, 7, -5, -6], 2),
            3: Surface.create([5, 10, -8, -9], 3),
            4: Surface.create([8, 12, -1, -11], 4),
            5: Surface.create([3, 6, 9, 11], 5),
            6: Surface.create([4, 7, 10, 12], 6)
        }

        geometry_1.volumes = {1: Volume.create([-1, -2, -3, -4, -5, 6], 1)}

        geometry_2 = Geometry()

        geometry_2.points = {
            9: Point.create([1.0, 2.0, 0.0], 9),
            10: Point.create([1., 2., 1.], 10),
            12: Point.create([0.0, 2., 1.], 12),
            11: Point.create([0, 2., 0.], 11),
            8: Point.create([0., 1., 1], 8),
            7: Point.create([0., 1., 0], 7),
            5: Point.create([1, 1., 0], 5),
            6: Point.create([1, 1., 1], 6)
        }

        geometry_2.lines = {
            13: Line.create([9, 10], 13),
            16: Line.create([10, 12], 16),
            14: Line.create([11, 12], 14),
            15: Line.create([9, 11], 15),
            18: Line.create([12, 8], 18),
            8: Line.create([7, 8], 8),
            17: Line.create([11, 7], 17),
            5: Line.create([5, 6], 5),
            10: Line.create([6, 8], 10),
            9: Line.create([5, 7], 9),
            20: Line.create([6, 10], 20),
            19: Line.create([5, 9], 19)
        }

        geometry_2.surfaces = {
            7: Surface.create([13, 16, -14, -15], 7),
            8: Surface.create([14, 18, -8, -17], 8),
            3: Surface.create([5, 10, -8, -9], 3),
            9: Surface.create([5, 20, -13, -19], 9),
            10: Surface.create([15, 17, -9, 19], 10),
            11: Surface.create([16, 18, -10, 20], 11)
        }

        geometry_2.volumes = {2: Volume.create([-7, -8, 3, -9, -10, 11], 2)}

        return geometry_1, geometry_2

    @pytest.fixture
    def expected_geometry_two_layers_3D_geo_file(self):
        """
        Expected geometry data for a 3D geometry create in a geo file. The geometry is 2 stacked blocks, where the top
        and bottom blocks are in different groups.

        Returns:
            - Tuple[:class:`stem.geometry.Geometry`,:class:`stem.geometry.Geometry`]: expected geometry data
        """

        geometry_1 = Geometry()
        geometry_1.volumes = {1: Volume.create([-10, 39, 26, 30, 34, 38], 1)}

        geometry_1.surfaces = {
            10: Surface.create([5, 6, 7, 8], 10),
            39: Surface.create([19, 20, 21, 22], 39),
            26: Surface.create([5, 25, -19, -24], 26),
            30: Surface.create([6, 29, -20, -25], 30),
            34: Surface.create([7, 33, -21, -29], 34),
            38: Surface.create([8, 24, -22, -33], 38)
        }

        geometry_1.lines = {
            5: Line.create([1, 2], 5),
            6: Line.create([2, 3], 6),
            7: Line.create([3, 4], 7),
            8: Line.create([4, 1], 8),
            19: Line.create([13, 14], 19),
            20: Line.create([14, 18], 20),
            21: Line.create([18, 22], 21),
            22: Line.create([22, 13], 22),
            25: Line.create([2, 14], 25),
            24: Line.create([1, 13], 24),
            29: Line.create([3, 18], 29),
            33: Line.create([4, 22], 33)
        }

        geometry_1.points = {
            1: Point.create([0., 0., 0.], 1),
            2: Point.create([0.5, 0., 0.], 2),
            3: Point.create([0.5, 1., 0.], 3),
            4: Point.create([0., 1., 0.], 4),
            13: Point.create([0., 0., -0.5], 13),
            14: Point.create([0.5, 0., -0.5], 14),
            18: Point.create([0.5, 1., -0.5], 18),
            22: Point.create([0., 1., -0.5], 22)
        }

        geometry_2 = Geometry()
        geometry_2.volumes = {2: Volume.create([-17, 61, -48, -34, -56, -60], 2)}

        geometry_2.surfaces = {
            17: Surface.create([-13, -7, -15, -14], 17),
            61: Surface.create([41, -21, 43, 44], 61),
            48: Surface.create([-13, 33, -41, -46], 48),
            34: Surface.create([7, 33, -21, -29], 34),
            56: Surface.create([-15, 55, -43, -29], 56),
            60: Surface.create([-14, 46, -44, -55], 60)
        }

        geometry_2.lines = {
            13: Line.create([4, 11], 13),
            7: Line.create([3, 4], 7),
            15: Line.create([12, 3], 15),
            14: Line.create([11, 12], 14),
            41: Line.create([23, 22], 41),
            21: Line.create([18, 22], 21),
            43: Line.create([18, 32], 43),
            44: Line.create([32, 23], 44),
            33: Line.create([4, 22], 33),
            46: Line.create([11, 23], 46),
            29: Line.create([3, 18], 29),
            55: Line.create([12, 32], 55)
        }

        geometry_2.points = {
            4: Point.create([0., 1., 0.], 4),
            11: Point.create([0., 2., 0.], 11),
            3: Point.create([0.5, 1., 0.], 3),
            12: Point.create([0.5, 2., 0.], 12),
            23: Point.create([0., 2., -0.5], 23),
            22: Point.create([0., 1., -0.5], 22),
            18: Point.create([0.5, 1., -0.5], 18),
            32: Point.create([0.5, 2., -0.5], 32)
        }

        return geometry_1, geometry_2

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

    def test_add_single_soil_layer_2D(self, expected_geometry_single_layer_2D: Geometry,
                                      create_default_2d_soil_material: SoilMaterial):
        """
        Test if a single soil layer is added correctly to the model in a 2D space. A single soil layer is generated
        and a single soil material is created and added to the model.

        Args:
            - expected_geometry_single_layer_2D (:class:`stem.geometry.Geometry`): expected geometry of the model
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 2

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        # define soil material
        soil_material = create_default_2d_soil_material

        # create model
        model = Model(ndim)

        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material

        # check if geometry is added correctly
        generated_geometry = model.body_model_parts[0].geometry
        expected_geometry = expected_geometry_single_layer_2D

        # check if points are added correctly
        TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_single_soil_layer_3D(self, expected_geometry_single_layer_3D: Geometry,
                                      create_default_3d_soil_material: SoilMaterial):
        """
        Test if a single soil layer is added correctly to the model in a 3D space. A single soil layer is generated
        and a single soil material is created and added to the model.

        Args:
            - expected_geometry_single_layer_3D (:class:`stem.geometry.Geometry`): expected geometry of the model
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        # define soil material
        soil_material = create_default_3d_soil_material

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        model.project_parameters = TestUtils.create_default_solver_settings()

        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material

        # check if geometry is added correctly
        generated_geometry = model.body_model_parts[0].geometry
        expected_geometry = expected_geometry_single_layer_3D

        # check if points are added correctly
        TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_multiple_soil_layers_2D(self, expected_geometry_two_layers_2D: Tuple[Geometry, Geometry, Geometry],
                                         create_default_2d_soil_material: SoilMaterial):
        """
        Test if multiple soil layers are added correctly to the model in a 2D space. Multiple soil layers are generated
        and multiple soil materials are created and added to the model.

        Args:
            - expected_geometry_two_layers_2D (Tuple[:class:`stem.geometry.Geometry`, :class:`stem.geometry.Geometry`, \
              :class:`stem.geometry.Geometry`]): expected geometry of the model
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

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
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

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

            TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_multiple_soil_layers_3D(self, expected_geometry_two_layers_3D_extruded: Tuple[Geometry, Geometry],
                                         create_default_3d_soil_material: SoilMaterial):
        """
        Test if multiple soil layers are added correctly to the model in a 3D space. Multiple soil layers are generated
        and multiple soil materials are created and added to the model.

        Args:
            - expected_geometry_two_layers_3D_extruded (Tuple[:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`]): expected geometry of the model which is created by extruding \
                a 2D geometry
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3

        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        soil_material2 = create_default_3d_soil_material
        soil_material2.name = "soil2"

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        model.synchronise_geometry()

        # check if layers are added correctly
        assert len(model.body_model_parts) == 2
        assert model.body_model_parts[0].name == "layer1"
        assert model.body_model_parts[0].material == soil_material1
        assert model.body_model_parts[1].name == "layer2"
        assert model.body_model_parts[1].material == soil_material2

        # check if geometry is added correctly for each layer
        for i in range(len(model.body_model_parts)):
            generated_geometry = model.body_model_parts[i].geometry
            expected_geometry = expected_geometry_two_layers_3D_extruded[i]

            TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_validation_of_adding_soil_layers(self, create_default_3d_soil_material: SoilMaterial):
        """
        Tests that errors are raised when groups are not specified or added multiple times.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3

        shape1 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)
        # add a valid group
        model.add_group_for_extrusion(group_name="Group1", reference_depth=0, extrusion_length=1)

        # expect it raises an error when adding a layer to a non-existing section
        with pytest.raises(
                ValueError,
                match="For 3D models either the extrusion length or the group name for the extrusion must be specified."
        ):
            model.add_soil_layer_by_coordinates(shape1, soil_material1, "layer1")

        # expect it raises an error when adding a layer to a non-existing section
        with pytest.raises(ValueError, match="Non-existent group specified `Group2`."):
            model.add_soil_layer_by_coordinates(shape1, soil_material1, "layer1", group_name="Group2")

        # add a soil layer that doesn't contain the reference point of the group
        shape2 = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

        with pytest.raises(
                ValueError,
                match="The reference coordinate of group: Group1, does not lay on the same plane as soil layer: layer2"
        ):
            model.add_soil_layer_by_coordinates(shape2, soil_material1, "layer2", group_name="Group1")

        # add a soil layer which section is not planar
        shape3 = [(0, 0, 0), (1, 0, 0), (1, 1, 2), (0, 1, 3)]

        with pytest.raises(ValueError, match="Polygon for the soil layer are not on the same plane."):
            model.add_soil_layer_by_coordinates(shape3, soil_material1, "layer3", group_name="Group1")

    def test_validation_of_adding_groups(self):
        """
        Tests that errors are raised when groups are not specified or added multiple times.

        """

        ndim = 3

        # create model
        model = Model(ndim)
        # add a valid group
        model.add_group_for_extrusion(group_name="Group1", reference_depth=0, extrusion_length=1)

        # expect it raises an error when adding an already existing section
        with pytest.raises(ValueError, match="The group `Group1` already exists, but group names must be unique."):
            model.add_group_for_extrusion(group_name="Group1", reference_depth=0, extrusion_length=1)

    def test_adding_model_parts_to_groups(self):
        """
        Tests validation of adding model parts to groups.

        """

        ndim = 3

        # create model
        model = Model(ndim)
        # add a valid group
        model.add_group_for_extrusion(group_name="Group1", reference_depth=0, extrusion_length=1)

        # test if raises an error when adding a model part to a non existing group
        with pytest.raises(ValueError, match="The group specified `Group2` does not exist."):
            model.add_model_part_to_group(group_name="Group2", part_name="test_part")

        # test if raises an error when adding a non existing model part to a group
        with pytest.raises(ValueError, match="The model part specified `test_part` does not exist."):
            model.add_model_part_to_group(group_name="Group1", part_name="test_part")

    def test_add_multiple_sections_3D(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if two extruded sections are added correctly to the model in a 3D space. Two triangular sections are
        sequentially extruded and multiple soil materials are created and added to the model.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3

        shape1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        shape2 = [(0, 0.5, 1), (0.5, 0.5, 1), (0, 1, 1)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        soil_material2 = create_default_3d_soil_material
        soil_material2.name = "soil2"

        # create model
        model = Model(ndim)
        model.add_group_for_extrusion(group_name="Group1", reference_depth=0, extrusion_length=1)
        model.add_group_for_extrusion(group_name="Group2", reference_depth=1, extrusion_length=1)

        # add soil layers
        model.add_soil_layer_by_coordinates(shape1, soil_material1, "layer1", group_name="Group1")
        model.add_soil_layer_by_coordinates(shape2, soil_material2, "layer2", group_name="Group2")

        model.synchronise_geometry()

        # check if layers are added correctly
        assert len(model.body_model_parts) == 2
        assert model.body_model_parts[0].name == "layer1"
        assert model.body_model_parts[0].material == soil_material1
        assert model.body_model_parts[1].name == "layer2"
        assert model.body_model_parts[1].material == soil_material2

        # check if geometry is added correctly for each layer
        geometry_1 = Geometry()
        geometry_1.points = {
            12: Point.create([0.0, 0.0, 0.0], 12),
            13: Point.create([0.0, 0.0, 1.0], 13),
            15: Point.create([1.0, 0.0, 1.0], 15),
            14: Point.create([1.0, 0.0, 0.0], 14),
            8: Point.create([0.5, 0.5, 1.0], 8),
            6: Point.create([0.0, 1.0, 1.0], 6),
            16: Point.create([0.0, 1.0, 0.0], 16),
            7: Point.create([0.0, 0.5, 1.0], 7)
        }

        geometry_1.lines = {
            19: Line.create([12, 13], 19),
            22: Line.create([13, 15], 22),
            20: Line.create([14, 15], 20),
            21: Line.create([12, 14], 21),
            25: Line.create([15, 8], 25),
            11: Line.create([8, 6], 11),
            23: Line.create([16, 6], 23),
            24: Line.create([14, 16], 24),
            12: Line.create([6, 7], 12),
            27: Line.create([7, 13], 27),
            26: Line.create([16, 12], 26),
            10: Line.create([7, 8], 10)
        }

        geometry_1.surfaces = {
            11: Surface.create([19, 22, -20, -21], 11),
            12: Surface.create([20, 25, 11, -23, -24], 12),
            13: Surface.create([23, 12, 27, -19, -26], 13),
            14: Surface.create([21, 24, 26], 14),
            15: Surface.create([27, 22, 25, -10], 15),
            6: Surface.create([10, 11, 12], 6)
        }

        geometry_1.volumes = {1: Volume.create([-11, -12, -13, -14, 15, 6], 1)}

        geometry_2 = Geometry()

        geometry_2.points = {
            7: Point.create([0.0, 0.5, 1.0], 7),
            9: Point.create([0.0, 0.5, 2.0], 9),
            10: Point.create([0.5, 0.5, 2.0], 10),
            8: Point.create([0.5, 0.5, 1.0], 8),
            11: Point.create([0.0, 1.0, 2.0], 11),
            6: Point.create([0.0, 1.0, 1.0], 6)
        }

        geometry_2.lines = {
            13: Line.create([7, 9], 13),
            15: Line.create([9, 10], 15),
            14: Line.create([8, 10], 14),
            10: Line.create([7, 8], 10),
            17: Line.create([10, 11], 17),
            16: Line.create([6, 11], 16),
            11: Line.create([8, 6], 11),
            18: Line.create([11, 9], 18),
            12: Line.create([6, 7], 12)
        }

        geometry_2.surfaces = {
            7: Surface.create([13, 15, -14, -10], 7),
            8: Surface.create([14, 17, -16, -11], 8),
            9: Surface.create([16, 18, -13, -12], 9),
            6: Surface.create([10, 11, 12], 6),
            10: Surface.create([15, 17, 18], 10)
        }

        geometry_2.volumes = {2: Volume.create([-7, -8, -9, -6, 10], 2)}

        expected_geometries = [geometry_1, geometry_2]

        for model_part, expected_geometry in zip(model.body_model_parts, expected_geometries):

            TestUtils.assert_almost_equal_geometries(expected_geometry, model_part.geometry)

    def test_add_all_layers_from_geo_file_2D(self, expected_geometry_two_layers_2D: Tuple[Geometry, Geometry,
                                                                                          Geometry]):
        """
        Tests if all layers are added correctly to the model in a 2D space. A geo file is read and all layers are
        added to the model.

        Args:
            - expected_geometry_two_layers_2D (Tuple[:class:`stem.geometry.Geometry`, :class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`]): expected geometry of the model

        """

        geo_file_name = "tests/test_data/gmsh_utils_two_blocks_2D.geo"

        # create model
        model = Model(ndim=2)
        model.add_all_layers_from_geo_file(geo_file_name, ["group_1"])

        # check if body model parts are added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "group_1"

        # check if process model part is added correctly
        assert len(model.process_model_parts) == 1
        assert model.process_model_parts[0].name == "group_2"

        # check if geometry is added correctly for each layer
        for i in range(len(model.body_model_parts)):
            generated_geometry = model.body_model_parts[i].geometry
            expected_geometry = expected_geometry_two_layers_2D[i]

            TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_all_layers_from_geo_file_3D(self, expected_geometry_two_layers_3D_geo_file: Tuple[Geometry, Geometry]):
        """
        Tests if all layers are added correctly to the model in a 3D space. A geo file is read and all layers are
        added to the model.

        Args:
            - expected_geometry_two_layers_3D_geo_file (Tuple[:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`]): expected geometry of the model

        """

        geo_file_name = "tests/test_data/gmsh_utils_column_3D_tetra4.geo"

        # create model
        model = Model(ndim=3)
        model.add_all_layers_from_geo_file(geo_file_name, ["group_1"])

        # check if body model parts are added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "group_1"

        # check if process model part is added correctly
        assert len(model.process_model_parts) == 1
        assert model.process_model_parts[0].name == "group_2"

        # check if geometry is added correctly
        all_model_parts = []
        all_model_parts.extend(model.body_model_parts)
        all_model_parts.extend(model.process_model_parts)

        # check if geometry is added correctly for each layer
        for i in range(len(all_model_parts)):
            generated_geometry = all_model_parts[i].geometry
            expected_geometry = expected_geometry_two_layers_3D_geo_file[i]

            TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_synchronise_geometry_2D(self, expected_geometry_two_layers_2D_after_sync: Tuple[Geometry, Geometry,
                                                                                             Geometry],
                                     create_default_2d_soil_material: SoilMaterial):
        """
        Test if the geometry is synchronised correctly in 2D after adding a new layer to the model. Where the new layer
        overlaps with the existing layer, the existing layer is cut and the overlapping part is removed.

        Args:
            - expected_geometry_two_layers_2D_after_sync (Tuple[:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`, :class:`stem.geometry.Geometry`]): The expected geometry after \
                synchronising the geometry.
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # define layer coordinates
        ndim = 2
        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0.5, 1, 0), (0.5, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_2d_soil_material
        soil_material1.name = "soil1"

        soil_material2 = create_default_2d_soil_material
        soil_material2.name = "soil2"

        # create model
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        # synchronise geometry and recalculates the ids
        model.synchronise_geometry()

        # collect all generated geometries
        generated_geometries = [model.body_model_parts[0].geometry, model.body_model_parts[1].geometry, model.geometry]

        # check if geometry is added correctly for each layer
        for generated_geometry, expected_geometry in zip(generated_geometries,
                                                         expected_geometry_two_layers_2D_after_sync):

            TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_synchronise_geometry_3D(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if the geometry is synchronised correctly in 3D after adding a new layer to the model. Where the new layer
        overlaps with the existing layer, the existing layer is cut and the overlapping part is removed.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # define layer coordinates
        ndim = 3
        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0.5, 1, 0), (0.5, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        soil_material2 = create_default_3d_soil_material
        soil_material2.name = "soil2"

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        # synchronise geometry and recalculates the ids
        model.synchronise_geometry()

        with open("tests/test_data/expected_geometry_after_sync_3D.pickle", "rb") as f:
            expected_geometry_two_layers_3D_after_sync = pickle.load(f)

        # collect all generated geometries
        generated_geometries = [model.body_model_parts[0].geometry, model.body_model_parts[1].geometry, model.geometry]

        # check if geometry is added correctly for each layer
        for generated_geometry, expected_geometry in zip(generated_geometries,
                                                         expected_geometry_two_layers_3D_after_sync):

            TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_point_loads_to_2_points(self, create_default_point_load_parameters: PointLoad):
        """
        Test if a single soil point load is added correctly to the model. Two points are generated
        and a single load is created and added to the model.

        Args:
            - create_default_point_load_properties (:class:`stem.load.PointLoad`): default point load parameters

        """

        ndim = 3

        point_coordinates = [(-0.5, 0, 0), (0.5, 0, 0)]

        # define soil material
        load_parameters = create_default_point_load_parameters

        # create model
        model = Model(ndim)
        # add point load
        model.add_load_by_coordinates(point_coordinates, load_parameters, "point_load_1")

        # check if layer is added correctly
        assert len(model.process_model_parts) == 1
        assert model.process_model_parts[0].name == "point_load_1"
        TestUtils.assert_dictionary_almost_equal(model.process_model_parts[0].parameters.__dict__,
                                                 load_parameters.__dict__)

        # check if geometry is added correctly
        generated_geometry = model.process_model_parts[0].geometry
        expected_geometry = Geometry(points={1: Point.create([-0.5, 0, 0], 1), 2: Point.create([0.5, 0, 0], 2)})

        TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_line_load_to_3_edges(self, expected_geometry_line_load: Geometry,
                                      create_default_line_load_parameters: PointLoad):
        """
        Test if a line load is added correctly to the model when applied on 3 edges. 4 points are generated
        and a single soil material is created and added to the model.

        Args:
            - expected_geometry_line_load (:class:`stem.geometry.Geometry`): expected geometry of the model
            - create_default_line_load_parameters (:class:`stem.load.LineLoad`): default line load parameters

        """

        ndim = 3

        point_coordinates = [(0, 0, 0), (3, 0, 0), (4, -1, 0), (10, -1, 0)]

        # define soil material
        load_parameters = create_default_line_load_parameters

        # create model
        model = Model(ndim)
        # add line load
        model.add_load_by_coordinates(point_coordinates, load_parameters, "line_load_1")

        # check if layer is added correctly
        assert len(model.process_model_parts) == 1
        assert model.process_model_parts[0].name == "line_load_1"
        TestUtils.assert_dictionary_almost_equal(model.process_model_parts[0].parameters.__dict__,
                                                 load_parameters.__dict__)
        # check if geometry is added correctly
        generated_geometry = model.process_model_parts[0].geometry
        expected_geometry = expected_geometry_line_load

        TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_add_moving_point_load(self, expected_geometry_line_load: Geometry,
                                   create_default_moving_load_parameters: MovingLoad):
        """
        Test if a single soil point load is added correctly to the model. Two points are generated
        and a single load is created and added to the model.
        Args:
            - expected_geometry_line_load (:class:`stem.geometry.Geometry`): expected geometry of the model
            - create_default_moving_load_parameters (:class:`stem.load.MovingLoad`): default moving load parameters
        """

        ndim = 3

        point_coordinates = [(0, 0, 0), (3, 0, 0), (4, -1, 0), (10, -1, 0)]
        # origin is in (3.5, -0.5, 0) thus in the trajectory

        # define soil material
        load_parameters = create_default_moving_load_parameters

        # create model
        model = Model(ndim)
        # add moving load
        model.add_load_by_coordinates(point_coordinates, load_parameters, "moving_load_1")

        # check if layer is added correctly
        assert len(model.process_model_parts) == 1
        assert model.process_model_parts[0].name == "moving_load_1"
        TestUtils.assert_dictionary_almost_equal(model.process_model_parts[0].parameters.__dict__,
                                                 load_parameters.__dict__)

        # check if geometry is added correctly
        generated_geometry = model.process_model_parts[0].geometry
        expected_geometry = expected_geometry_line_load

        TestUtils.assert_almost_equal_geometries(expected_geometry, generated_geometry)

    def test_validation_moving_load(self, create_default_moving_load_parameters: MovingLoad):
        """
        Test validation of moving load when points is not collinear to the trajectory.

        Args:
            - create_default_moving_load_parameters (:class:`stem.load.MovingLoad`): default moving load parameters

        """

        ndim = 3

        point_coordinates = [(0.0, 0, 0), (1, 0, 0), (2, 0, 0), (4, 0, 0)]
        # origin is in (1.5, 0.5, 0) thus not in the trajectory

        # define soil material
        load_parameters = create_default_moving_load_parameters
        # create model
        model = Model(ndim)

        with pytest.raises(ValueError,
                           match="None of the lines are aligned with the origin of the moving load. Error."):
            model.add_load_by_coordinates(point_coordinates, load_parameters, "moving_load_1")

    def test_generate_mesh_with_only_a_body_model_part_2d(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if the mesh is generated correctly in 2D if there is only one body model part.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(2)

        # add soil material
        soil_material = create_default_2d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.synchronise_geometry()

        # generate mesh
        model.generate_mesh()

        mesh = model.body_model_parts[0].mesh

        assert mesh.ndim == 2

        unique_element_ids = []
        # check if mesh is generated correctly, i.e. if the number of elements is correct and if the element type is
        # correct and if the element ids are unique and if the number of nodes per element is correct
        assert len(mesh.elements) == 162
        for element_id, element in mesh.elements.items():
            assert element.element_type == "TRIANGLE_3N"
            assert element_id not in unique_element_ids
            assert len(element.node_ids) == 3
            unique_element_ids.append(element.id)

        # check if nodes are generated correctly, i.e. if there are nodes in the mesh and if the node ids are unique
        # and if the number of coordinates per node is correct
        unique_node_ids = []
        assert len(mesh.nodes) == 98
        for node_id, node in mesh.nodes.items():
            assert node_id not in unique_node_ids
            assert len(node.coordinates) == 3
            unique_node_ids.append(node.id)

    def test_add_output_to_non_existing_model_part(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if output nodes are correctly accounted for when meshing a surface.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # define layer coordinates
        ndim = 2
        layer1_coordinates = [(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)]

        # define soil materials
        soil_material1 = create_default_2d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")

        # synchronise geometry and recalculates the ids
        model.synchronise_geometry()
        # define output object
        # Nodal results
        nodal_results = [NodalOutput.ACCELERATION]
        # add outputs to existing model part
        model.add_output_settings(part_name="layer1",
                                  output_name="gid_nodal_accelerations_top",
                                  output_dir="dir_test",
                                  output_parameters=GiDOutputParameters(file_format="binary",
                                                                        output_interval=100,
                                                                        nodal_results=nodal_results))
        # add output to non-existing model part
        msg = "Model part for which output needs to be requested doesn't exist."
        with pytest.raises(ValueError, match=msg):
            model.add_output_settings(part_name="layer2",
                                      output_name="gid_nodal_accelerations_top",
                                      output_dir="dir_test",
                                      output_parameters=GiDOutputParameters(file_format="binary",
                                                                            output_interval=100,
                                                                            nodal_results=nodal_results))

    def test_add_output_to_a_surface_2d(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if output nodes are correctly accounted for when meshing a surface.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # define layer coordinates
        ndim = 2
        layer1_coordinates = [(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)]

        # define soil materials
        soil_material1 = create_default_2d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")

        # synchronise geometry and recalculates the ids
        model.synchronise_geometry()
        # Define nodal results
        nodal_results = [NodalOutput.ACCELERATION]
        # Define output coordinates
        output_coordinates = [(1.5, 1, 0), (1.5, 0.5, 0), (2.5, 0.5, 0), (2.5, 0, 0)]

        # add output settings
        model.add_output_settings_by_coordinates(output_coordinates,
                                                 part_name="nodal_accelerations",
                                                 output_name="json_nodal_accelerations_top",
                                                 output_dir="dir_test",
                                                 output_parameters=JsonOutputParameters(output_interval=100,
                                                                                        nodal_results=nodal_results))
        model.synchronise_geometry()
        model.generate_mesh()

        unique_element_ids = []
        unique_node_ids = []

        part = model.body_model_parts[0]
        assert part.mesh.ndim == 2

        # check if mesh is generated correctly, i.e. if the number of elements is correct and if the element type is
        # correct and if the element ids are unique and if the number of nodes per element is correct
        assert len(part.mesh.elements) == 98

        for element_id, element in part.mesh.elements.items():
            assert element.element_type == "TRIANGLE_3N"
            assert element_id not in unique_element_ids
            assert len(element.node_ids) == 3
            unique_element_ids.append(element.id)

        # check if nodes are generated correctly, i.e. if there are nodes in the mesh and if the node ids are unique
        # and if the number of coordinates per node is correct
        assert len(part.mesh.nodes) == 64
        for node_id, node in part.mesh.nodes.items():
            assert node_id not in unique_node_ids
            assert len(node.coordinates) == 3
            unique_node_ids.append(node.id)

        # assert the output parts
        part = model.process_model_parts[0]
        assert part.mesh.ndim == 1

        unique_node_ids = []
        # check if nodes are generated correctly, number of nodes are equal to the one requested in output,
        # no elements generated, unique node ids, and correct number of coordinates per node
        assert len(part.mesh.nodes) == len(output_coordinates)

        for (node_id, node), actual_output_coordinates in zip(part.mesh.nodes.items(), output_coordinates):
            assert node_id not in unique_node_ids
            assert len(node.coordinates) == 3
            # assert that the order of the nodes in the new model part is the same as the one in input
            # meaning, the coordinate of the output nodes has to match one-by-one with the requested output nodes
            npt.assert_almost_equal(actual_output_coordinates, node.coordinates)
            unique_node_ids.append(node.id)

        assert part.mesh.elements == {}

    def test_add_output_to_a_surface_3d(self, create_default_3d_soil_material: SoilMaterial,
                                        create_default_outputs: Output):
        """
        Test if output nodes are correctly accounted for when meshing a surface in 3d.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.
            - create_default_outputs (:class:`stem.output.Output`): the output object containing the \
                output info.
        """

        # define layer coordinates
        ndim = 3
        layer1_coordinates = [(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)
        model.extrusion_length = 4

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "soil1")

        # synchronise geometry and recalculates the ids
        model.synchronise_geometry()
        # Define nodal results
        nodal_results = [NodalOutput.ACCELERATION]
        # Define output coordinates
        output_coordinates = [(0, 1, 2), (2, 1, 2), (4, 1, 2)]

        # add output settings
        model.add_output_settings_by_coordinates(output_coordinates,
                                                 part_name="nodal_accelerations",
                                                 output_name="json_nodal_accelerations_top",
                                                 output_dir="dir_test",
                                                 output_parameters=JsonOutputParameters(output_interval=100,
                                                                                        nodal_results=nodal_results))

        model.set_mesh_size(1)
        model.synchronise_geometry()

        model.generate_mesh()

        unique_element_ids = []
        unique_node_ids = []

        body_model_part = model.body_model_parts[0]
        assert body_model_part.mesh.ndim == 3

        # check if mesh is generated correctly, i.e. if the number of elements is correct and if the element type is
        # correct and if the element ids are unique and if the number of nodes per element is correct
        assert len(body_model_part.mesh.elements) == 187

        for element_id, element in body_model_part.mesh.elements.items():
            assert element.element_type == "TETRAHEDRON_4N"
            assert element_id not in unique_element_ids
            assert len(element.node_ids) == 4
            unique_element_ids.append(element.id)

        # check if nodes are generated correctly, i.e. if there are nodes in the mesh and if the node ids are unique
        # and if the number of coordinates per node is correct
        assert len(body_model_part.mesh.nodes) == 77
        for node_id, node in body_model_part.mesh.nodes.items():
            assert node_id not in unique_node_ids
            assert len(node.coordinates) == 3
            unique_node_ids.append(node.id)

        # assert the output parts
        output_model_part = model.process_model_parts[0]
        assert output_model_part.mesh.ndim == 1

        unique_node_ids = []
        # check if nodes are generated correctly, number of nodes are equal to the one requested in output,
        # no elements generated, unique node ids, and correct number of coordinates per node
        assert len(output_model_part.mesh.nodes) == len(output_coordinates)
        for (node_id, node), actual_output_coordinates in zip(output_model_part.mesh.nodes.items(), output_coordinates):
            assert node_id not in unique_node_ids
            assert len(node.coordinates) == 3
            # assert that the order of the nodes in the new model part is the same as the one in input
            # meaning, the coordinate of the output nodes has to match one-by-one with the requested output nodes
            npt.assert_almost_equal(actual_output_coordinates, node.coordinates)
            unique_node_ids.append(node.id)

        # No element outputs, so the element attribute of the mesh must be an empty dictionary
        assert output_model_part.mesh.elements == {}

    def test_generate_mesh_with_only_a_body_model_part_3d(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if the mesh is generated correctly in 3D if there is only one body model part.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(3)
        model.extrusion_length = 1

        # add soil material
        soil_material = create_default_3d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.synchronise_geometry()

        # generate mesh
        model.generate_mesh()

        mesh = model.body_model_parts[0].mesh

        assert mesh.ndim == 3

        unique_element_ids = []
        # check if mesh is generated correctly, i.e. if the number of elements is correct and if the element type is
        # correct and if the element ids are unique and if the number of nodes per element is correct
        assert len(mesh.elements) == 1120

        for element_id, element in mesh.elements.items():
            assert element.element_type == "TETRAHEDRON_4N"
            assert element_id not in unique_element_ids
            assert len(element.node_ids) == 4
            unique_element_ids.append(element.id)

        # check if nodes are generated correctly, i.e. if there are nodes in the mesh and if the node ids are unique
        # and if the number of coordinates per node is correct
        unique_node_ids = []
        assert len(mesh.nodes) == 340
        for node_id, node in mesh.nodes.items():
            assert node_id not in unique_node_ids
            assert len(node.coordinates) == 3
            unique_node_ids.append(node.id)

    def test_generate_mesh_with_body_and_process_model_part(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if the mesh is generated correctly in the body model part and a process model part.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(2)

        # add soil material
        soil_material = create_default_2d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")

        # add process geometry
        gmsh_process_input = {"process_0d": {"coordinates": [[0, 0.5, 0]], "ndim": 0}}
        model.gmsh_io.generate_geometry(gmsh_process_input, "")

        # create process model part
        process_model_part = ModelPart("process_0d")

        # set the geometry of the process model part
        process_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "process_0d")

        # add process model part
        model.process_model_parts.append(process_model_part)

        # synchronise geometry and generate mesh
        model.synchronise_geometry()
        model.generate_mesh()

        # check mesh of body model part
        mesh_body = model.body_model_parts[0].mesh

        assert mesh_body.ndim == 2

        unique_element_ids = []
        # check if mesh is generated correctly, i.e. if the number of elements is correct and if the element type is
        # correct and if the element ids are unique and if the number of nodes per element is correct
        assert len(mesh_body.elements) == 162

        for element_id, element in mesh_body.elements.items():
            assert element.element_type == "TRIANGLE_3N"
            assert element_id not in unique_element_ids
            assert len(element.node_ids) == 3
            unique_element_ids.append(element_id)

        # check if nodes are generated correctly, i.e. if there are nodes in the mesh and if the node ids are unique
        # and if the number of coordinates per node is correct
        unique_body_node_ids = []
        assert len(mesh_body.nodes) == 98

        for node_id, node in mesh_body.nodes.items():
            assert node_id not in unique_body_node_ids
            assert len(node.coordinates) == 3
            unique_body_node_ids.append(node.id)

        # check process model part
        mesh_process = model.process_model_parts[0].mesh

        assert mesh_process.ndim == 0

        # check elements of process model part, i.e. if the number of elements is correct and if the element type is
        # correct and if the element ids are unique and if the number of nodes per element is correct
        assert len(mesh_process.elements) == 1
        for element_id, element in mesh_process.elements.items():
            assert element.element_type == "POINT_1N"
            assert element_id == 1
            assert element_id not in unique_element_ids
            assert len(element.node_ids) == 1
            unique_element_ids.append(element.id)

        # check nodes of process model part, i.e. if there is 1 node in the mesh and if the node ids are present in the
        # body mesh and if the number of coordinates per node is correct
        assert len(mesh_process.nodes) == 1
        for node_id, node in mesh_process.nodes.items():
            # check if node is also available in the body mesh
            assert node_id in unique_body_node_ids
            assert len(node.coordinates) == 3

    def test_generate_mesh_2d_2_layers_and_lineload(self, create_default_line_load_parameters: LineLoad,
                                                    create_default_2d_soil_material: SoilMaterial):
        """
        Test if the mesh is generated correctly in 2D for 2 layers plus lineload and fixed bottom.

        Args:
            - create_default_line_load_parameters (:class:`stem.load.LineLoad`): default line load parameters
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(2)

        # add soil material
        soil_material = create_default_2d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.add_soil_layer_by_coordinates([(0, 1, 0), (4, 1, 0), (4, 2, 0), (0, 2, 0)], soil_material, "layer2")

        # add line load
        model.add_load_by_coordinates([(4, 2, 0), (0, 2, 0)], create_default_line_load_parameters, "line_load1")

        # add same line load in reversed order
        model.add_load_by_coordinates([(0, 2, 0), (4, 2, 0)], create_default_line_load_parameters, "line_load2")
        model.synchronise_geometry()

        # generate mesh
        model.generate_mesh()

        # check if mesh is generated correctly, i.e. if the number of elements is correct and if the element type is
        # correct, the elements are counterclockwise and the number of nodes per element is correct
        nodes = model.get_all_nodes()
        for bmp in model.body_model_parts:

            for element_id, element in bmp.mesh.elements.items():
                coordinates = [nodes[node_id].coordinates for node_id in element.node_ids]
                assert not Utils.are_2d_coordinates_clockwise(coordinates)
                assert element.element_type == "TRIANGLE_3N"
                assert len(element.node_ids) == 3

        # Check if all condition elements have a body element neighbour
        mapper_process_model_part_1 = model._Model__find_matching_body_elements_for_process_model_part(
            model.process_model_parts[0])

        mapper_process_model_part_2 = model._Model__find_matching_body_elements_for_process_model_part(
            model.process_model_parts[1])

        actual_element_ids_process_1 = [(process_element.id, body_element.id)
                                        for process_element, body_element in mapper_process_model_part_1]

        actual_element_ids_process_2 = [(process_element.id, body_element.id)
                                        for process_element, body_element in mapper_process_model_part_2]

        expected_ids = [(1, 85), (2, 116), (3, 125), (4, 95), (5, 96), (6, 124), (7, 98), (8, 100), (9, 83)]

        # check if the element ids are correct, process model part 1 and 2 should have the same element ids in the same
        # order
        np.testing.assert_equal(desired=expected_ids, actual=actual_element_ids_process_1)
        np.testing.assert_equal(desired=expected_ids, actual=actual_element_ids_process_2)

        # check order of nodes is consistent with what expected.
        node_ids_process_model_part_1 = np.array(
            [el.node_ids for el in model.process_model_parts[0].mesh.elements.values()])

        node_ids_process_model_part_2 = np.array(
            [el.node_ids for el in model.process_model_parts[1].mesh.elements.values()])

        expected_process_connectivities = np.array([[5, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35],
                                                    [35, 36], [36, 6]])

        # check if the node ids are correct, process model part 1 and 2 should have the same node ids in the same
        # order
        npt.assert_equal(node_ids_process_model_part_1, expected_process_connectivities)
        npt.assert_equal(node_ids_process_model_part_2, expected_process_connectivities)

    def test_adjusting_mesh_for_spring_dashpot_elements(self):
        """
        Test that the mesh is adjusted correctly when adding spring elements on nodes at the edges of a line.

        """
        model = Model(ndim=2)

        # add elastic spring damper element
        spring_damper = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1],
                                            NODAL_ROTATIONAL_STIFFNESS=[1, 1, 2],
                                            NODAL_DAMPING_COEFFICIENT=[1, 1, 3],
                                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[1, 1, 4])

        # create model part
        # 3 lines, one broken with a mid-point, which should result in 4 springs
        # the lines are in different size so all the line are broken in smaller lines except the last.

        top_coordinates = [(0, 1, 0), (0, 2, 0), (1, 1, 0), (2, 0.3, 0)]
        bottom_coordinates = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 0, 0)]

        gmsh_input_top = {"top_coordinates": {"coordinates": top_coordinates, "ndim": 0}}
        gmsh_input_bottom = {"bottom_coordinates": {"coordinates": bottom_coordinates, "ndim": 0}}

        model.gmsh_io.generate_geometry(gmsh_input_top, "")
        model.gmsh_io.generate_geometry(gmsh_input_bottom, "")

        # create rail pad geometries
        top_point_ids = model.gmsh_io.make_points(top_coordinates)
        bot_point_ids = model.gmsh_io.make_points(bottom_coordinates)

        spring_line_ids = [
            model.gmsh_io.create_line([top_point_id, bot_point_id])
            for top_point_id, bot_point_id in zip(top_point_ids, bot_point_ids)
        ]

        model.gmsh_io.add_physical_group("spring_damper", 1, spring_line_ids)
        # assign spring damper to geometry
        spring_damper_model_part = BodyModelPart("spring_damper")
        spring_damper_model_part.material = StructuralMaterial("spring_damper", spring_damper)
        spring_damper_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "spring_damper")

        # add model parts to model
        model.body_model_parts.append(spring_damper_model_part)
        model.synchronise_geometry()
        model.set_mesh_size(0.4)

        model.generate_mesh()

        # check the spring node ids are correct
        npt.assert_equal(list(model.body_model_parts[0].mesh.nodes), [2, 1, 5, 3, 6, 4, 7])
        # check that spring element ids are correct
        npt.assert_almost_equal(list(model.body_model_parts[0].mesh.elements), [18, 19, 20, 21])

        # remove mesh and check if raises are raised correctly
        model.body_model_parts[0].mesh = None
        # check that the function raises an error when the mesh is not generated
        expected_message = "Mesh not yet initialised. Please generate the mesh using Model.generate_mesh()"
        with pytest.raises(ValueError, match=expected_message):
            model._Model__adjust_mesh_spring_dampers()

    def test_adjusting_mesh_for_spring_dashpot_elements_with_soil_layer(self,
                                                                        create_default_2d_soil_material: SoilMaterial):
        """
        Test that the mesh is adjusted correctly when adding spring elements on top of a soil layer.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(ndim=2)

        # add soil material
        soil_material = create_default_2d_soil_material

        top_coordinates = [(0, 2, 0), (1, 2, 0), (2, 2, 0)]
        bottom_coordinates = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]

        soil_coordinates_loop = top_coordinates + bottom_coordinates[::-1]

        # add soil layers
        model.add_soil_layer_by_coordinates(soil_coordinates_loop, soil_material, "layer1")

        # add elastic spring damper element
        spring_damper = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1],
                                            NODAL_ROTATIONAL_STIFFNESS=[1, 1, 2],
                                            NODAL_DAMPING_COEFFICIENT=[1, 1, 3],
                                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[1, 1, 4])

        # generate geometries of bottom and top coordinates
        gmsh_input_top = {"top_coordinates": {"coordinates": top_coordinates, "ndim": 0}}
        gmsh_input_bottom = {"bottom_coordinates": {"coordinates": bottom_coordinates, "ndim": 0}}

        model.gmsh_io.generate_geometry(gmsh_input_top, "")
        model.gmsh_io.generate_geometry(gmsh_input_bottom, "")

        # create spring damper geometries and physical group
        top_point_ids = model.gmsh_io.make_points(top_coordinates)
        bot_point_ids = model.gmsh_io.make_points(bottom_coordinates)

        spring_line_ids = [
            model.gmsh_io.create_line([top_point_id, bot_point_id])
            for top_point_id, bot_point_id in zip(top_point_ids, bot_point_ids)
        ]

        model.gmsh_io.add_physical_group("spring_damper", 1, spring_line_ids)

        # assign spring damper to geometry
        spring_damper_model_part = BodyModelPart("spring_damper")
        spring_damper_model_part.material = StructuralMaterial("spring_damper", spring_damper)
        spring_damper_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "spring_damper")

        # add model part to model
        model.body_model_parts.append(spring_damper_model_part)
        model.synchronise_geometry()
        model.set_mesh_size(1)

        # generate mesh
        model.generate_mesh(open_gmsh_gui=False)

        # check if the soil layer is meshed correctly
        assert len(model.body_model_parts[0].mesh.nodes) == 13
        assert len(model.body_model_parts[0].mesh.elements) == 16

        # check if the spring is meshed correctly
        assert len(model.body_model_parts[1].mesh.nodes) == 6
        for node in model.body_model_parts[1].mesh.nodes.values():
            # check if spring damper node is also in soil layer and if the coordinates are the same
            assert node.id in model.body_model_parts[0].mesh.nodes.keys()
            npt.assert_almost_equal(node.coordinates, model.body_model_parts[0].mesh.nodes[node.id].coordinates)

        # check if the spring element ids are correct and not in the soil layer
        assert len(model.body_model_parts[1].mesh.elements) == 3
        for element in model.body_model_parts[1].mesh.elements.values():
            assert element.id not in model.body_model_parts[0].mesh.elements.keys()

    def test__get_line_string_end_nodes_expected_raises(self):
        """
        Test that the function to get the spring end nodes and first element raises errors correctly.

        """

        # create empty modelpart
        model = Model(ndim=2)
        model_part = ModelPart("test")

        # check that the function raises an error when the geometry is not initialised
        with pytest.raises(ValueError, match=f"Geometry of model part `test` not yet initialised."):
            model._Model__get_line_string_end_nodes(model_part)

        # check that the function raises an error when the mesh is not initialised
        model_part.geometry = Geometry()
        with pytest.raises(ValueError, match=f"Mesh of model part `test` not yet initialised."):
            model._Model__get_line_string_end_nodes(model_part)

    def test_find_next_node_along_line_elements(self):
        """
        Test that the function to find the next node along the line elements works correctly. And that it raises
        errors correctly.

        """

        # create empty model
        model = Model(ndim=2)

        # create remaining element ids in random order
        remaining_element_ids = {4, 5, 3, 2, 1}

        # create remaining node ids in random order
        remaining_node_ids = {2, 6, 4, 3, 5}

        # fill in which elements are connected to which nodes
        node_to_elements = {1: [1], 2: [2, 3], 3: [1, 2], 4: [3, 4], 5: [4, 5], 6: [5]}

        # create 5 connected line elements
        line_elements = {
            1: Element(1, "LINE_2N", [1, 3]),
            2: Element(2, "LINE_2N", [3, 2]),
            3: Element(3, "LINE_2N", [2, 4]),
            4: Element(4, "LINE_2N", [4, 5]),
            5: Element(5, "LINE_2N", [5, 6])
        }
        target_node_ids = np.array([2, 3, 4, 5, 6])

        # define expected connected nodes in correct order
        expected_connected_nodes = [3, 2, 4, 5, 6]

        # first node is the start node
        first_node = 1

        # find next node along line elements
        for i in range(len(expected_connected_nodes)):
            next_node = model._Model__find_next_node_along_line_elements(first_node, remaining_element_ids,
                                                                         remaining_node_ids, node_to_elements,
                                                                         line_elements, target_node_ids)

            assert next_node == expected_connected_nodes[i]

            first_node = next_node

        # check if error is raised because the next node cannot be found
        target_node_ids = np.array([9])

        with pytest.raises(ValueError,
                           match=re.escape("Next node along the line cannot be found. "
                                           "As it is not included in the search space")):
            _ = model._Model__find_next_node_along_line_elements(first_node, remaining_element_ids, remaining_node_ids,
                                                                 node_to_elements, line_elements, target_node_ids)

        # create a fork
        line_elements[6] = Element(6, "LINE_2N", [3, 7])
        target_node_ids = np.array([3])
        remaining_node_ids = {2, 6, 4, 3, 5, 7}
        remaining_element_ids = {1, 2, 6}
        node_to_elements[3] = [1, 2, 6]
        node_to_elements[7] = [6]

        # check if fork is detected and error is raised
        first_node = 3
        with pytest.raises(ValueError,
                           match=re.escape("There is a fork in the mesh at elements: {1, 2, 6}, "
                                           "the next node along the line cannot be found.")):
            _ = model._Model__find_next_node_along_line_elements(first_node, remaining_element_ids, remaining_node_ids,
                                                                 node_to_elements, line_elements, target_node_ids)

        # check if error is raised when not all elements are line elements
        line_elements[7] = Element(7, "TRIANGLE_3N", [3, 7, 8])
        with pytest.raises(ValueError, match=re.escape("Not all elements are line elements.")):
            _ = model._Model__find_next_node_along_line_elements(first_node, remaining_element_ids, remaining_node_ids,
                                                                 node_to_elements, line_elements, target_node_ids)

    def test_add_field_raises_errors(self, create_default_2d_soil_material: SoilMaterial):
        """
        Checks that the function to add parameter field raises errors correctly.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(2)

        # add soil material
        soil_material = create_default_2d_soil_material

        # add fake body model part with no material
        model.body_model_parts.append(BodyModelPart(name="fake part"))
        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        line_load_parameters = LineLoad(active=[True, True, True], value=[0, -1000, 0])
        model.add_load_by_coordinates(name="line_load",
                                      coordinates=[(0, 0, 0), (0, 1, 0)],
                                      load_parameters=line_load_parameters)

        # Define the field generator
        correct_rf_generator = RandomFieldGenerator(cov=0.1,
                                                    model_name="Gaussian",
                                                    v_scale_fluctuation=5,
                                                    anisotropy=0.5,
                                                    angle=0,
                                                    seed=42)

        # define the field parameters
        correct_field_parameters_json = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                                 function_type="json_file",
                                                                 field_file_names=["json_file.json"],
                                                                 field_generator=correct_rf_generator)

        # Define the field generator
        wrong_rf_generator = RandomFieldGenerator(cov=0.1,
                                                  model_name="Gaussian",
                                                  v_scale_fluctuation=5,
                                                  anisotropy=0.5,
                                                  angle=0,
                                                  seed=42)
        wrong_field_parameters_json = ParameterFieldParameters(property_names=["YOUNGS_MODULUS"],
                                                               function_type="json_file",
                                                               field_file_names=["json_file.json"],
                                                               field_generator=wrong_rf_generator)

        wrong_field_parameters_json_boolean = ParameterFieldParameters(property_names=["IS_DRAINED"],
                                                                       function_type="json_file",
                                                                       field_file_names=["json_file.json"],
                                                                       field_generator=wrong_rf_generator)

        # add random field to process model part
        msg = "The target part, `line_load`, is not a body model part."
        with pytest.raises(ValueError, match=msg):
            model.add_field(part_name="line_load", field_parameters=correct_field_parameters_json)

        # add random field to part with no material
        msg = "No material assigned to the body model part!"
        with pytest.raises(ValueError, match=msg):
            model.add_field(part_name="fake part", field_parameters=correct_field_parameters_json)

        # add random field to non-existing property
        msg = "Property YOUNGS_MODULUS is not one of the parameters of the soil material"
        with pytest.raises(ValueError, match=msg):
            model.add_field(part_name="layer1", field_parameters=wrong_field_parameters_json)

        # add random field to boolean property
        msg = "The property for which a random field needs to be generated, `IS_DRAINED` is not a numeric value."
        with pytest.raises(ValueError, match=msg):
            model.add_field(part_name="layer1", field_parameters=wrong_field_parameters_json_boolean)

    def test_random_field_generation_2d(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test the correct generation of the random field for a 2D model with one body model part.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(2)

        # add soil material
        soil_material = create_default_2d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.set_mesh_size(1)

        # Define the field generator
        random_field_generator = RandomFieldGenerator(cov=0.1,
                                                      model_name="Gaussian",
                                                      v_scale_fluctuation=1,
                                                      anisotropy=[0.5],
                                                      angle=[0],
                                                      seed=42)

        field_parameters_json = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                         function_type="json_file",
                                                         field_generator=random_field_generator)

        model.add_field(part_name="layer1", field_parameters=field_parameters_json)
        model.synchronise_geometry()

        # generate mesh
        model.generate_mesh()

        actual_rf_values = model.process_model_parts[-1].parameters.field_generator.generated_fields[0]

        # assert the number of generated values to be equal to the amount of elements of the part
        assert len(actual_rf_values) == len(model.body_model_parts[0].mesh.elements)
        # assert the generated values against the expected values
        expected_rf_values = [104971256.1059345, 113280413.42177339, 105124797.09835173, 109686556.57934019]

        npt.assert_allclose(actual=actual_rf_values, desired=expected_rf_values)

    @pytest.mark.skipif(IS_LINUX,
                        reason="The 3D random field samples different values for linux and windows, "
                        "because the mesh is slightly different. See also the test for mdpa_file in "
                        "3d in test_kratos_io.py.")
    def test_random_field_generation_3d(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test the correct generation of the random field for a 3D model with one body model part.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(3)
        model.extrusion_length = 1

        # add soil material
        soil_material = create_default_3d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.set_mesh_size(1.0)

        # Define the field generator
        random_field_generator = RandomFieldGenerator(cov=0.1,
                                                      model_name="Gaussian",
                                                      v_scale_fluctuation=1,
                                                      anisotropy=[0.5, 0.5],
                                                      angle=[0, 0],
                                                      seed=42)

        field_parameters_json = ParameterFieldParameters(property_names=["YOUNG_MODULUS"],
                                                         function_type="json_file",
                                                         field_generator=random_field_generator)

        model.add_field(part_name="layer1", field_parameters=field_parameters_json)

        model.synchronise_geometry()

        # generate mesh
        model.generate_mesh()

        actual_rf_values = model.process_model_parts[0].parameters.field_generator.generated_fields[0]

        # TODO: make test for Unix  with different values

        # assert the number of generated values to be equal to the amount of elements of the part
        assert len(actual_rf_values) == len(model.body_model_parts[0].mesh.elements)

        expected_rf_values = [
            109219152.50312316, 103358912.90787594, 105339578.47289738, 107804266.66256714, 116674453.0103657,
            121205355.8771256, 117518624.66410118, 109641232.38516402, 108150391.42392428, 93740844.72077464,
            106608642.49695791, 111016462.96330133, 95787906.70407471, 109879617.69834961, 103724463.91386327,
            92715313.3744301, 115556177.86463425, 119222050.2452586, 112966908.38899206, 94554356.2203453,
            112709106.84842391, 93573278.00303535, 100680007.50177462, 105511523.87671089
        ]

        npt.assert_allclose(actual=actual_rf_values, desired=expected_rf_values)

    def test_validate_expected_success(self):
        """
        Test if the model is validated correctly. A model is created with two process model parts which both have
        a unique name.

        """

        model = Model(2)

        model_part1 = ModelPart("test1")
        model_part2 = ModelPart("test2")

        model.process_model_parts = [model_part1, model_part2]

        model.validate()

    def test_validate_expected_fail_non_unique_names(self):
        """
        Test if the model is validated correctly. A model is created with two process model parts which both have
        the same name. This should raise a ValueError.

        """

        model = Model(2)

        model_part1 = ModelPart("test")
        model_part2 = ModelPart("test")

        model.process_model_parts = [model_part1, model_part2]

        pytest.raises(ValueError, model.validate)

    def test_validate_expected_fail_no_name(self):
        """
        Test if the model is validated correctly. A model is created with a process model part which does not contain
        a name. This should raise a ValueError.

        """

        model = Model(2)

        model_part1 = ModelPart(None)
        model.process_model_parts = [model_part1]

        pytest.raises(ValueError, model.validate)

    def test_add_boundary_condition_by_geometry_ids(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a boundary condition is added correctly to the model. A boundary condition is added to the model by
        specifying the geometry ids to which the boundary condition should be applied.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # create a 3D model
        model = Model(3)
        model.extrusion_length = 1

        # create multiple boundary condition parameters
        no_rotation_parameters = RotationConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True],
                                                    value=[0, 0, 0])

        absorbing_parameters = AbsorbingBoundary(absorbing_factors=[1, 1], virtual_thickness=0)

        no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, True, True],
                                                            value=[0, 0, 0])

        # add body model part
        soil_material = create_default_3d_soil_material
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "test_soil")

        # add boundary conditions in 0d, 1d and 2d
        model.add_boundary_condition_by_geometry_ids(0, [1, 2], no_rotation_parameters, "no_rotation")
        model.add_boundary_condition_by_geometry_ids(1, [8], absorbing_parameters, "absorbing")
        model.add_boundary_condition_by_geometry_ids(2, [1, 2], no_displacement_parameters, "no_displacement")

        model.synchronise_geometry()

        # set expected parameters of the boundary conditions
        expected_0d_model_part_parameters = RotationConstraint(active=[True, True, True],
                                                               is_fixed=[True, True, True],
                                                               value=[0, 0, 0])

        expected_1d_model_part_parameters = AbsorbingBoundary(absorbing_factors=[1, 1], virtual_thickness=0)

        expected_2d_model_part_parameters = DisplacementConstraint(active=[True, True, True],
                                                                   is_fixed=[True, True, True],
                                                                   value=[0, 0, 0])

        # set expected geometry 0d boundary condition
        expected_boundary_points = {1: Point.create([0, 0, 0], 1), 2: Point.create([1, 0, 0], 2)}
        expected_boundary_lines = {1: Line.create([1, 2], 1)}
        expected_boundary_surfaces = {}
        expected_boundary_volumes = {}

        expected_boundary_geometry_0d = Geometry(expected_boundary_points, expected_boundary_lines,
                                                 expected_boundary_surfaces, expected_boundary_volumes)

        # set expected geometry 1d boundary condition
        expected_boundary_points = {3: Point.create([1, 1, 0], 3), 7: Point.create([1, 1, 1], 7)}
        expected_boundary_lines = {8: Line.create([3, 7], 8)}
        expected_boundary_surfaces = {}
        expected_boundary_volumes = {}

        expected_boundary_geometry_1d = Geometry(expected_boundary_points, expected_boundary_lines,
                                                 expected_boundary_surfaces, expected_boundary_volumes)

        # set expected geometry 2d boundary condition

        expected_boundary_geometry_2d = Geometry()
        expected_boundary_geometry_2d.points = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([1, 0, 0], 2),
            3: Point.create([1, 1, 0], 3),
            4: Point.create([0, 1, 0], 4),
            5: Point.create([0, 0, 1], 5),
            6: Point.create([1, 0, 1], 6)
        }

        expected_boundary_geometry_2d.lines = {
            1: Line.create([1, 2], 1),
            2: Line.create([2, 3], 2),
            3: Line.create([3, 4], 3),
            4: Line.create([4, 1], 4),
            5: Line.create([1, 5], 5),
            7: Line.create([5, 6], 7),
            6: Line.create([2, 6], 6)
        }

        expected_boundary_geometry_2d.surfaces = {
            1: Surface.create([1, 2, 3, 4], 1),
            2: Surface.create([5, 7, -6, -1], 2)
        }

        expected_boundary_geometry_2d.volumes = {}

        # collect all expected geometries
        all_expected_geometries = [
            expected_boundary_geometry_0d, expected_boundary_geometry_1d, expected_boundary_geometry_2d
        ]

        # check 0d parameters
        npt.assert_allclose(model.process_model_parts[0].parameters.active, expected_0d_model_part_parameters.active)
        npt.assert_allclose(model.process_model_parts[0].parameters.is_fixed,
                            expected_0d_model_part_parameters.is_fixed)
        npt.assert_allclose(model.process_model_parts[0].parameters.value, expected_0d_model_part_parameters.value)

        # check 1d parameters
        npt.assert_allclose(model.process_model_parts[1].parameters.absorbing_factors,
                            expected_1d_model_part_parameters.absorbing_factors)
        npt.assert_allclose(model.process_model_parts[1].parameters.virtual_thickness,
                            expected_1d_model_part_parameters.virtual_thickness)

        # check 2d parameters
        npt.assert_allclose(model.process_model_parts[2].parameters.active, expected_2d_model_part_parameters.active)
        npt.assert_allclose(model.process_model_parts[2].parameters.is_fixed,
                            expected_2d_model_part_parameters.is_fixed)
        npt.assert_allclose(model.process_model_parts[2].parameters.value, expected_2d_model_part_parameters.value)

        for expected_geometry, model_part in zip(all_expected_geometries, model.process_model_parts):

            TestUtils.assert_almost_equal_geometries(expected_geometry, model_part.geometry)

    def test_add_load_by_geometry_ids(self, create_default_3d_soil_material: SoilMaterial,
                                      create_default_point_load_parameters: PointLoad,
                                      create_default_line_load_parameters: LineLoad,
                                      create_default_surface_load_parameters: SurfaceLoad):
        """
        Test if a load is added correctly to the model. Here the load is added to the model by
        specifying the geometry ids to which the load should be applied.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.
            - create_default_point_load_parameters (:class:`stem.load.PointLoad`): default point load parameters.
            - create_default_surface_load_parameters (:class:`stem.load.SurfaceLoad`): default surface load parameters.

        """

        ndim = 3

        # create a 3D model
        model = Model(ndim)
        model.extrusion_length = 1

        # set expected parameters of the load conditions
        expected_point_load_parameters = PointLoad(active=[False, True, False], value=[0, -200, 0])
        expected_line_load_parameters = LineLoad(active=[False, True, False], value=[0, -20, 0])
        expected_surface_load_parameters = SurfaceLoad(active=[False, True, False], value=[0, -2, 0])
        expected_moving_load_parameters = MovingLoad(origin=[0, 1, 0.5],
                                                     load=[0.0, -10.0, 0.0],
                                                     velocity=5.0,
                                                     offset=3.0,
                                                     direction=[1, 1, 1])

        # add body model part
        soil_material = create_default_3d_soil_material
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "test_soil")

        # add point, line, surface and moving loads by geometry id
        model.add_load_by_geometry_ids([3, 4, 7, 8], create_default_point_load_parameters, "point_loads")
        model.add_load_by_geometry_ids([3, 8, 10, 11], create_default_line_load_parameters, "line_loads")
        model.add_load_by_geometry_ids([4], create_default_surface_load_parameters, "surface_load")
        model.add_load_by_geometry_ids([3, 8, 10], expected_moving_load_parameters, "moving_load")

        # set expected geometry point load
        expected_load_points = {
            3: Point.create([1, 1, 0], 3),
            4: Point.create([0, 1, 0], 4),
            7: Point.create([1, 1, 1], 7),
            8: Point.create([0, 1, 1], 8)
        }
        expected_point_load_geometry = Geometry(expected_load_points, {}, {}, {})

        # set expected geometry line load
        expected_load_lines = {
            3: Line.create([3, 4], 3),
            8: Line.create([3, 7], 8),
            10: Line.create([4, 8], 10),
            11: Line.create([7, 8], 11)
        }

        expected_line_load_geometry = Geometry(expected_load_points, expected_load_lines, {}, {})

        # set expected geometry surface load
        expected_surface_load_geometry = Geometry()
        expected_surface_load_geometry.points = {
            3: Point.create([1, 1, 0], 3),
            7: Point.create([1, 1, 1], 7),
            8: Point.create([0, 1, 1], 8),
            4: Point.create([0, 1, 0], 4)
        }
        expected_surface_load_geometry.lines = {
            8: Line.create([3, 7], 8),
            11: Line.create([7, 8], 11),
            10: Line.create([4, 8], 10),
            3: Line.create([3, 4], 3)
        }
        expected_surface_load_geometry.surfaces = {4: Surface.create([8, 11, -10, -3], 4)}

        # set expected geometry moving load
        expected_surface_load_geometry = Geometry()
        expected_surface_load_geometry.points = {
            3: Point.create([1, 1, 0], 3),
            7: Point.create([1, 1, 1], 7),
            8: Point.create([0, 1, 1], 8),
            4: Point.create([0, 1, 0], 4)
        }
        expected_surface_load_geometry.lines = {
            8: Line.create([3, 7], 8),
            11: Line.create([7, 8], 11),
            10: Line.create([4, 8], 10),
            3: Line.create([3, 4], 3)
        }
        expected_surface_load_geometry.surfaces = {4: Surface.create([8, 11, -10, -3], 4)}

        # collect all expected geometriesl
        all_expected_geometries = [
            expected_point_load_geometry, expected_line_load_geometry, expected_surface_load_geometry
        ]

        for expected_geometry, model_part in zip(all_expected_geometries, model.process_model_parts):

            TestUtils.assert_almost_equal_geometries(expected_geometry, model_part.geometry)

        # check point load parameters
        npt.assert_allclose(model.process_model_parts[0].parameters.value, expected_point_load_parameters.value)
        npt.assert_allclose(model.process_model_parts[0].parameters.active, expected_point_load_parameters.active)

        # check line load parameters
        npt.assert_allclose(model.process_model_parts[1].parameters.value, expected_line_load_parameters.value)
        npt.assert_allclose(model.process_model_parts[1].parameters.active, expected_line_load_parameters.active)

        # check surface load parameters
        npt.assert_allclose(model.process_model_parts[2].parameters.value, expected_surface_load_parameters.value)
        npt.assert_allclose(model.process_model_parts[2].parameters.active, expected_surface_load_parameters.active)

        # check moving load parameters
        TestUtils.assert_dictionary_almost_equal(model.process_model_parts[3].parameters.__dict__,
                                                 expected_moving_load_parameters.__dict__)

    def test_add_load_by_geometry_ids_raises_error(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a load is added correctly to the model. Here the load is added to the model by
        specifying the geometry ids to which the load should be applied.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        ndim = 3

        # create a 3D model
        model = Model(ndim)
        model.extrusion_length = 1

        soil_material = create_default_3d_soil_material
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "test_soil")

        moving_load_parameters = MovingLoad(origin=[0, 1, 0.5],
                                            load=[0.0, -10.0, 0.0],
                                            velocity=5.0,
                                            offset=3.0,
                                            direction=[1, 1, 1])

        # check raising of errors
        msg = "Load parameter provided is not supported: `GravityLoad`."
        with pytest.raises(NotImplementedError, match=msg):
            model.add_load_by_geometry_ids([1], GravityLoad(value=[0, -9.81, 0], active=[True, True, True]),
                                           "gravity load")
        # lines disconnected

        msg = ("The lines defined for the moving load are not aligned on a path."
               "Discontinuities or loops/branching points are found.")
        with pytest.raises(ValueError, match=msg):
            model.add_load_by_geometry_ids([8, 10], moving_load_parameters, "moving_load_wrong_1")

        # origin not in path
        # test for branching points
        msg = "None of the lines are aligned with the origin of the moving load. Error."
        with pytest.raises(ValueError, match=msg):
            model.add_load_by_geometry_ids([3, 8, 11], moving_load_parameters, "moving_load_wrong_2")

    def test_add_gravity_load_1d_and_2d(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if a gravity load is added correctly to the model in a 2d space containing 1d and 2d elements. A gravity
        load is generated and added to the model.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # create model
        model = Model(2)

        # add a 2d layer
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0)], create_default_2d_soil_material, "soil1")

        # add a 1d layer
        layer_settings = {"beam": {"ndim": 1, "element_size": -1, "coordinates": [[0, 0, 0], [1, 0, 0]]}}

        model.gmsh_io.generate_geometry(layer_settings, "")
        model.synchronise_geometry()

        # add 1d model part to model
        body_model_part = BodyModelPart("beam")
        body_model_part.material = EulerBeam(ndim=2,
                                             YOUNG_MODULUS=1e6,
                                             POISSON_RATIO=0.3,
                                             DENSITY=1,
                                             CROSS_AREA=1,
                                             I33=1)
        body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "beam")

        model.body_model_parts.append(body_model_part)

        # add gravity load
        model._Model__add_gravity_load()

        assert len(model.process_model_parts) == 2
        assert model.process_model_parts[0].name == "gravity_load_1d"
        assert model.process_model_parts[1].name == "gravity_load_2d"

        # setup expected geometries for 1d and 2d
        expected_geometry_points_1d = {1: Point.create([0, 0, 0], 1), 2: Point.create([1, 0, 0], 2)}
        expected_geometry_lines_1d = {1: Line.create([1, 2], 1)}
        expected_geometry_gravity_1d = Geometry(expected_geometry_points_1d, expected_geometry_lines_1d, {}, {})

        expected_geometry_points_2d = {
            1: Point.create([0, 0, 0], 1),
            2: Point.create([1, 0, 0], 2),
            3: Point.create([1, 1, 0], 3)
        }
        expected_geometry_lines_2d = {1: Line.create([1, 2], 1), 2: Line.create([2, 3], 2), 3: Line.create([3, 1], 3)}
        expected_geometry_surfaces_2d = {1: Surface.create([1, 2, 3], 1)}
        expected_geometry_gravity_2d = Geometry(expected_geometry_points_2d, expected_geometry_lines_2d,
                                                expected_geometry_surfaces_2d, {})

        expected_geometries = [expected_geometry_gravity_1d, expected_geometry_gravity_2d]

        # check if all process model parts are correct
        for model_part in model.process_model_parts:

            # check if parameters are added correctly
            npt.assert_allclose(model_part.parameters.value, [0, -9.81, 0])
            npt.assert_allclose(model_part.parameters.active, [True, True, True])

            # check if geometry is added correctly
            generated_model_part = model_part.geometry

            TestUtils.assert_almost_equal_geometries(expected_geometries[0], generated_model_part)

    def test_add_gravity_load_two_layers_same_dimension(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if a gravity load is added correctly to the model in a 2d space containing 2 layers. A gravity load is
        generated and added to the model.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # create model
        model = Model(2)

        # add a 2d layer
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0)], create_default_2d_soil_material, "soil1")
        model.add_soil_layer_by_coordinates([(1, 0, 0), (0, 0, 0), (1, -1, 0)], create_default_2d_soil_material,
                                            "soil2")

        model.synchronise_geometry()

        # add gravity load
        model._Model__add_gravity_load()

        assert len(model.process_model_parts) == 1

        generated_geometry = model.process_model_parts[0].geometry

        # check if number of points, lines, surfaces are correct, i.e. if the number of points, lines, surfaces are the
        # same as the number of points, lines, surfaces of the model geometry
        assert len(generated_geometry.points) == len(model.geometry.points) == 4
        assert len(generated_geometry.lines) == len(model.geometry.lines) == 5
        assert len(generated_geometry.surfaces) == len(model.geometry.surfaces) == 2

        assert model.process_model_parts[0].name == "gravity_load_2d"
        npt.assert_allclose(model.process_model_parts[0].parameters.value, [0, -9.81, 0])
        npt.assert_allclose(model.process_model_parts[0].parameters.active, [True, True, True])

    def test_add_gravity_load_3d(self, create_default_3d_soil_material):
        """
        Test if a gravity load is added correctly to the model in a 3d space. A gravity load is generated and added to
        the model.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # create model
        model = Model(3)
        model.extrusion_length = 1

        # add a 2d layer
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0)], create_default_3d_soil_material, "soil1")

        model.synchronise_geometry()

        # add gravity load
        model._Model__add_gravity_load()

        assert len(model.process_model_parts) == 1

        generated_geometry = model.process_model_parts[0].geometry

        # check if number of points, lines, surfaces are correct, i.e. if the number of points, lines, surfaces and
        # volumes are the same as the number of points, lines, surfaces and volumes of the model geometry
        assert len(generated_geometry.points) == len(model.geometry.points) == 6
        assert len(generated_geometry.lines) == len(model.geometry.lines) == 9
        assert len(generated_geometry.surfaces) == len(model.geometry.surfaces) == 5
        assert len(generated_geometry.volumes) == len(model.geometry.volumes) == 1

        assert model.process_model_parts[0].name == "gravity_load_3d"
        npt.assert_allclose(model.process_model_parts[0].parameters.value, [0, -9.81, 0])
        npt.assert_allclose(model.process_model_parts[0].parameters.active, [True, True, True])

    def test_setup_stress_initialisation(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if the stress initialisation is set up correctly. A model is created with a soil layer. It is checked if
        gravity is added in case the K0 procedure or gravity loading is used.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # set up solver settings
        analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW

        solution_type = SolutionType.QUASI_STATIC

        time_integration = TimeIntegration(start_time=0.0,
                                           end_time=1.0,
                                           delta_time=0.1,
                                           reduction_factor=0.5,
                                           increase_factor=2.0,
                                           max_delta_time_factor=500)

        convergence_criterion = DisplacementConvergenceCriteria()

        stress_initialisation_type = StressInitialisationType.NONE

        solver_settings = SolverSettings(analysis_type=analysis_type,
                                         solution_type=solution_type,
                                         stress_initialisation_type=stress_initialisation_type,
                                         time_integration=time_integration,
                                         is_stiffness_matrix_constant=True,
                                         are_mass_and_damping_constant=True,
                                         convergence_criteria=convergence_criterion)

        # set up problem data
        problem_data = Problem(problem_name="test", number_of_threads=2, settings=solver_settings)

        model_no_gravity = Model(2)
        model_no_gravity.project_parameters = problem_data

        # set up soil material
        soil_material = create_default_2d_soil_material
        model_no_gravity.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0)], soil_material, "soil1")
        model_no_gravity.synchronise_geometry()

        # setup_stress_initialisation
        model_no_gravity._Model__setup_stress_initialisation()

        model_k0 = Model(2)
        model_k0.project_parameters = problem_data

        model_k0.project_parameters.settings.stress_initialisation_type = StressInitialisationType.K0_PROCEDURE
        model_k0.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0)], soil_material, "soil1")
        model_k0.synchronise_geometry()

        # setup_stress_initialisation
        model_k0._Model__setup_stress_initialisation()

        model_gravity_loading = Model(2)
        model_gravity_loading.project_parameters = problem_data

        model_gravity_loading.project_parameters.settings.stress_initialisation_type = \
            StressInitialisationType.GRAVITY_LOADING
        model_gravity_loading.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0)], soil_material, "soil1")
        model_gravity_loading.synchronise_geometry()

        # setup_stress_initialisation
        model_gravity_loading._Model__setup_stress_initialisation()

        assert len(model_no_gravity.process_model_parts) == 0
        assert len(model_k0.process_model_parts) == 1
        assert len(model_gravity_loading.process_model_parts) == 1

        assert model_k0.process_model_parts[0].name == "gravity_load_2d"
        assert model_gravity_loading.process_model_parts[0].name == "gravity_load_2d"

    def test_setup_stress_initialisation_without_project_parameters(self):
        """
        A model is created without project parameters. It is
        checked if a ValueError is raised while setting up the stress initialisation.

        """
        # create model
        model = Model(2)

        # test if value error is raised
        with pytest.raises(ValueError,
                           match=r"Project parameters must be set before setting up the stress initialisation"):
            model._Model__setup_stress_initialisation()

    def test_check_ordering_process_model_part_2d(self):
        """
        Test if the node order of the process model part is flipped, such that the nodes follow the same order as
        the neighbour element. After filling in the nodes of the process model part in reverse order.

        """

        # create model
        model = Model(2)

        # manually set mesh data nodes
        model.gmsh_io._GmshIO__mesh_data = {"nodes": {1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 1, 0]}}

        # manually create process model part with nodes in reverse order
        process_element = Element(1, "LINE_2N", [2, 1])
        process_model_part = ModelPart("process")
        process_mesh = Mesh(1)
        process_mesh.elements = {1: process_element}
        process_mesh.nodes = {1: Node(1, [0, 0, 0]), 2: Node(2, [1, 0, 0])}
        process_model_part.mesh = process_mesh
        model.process_model_parts = [ModelPart("process")]

        # create body_model_part
        body_element = Element(2, "TRIANGLE_3N", [1, 2, 3])

        # check ordering of process model part connectivities
        mapper = [(process_element, body_element)]
        model._Model__check_ordering_process_model_part(mapper, process_model_part)

        # check if the node ids of the process model part are in the correct order
        assert process_model_part.mesh.elements[1].node_ids == [1, 2]

    def test_check_ordering_process_model_part_2d_multiple_elements(self):
        """
        Test if the node order of the first element in process model part is flipped in a 2D case. The second element
        should not be flipped. This test check for any order of the process element to body element mapping.

        """

        # create model
        model = Model(2)

        # manually set mesh data nodes
        model.gmsh_io._GmshIO__mesh_data = {"nodes": {1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 1, 0]}}

        # manually create process model part with nodes in reverse order
        process_element1 = Element(1, "LINE_2N", [2, 1])
        process_element2 = Element(2, "LINE_2N", [2, 3])
        process_model_part = ModelPart("process")
        process_mesh = Mesh(1)
        process_mesh.elements = {1: process_element1, 2: process_element2}
        process_mesh.nodes = {1: Node(1, [0, 0, 0]), 2: Node(2, [1, 0, 0]), 3: Node(3, [1, 1, 0])}
        process_model_part.mesh = process_mesh
        model.process_model_parts = [ModelPart("process")]

        # create body_model_part
        body_element = Element(2, "TRIANGLE_3N", [1, 2, 3])

        # check ordering of process model part connectivities
        mapper = [(process_element1, body_element), (process_element2, body_element)]
        model._Model__check_ordering_process_model_part(mapper, process_model_part)

        # check if the node ids of the process model part are in the correct order, i.e. the node order of only the
        # first element should be flipped
        assert process_model_part.mesh.elements[1].node_ids == [1, 2]
        assert process_model_part.mesh.elements[2].node_ids == [2, 3]

        # redefine process elements and redefine mapper in reverse order
        # manually create process model part with nodes in outwards normal order
        process_element1 = Element(1, "LINE_2N", [2, 1])
        process_element2 = Element(2, "LINE_2N", [2, 3])
        process_mesh.elements = {1: process_element1, 2: process_element2}

        # add process_element and body_element to mapper
        mapper = [(process_element2, body_element), (process_element1, body_element)]

        # check ordering of process model part connectivities
        model._Model__check_ordering_process_model_part(mapper, process_model_part)

        # check if the node ids of the process model part are in the correct order, i.e. the node order of only the
        # first element should be flipped (the same order as before)
        assert process_model_part.mesh.elements[1].node_ids == [1, 2]
        assert process_model_part.mesh.elements[2].node_ids == [2, 3]

    def test_check_ordering_process_model_part_3d(self):
        """
        Test if the node order of the process model part is flipped, such that the normal is inwards. After filling in
        the nodes of the process model part in outwards normal order.

        """

        # create model
        model = Model(3)

        # manually set mesh data nodes
        model.gmsh_io._GmshIO__mesh_data = {"nodes": {1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 0, 1]}}

        # manually create process model part with nodes in outwards normal order
        process_element = Element(1, "TRIANGLE_3N", [2, 1, 3])
        process_model_part = ModelPart("process")
        process_mesh = Mesh(1)
        process_mesh.elements = {1: process_element}
        process_mesh.nodes = {1: Node(1, [0, 0, 0]), 2: Node(2, [1, 0, 0])}
        process_model_part.mesh = process_mesh
        model.process_model_parts = [ModelPart("process")]

        # create body_model_part
        body_element = Element(2, "TETRAHEDRON_4N", [1, 2, 3, 4])

        # check ordering of process model part connectivities
        mapper = [(process_element, body_element)]
        model._Model__check_ordering_process_model_part(mapper, process_model_part)

        # check if the node ids of the process model part are in the correct order, i.e. the node order should be
        # flipped, such that the normal is inwards
        assert process_model_part.mesh.elements[1].node_ids == [2, 1, 3]

    def test_check_ordering_process_model_part_3d_multiple_elements(self):
        """
        Test if the node order of the first element in process model part is flipped, such that the normal is inwards.
        The second element should not be flipped. This test check for any order of the process element to body element
        mapping.

        """

        # create model
        model = Model(3)

        # manually set mesh data nodes
        model.gmsh_io._GmshIO__mesh_data = {"nodes": {1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 0, 1]}}

        # manually create process model part with nodes in outwards normal order
        process_element1 = Element(1, "TRIANGLE_3N", [2, 1, 3])
        process_element2 = Element(2, "TRIANGLE_3N", [4, 3, 2])
        process_model_part = ModelPart("process")
        process_mesh = Mesh(2)
        process_mesh.elements = {1: process_element1, 2: process_element2}
        process_mesh.nodes = {
            1: Node(1, [0, 0, 0]),
            2: Node(2, [1, 0, 0]),
            3: Node(3, [1, 1, 0]),
            4: Node(4, [0, 0, 1])
        }
        process_model_part.mesh = process_mesh
        model.process_model_parts = [ModelPart("process")]

        # create body_model_part
        body_element = Element(2, "TETRAHEDRON_4N", [1, 2, 3, 4])

        # add process_element and body_element to mapper
        mapper = [(process_element1, body_element), (process_element2, body_element)]

        # check ordering of process model part connectivities
        model._Model__check_ordering_process_model_part(mapper, process_model_part)

        # check if the node ids of the process model part are in the correct order, i.e. the node order of only the
        # first element should be flipped, such that the normal is inwards
        assert process_model_part.mesh.elements[1].node_ids == [2, 1, 3]
        assert process_model_part.mesh.elements[2].node_ids == [2, 3, 4]

        # redefine process elements and redefine mapper in reverse order
        # manually create process model part with nodes in outwards normal order
        process_element1 = Element(1, "TRIANGLE_3N", [2, 1, 3])
        process_element2 = Element(2, "TRIANGLE_3N", [4, 3, 2])
        process_mesh.elements = {1: process_element1, 2: process_element2}

        # add process_element and body_element to mapper
        mapper = [(process_element2, body_element), (process_element1, body_element)]

        # check ordering of process model part connectivities
        model._Model__check_ordering_process_model_part(mapper, process_model_part)

        # check if the node ids of the process model part are in the correct order, i.e. the node order of only the
        # first element should be flipped, such that the normal is inwards (the same order as before)
        assert process_model_part.mesh.elements[1].node_ids == [2, 1, 3]
        assert process_model_part.mesh.elements[2].node_ids == [2, 3, 4]

    def test_show_geometry_file(self, create_default_3d_soil_material):
        """
        Test if the geometry html file is generated. A model is created with a soil layer. The geometry is plotted to a
         html file and the file is checked if it is created.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        # define soil material
        soil_material = create_default_3d_soil_material

        # create model
        model = Model(3)
        model.extrusion_length = 1

        # add soil layer
        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")
        model.synchronise_geometry()

        model.show_geometry(file_name=r"tests/test_geometry.html", auto_open=False)

        # check if the file is created with pathlib
        assert Path(r"tests/test_geometry.html").exists()

        # remove file
        Path(r"tests/test_geometry.html").unlink()

    def test_post_setup_with_gravity_2D(self, expected_geometry_two_layers_2D: Tuple[Geometry, Geometry, Geometry],
                                        create_default_2d_soil_material: SoilMaterial):
        """
        Tests if gravity loading and zero water pressure is added correctly when using post setup. Gravity load and zero
        water pressure should be present on all nodes of the model.

        Args:
            - expected_geometry_single_layer_2D (Tuple[:class:`stem.geometry.Geometry`, \
              :class:`stem.geometry.Geometry`, :class:`stem.geometry.Geometry`]): expected geometry of the model
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

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
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        # set up gravity loading
        project_parameters = TestUtils.create_default_solver_settings()
        model.project_parameters = project_parameters
        model.project_parameters.settings.stress_initialisation_type = StressInitialisationType.GRAVITY_LOADING

        # add gravity through post setup
        model.post_setup()

        # get water and gravity model parts
        water_pressure_model_part = model.process_model_parts[0]
        gravity_model_part = model.process_model_parts[1]

        # assert if the water and gravity model part are the same as the expected model parts
        expected_water_pressure_geometry = expected_geometry_two_layers_2D[-1]
        expected_gravity_geometry = expected_geometry_two_layers_2D[-1]

        TestUtils.assert_almost_equal_geometries(expected_water_pressure_geometry, water_pressure_model_part.geometry)
        TestUtils.assert_almost_equal_geometries(expected_gravity_geometry, gravity_model_part.geometry)

        assert water_pressure_model_part.name == "zero_water_pressure"
        assert gravity_model_part.name == "gravity_load_2d"

        assert pytest.approx(water_pressure_model_part.parameters.water_pressure) == 0
        assert water_pressure_model_part.parameters.is_fixed

        npt.assert_allclose([0, -9.81, 0], gravity_model_part.parameters.value)
        npt.assert_allclose([True, True, True], gravity_model_part.parameters.active)

    def test_post_setup_with_water_pressure_3D(self, expected_geometry_two_layers_3D_extruded: Tuple[Geometry,
                                                                                                     Geometry],
                                               create_default_3d_soil_material: SoilMaterial):
        """
        Tests if gravity loading is not applied and zero water pressure is not added when using post setup. Water pressure
        should only be present on layer 1.

        Args:
            - expected_geometry_two_layers_3D_extruded (Tuple[:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`]): expected geometry of the model
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3

        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        soil_material2 = create_default_3d_soil_material
        soil_material2.name = "soil2"

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material2, "layer2")

        # manually add water pressure model part
        water_pressure_model_part = ModelPart("water_pressure_part")
        water_pressure_model_part.geometry = model.body_model_parts[0].geometry
        water_pressure_model_part.parameters = UniformWaterPressure(water_pressure=100)
        model.process_model_parts.append(water_pressure_model_part)

        model.gmsh_io.add_physical_group(
            "water_pressure_part", 3, geometry_ids=model.gmsh_io.geo_data["physical_groups"]["layer1"]["geometry_ids"])

        # add project parameters
        project_parameters = TestUtils.create_default_solver_settings()

        model.project_parameters = project_parameters
        model.post_setup()

        # only 1 process model part should be present which is the water pressure on layer 1
        assert len(model.process_model_parts) == 1

        # check if the water pressure model part is the same as the expected model part
        assert model.process_model_parts[0].name == "water_pressure_part"
        assert pytest.approx(model.process_model_parts[0].parameters.water_pressure) == 100
        assert model.process_model_parts[0].parameters.is_fixed

        # check if the water pressure is only applied to the nodes of layer 1
        expected_water_pressure_geometry = expected_geometry_two_layers_3D_extruded[0]
        TestUtils.assert_almost_equal_geometries(expected_water_pressure_geometry,
                                                 model.process_model_parts[0].geometry)

    def test_post_setup_only_structural_material(self):
        """
        Test if the post setup is done correctly when only structural materials are present. Gravity loading should
        be applied, but no water pressures

        """

        ndim = 2

        model = Model(ndim)

        # Specify beam material model
        beam_material = EulerBeam(ndim, 210e9, 0.3, 7850, 0.01, 0.0001)
        name = "beam"
        structural_material = StructuralMaterial(name, beam_material)
        # Specify the coordinates for the beam: x:1m x y:0m
        beam_coordinates = [(0, 0, 0), (1, 0, 0)]
        # Create the beam
        gmsh_input = {name: {"coordinates": beam_coordinates, "ndim": 1}}
        # check if extrusion length is specified in 3D
        model.gmsh_io.generate_geometry(gmsh_input, "")
        #
        # create body model part
        body_model_part = BodyModelPart(name)
        body_model_part.material = structural_material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, name)
        model.body_model_parts.append(body_model_part)

        # add project parameters and set up gravity loading
        project_parameters = TestUtils.create_default_solver_settings()
        project_parameters.settings.stress_initialisation_type = StressInitialisationType.GRAVITY_LOADING

        model.project_parameters = project_parameters

        # add gravity through post setup, do not add water pressure through post setup
        model.post_setup()

        # set expected geometry
        expected_geometry = Geometry()
        expected_geometry.points = {1: Point.create([0, 0, 0], 1), 2: Point.create([1, 0, 0], 2)}
        expected_geometry.lines = {1: Line.create([1, 2], 1)}

        # check if only gravity process model part is present, no water pressure should be present
        assert len(model.process_model_parts) == 1
        assert model.process_model_parts[0].name == "gravity_load_1d"
        assert model.process_model_parts[0].parameters.value == [0, -9.81, 0]
        assert model.process_model_parts[0].parameters.active == [True, True, True]

        # check if the geometry of the process model part is correct
        TestUtils.assert_almost_equal_geometries(expected_geometry, model.process_model_parts[0].geometry)

    def test_generate_straight_track_2d(self):
        """
        Test if a straight track is generated correctly in a 2d space. A straight track is generated and added to the
        model. The geometry and material of the rails, sleepers and rail pads are checked.
        """

        # initialise model
        model = Model(2)

        rail_parameters = EulerBeam(2, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        sleeper_parameters = NodalConcentrated([1, 1, 1], 1, [1, 1, 1])

        origin_point = np.array([2.0, 3.0, 0])
        direction_vector = np.array([1, 0, 0])

        # create a straight track with rails, sleepers and rail pads
        model.generate_straight_track(0.6, 3, rail_parameters, sleeper_parameters, rail_pad_parameters, 0.02,
                                      origin_point, direction_vector, "track_1")

        # check geometry and material of the rail
        expected_rail_points = {
            4: Point.create([2.0, 3.02, 0], 4),
            5: Point.create([2.6, 3.02, 0], 5),
            6: Point.create([3.2, 3.02, 0], 6)
        }
        expected_rail_lines = {3: Line.create([4, 5], 3), 4: Line.create([5, 6], 4)}

        expected_rail_geometry = Geometry(expected_rail_points, expected_rail_lines)

        # check rail model part
        rail_model_part = model.body_model_parts[0]
        calculated_rail_geometry = rail_model_part.geometry
        calculated_rail_parameters = rail_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_geometry, calculated_rail_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_parameters.__dict__, calculated_rail_parameters.__dict__)

        # check geometry and material of the sleepers
        expected_sleeper_points = {
            1: Point.create([2.0, 3.0, 0], 1),
            2: Point.create([2.6, 3.0, 0], 2),
            3: Point.create([3.2, 3.0, 0], 3)
        }
        expected_sleeper_geometry = Geometry(expected_sleeper_points)

        sleeper_model_part = model.body_model_parts[1]
        calculated_sleeper_geometry = sleeper_model_part.geometry
        calculated_sleeper_parameters = sleeper_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_sleeper_geometry, calculated_sleeper_geometry)
        TestUtils.assert_dictionary_almost_equal(sleeper_parameters.__dict__, calculated_sleeper_parameters.__dict__)

        # check geometry and material of the rail pads
        rail_pad_model_part = model.body_model_parts[2]
        calculated_rail_pad_geometry = rail_pad_model_part.geometry
        calculated_rail_pad_parameters = rail_pad_model_part.material.material_parameters

        expected_rail_pad_points = {
            4: Point.create([2.0, 3.02, 0], 4),
            1: Point.create([2.0, 3.0, 0], 1),
            5: Point.create([2.6, 3.02, 0], 5),
            2: Point.create([2.6, 3.0, 0], 2),
            6: Point.create([3.2, 3.02, 0], 6),
            3: Point.create([3.2, 3.0, 0], 3)
        }

        expected_rail_pad_lines = {5: Line.create([4, 1], 5), 6: Line.create([5, 2], 6), 7: Line.create([6, 3], 7)}

        expected_rail_pad_geometry = Geometry(expected_rail_pad_points, expected_rail_pad_lines)

        TestUtils.assert_almost_equal_geometries(expected_rail_pad_geometry, calculated_rail_pad_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_pad_parameters.__dict__, calculated_rail_pad_parameters.__dict__)

    def test_generate_straight_track_3d_volume_sleeper_error_no_sleeper_dims(
            self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if an error is raised when no sleeper dimensions are provided. A straight track is generated in 3d space
        without sleeper dimensions. An error should be raised.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """
        ndim = 3
        model = Model(ndim)

        rail_parameters = EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])

        origin_point = np.array([2.5, 1.0, 0.0])
        direction_vector = np.array([0, 0, 1])

        with pytest.raises(
                ValueError,
                match=r"If sleeper parameters are SoilMaterial, dimensions must be a list of length, width, height."):
            model.generate_straight_track(5.0, 2, rail_parameters, create_default_3d_soil_material, rail_pad_parameters,
                                          0.02, origin_point, direction_vector, "track_1", None)

    def test_generate_straight_track_3d_volume_sleeper_on_soil(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a straight track is generated correctly in a 3d space. A straight track is generated and added to the
        model. The geometry and material of the rails, sleepers and rail pads are checked. The sleepers are modelled as
        volumes. The sleepers are placed on the soil layer.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """
        # define the mock model
        ndim = 3
        model = Model(ndim)
        model.extrusion_length = 19.5
        # define the soil dimensions and material and assign it to the model
        material_soil = create_default_3d_soil_material
        # Specify the coordinates for the soil layer in the model
        layer1_coordinates = [(0.0, 0.0, -5.0), (4.0, 0.0, -5.0), (4.0, 1.0, -5.0), (0.0, 1.0, -5.0)]
        model.add_soil_layer_by_coordinates(layer1_coordinates, material_soil, "soil_layer_1")
        # define the rail parameters
        rail_parameters = EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        sleeper_parameters = create_default_3d_soil_material
        origin_point = np.array([2.5, 1.0, 0.0])
        direction_vector = np.array([0, 0, 1])
        # dimensions of the sleeper
        sleeper_height = 0.3
        rail_pad_thickness = 0.02
        sleeper_length = 2.6
        sleeper_width = 0.234
        sleeper_distance = 5.0
        sleeper_rail_pad_offset = sleeper_length / 2
        sleeper_dimensions = [sleeper_length, sleeper_width, sleeper_height]
        # create a straight track with rails, sleepers and rail pads
        model.generate_straight_track(sleeper_distance, 2, rail_parameters, sleeper_parameters, rail_pad_parameters,
                                      rail_pad_thickness, origin_point, direction_vector, "track_1",
                                      sleeper_rail_pad_offset, sleeper_dimensions)

        # check geometry and material of the rail
        expected_rail_points = {
            41: Point.create([origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness, origin_point[2]],
                             41),  #
            42: Point.create([
                origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness,
                origin_point[2] + sleeper_distance
            ], 42),
        }
        expected_rail_lines = {64: Line.create([41, 42], 64)}

        expected_rail_geometry = Geometry(expected_rail_points, expected_rail_lines)

        # check rail model part
        rail_model_part = model.body_model_parts[1]
        calculated_rail_geometry = rail_model_part.geometry
        calculated_rail_parameters = rail_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_geometry, calculated_rail_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_parameters.__dict__, calculated_rail_parameters.__dict__)

        # Check sleepers
        expected_sleeper_points = {
            10: Point.create(
                [origin_point[0] + sleeper_length / 2, origin_point[1], origin_point[2] + sleeper_width / 2], 10),
            17: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2
            ], 17),
            18: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2
            ], 18),
            9: Point.create(
                [origin_point[0] + sleeper_length / 2, origin_point[1], origin_point[2] - sleeper_width / 2], 9),
            19: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2
            ], 19),
            16: Point.create(
                [origin_point[0] - sleeper_length / 2, origin_point[1], origin_point[2] - sleeper_width / 2], 16),
            8: Point.create([origin_point[0], origin_point[1], origin_point[2] - sleeper_width / 2], 8),
            20: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2
            ], 20),
            15: Point.create(
                [origin_point[0] - sleeper_length / 2, origin_point[1], origin_point[2] + sleeper_width / 2], 15),
            11: Point.create([origin_point[0], origin_point[1], origin_point[2] + sleeper_width / 2], 11),
            30: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1],
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 30),
            37: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 37),
            38: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 38),
            29: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1],
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 29),
            39: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 39),
            36: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1],
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 36),
            28: Point.create([origin_point[0], origin_point[1], origin_point[2] - sleeper_width / 2 + sleeper_distance],
                             28),
            40: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 40),
            35: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1],
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 35),
            31: Point.create([origin_point[0], origin_point[1], origin_point[2] + sleeper_width / 2 + sleeper_distance],
                             31),
        }

        expected_sleeper_lines = {
            24: Line.create([10, 17], 24),
            26: Line.create([17, 18], 26),
            25: Line.create([9, 18], 25),
            11: Line.create([10, 9], 11),
            28: Line.create([18, 19], 28),
            27: Line.create([16, 19], 27),
            21: Line.create([8, 16], 21),
            10: Line.create([9, 8], 10),
            30: Line.create([19, 20], 30),
            29: Line.create([15, 20], 29),
            20: Line.create([16, 15], 20),
            31: Line.create([20, 17], 31),
            12: Line.create([11, 10], 12),
            19: Line.create([15, 11], 19),
            15: Line.create([8, 11], 15),
            56: Line.create([30, 37], 56),
            58: Line.create([37, 38], 58),
            57: Line.create([29, 38], 57),
            43: Line.create([30, 29], 43),
            60: Line.create([38, 39], 60),
            59: Line.create([36, 39], 59),
            53: Line.create([28, 36], 53),
            42: Line.create([29, 28], 42),
            62: Line.create([39, 40], 62),
            61: Line.create([35, 40], 61),
            52: Line.create([36, 35], 52),
            63: Line.create([40, 37], 63),
            44: Line.create([31, 30], 44),
            51: Line.create([35, 31], 51),
            47: Line.create([28, 31], 47),
        }
        expected_surfaces_sleeper = {
            10: Surface.create([24, 26, -25, -11], 10),
            11: Surface.create([25, 28, -27, -21, -10], 11),
            12: Surface.create([27, 30, -29, -20], 12),
            13: Surface.create([29, 31, -24, -12, -19], 13),
            4: Surface.create([-15, -10, -11, -12], 4),
            6: Surface.create([-21, 15, -19, -20], 6),
            14: Surface.create([26, 28, 30, 31], 14),
            24: Surface.create([56, 58, -57, -43], 24),
            25: Surface.create([57, 60, -59, -53, -42], 25),
            26: Surface.create([59, 62, -61, -52], 26),
            27: Surface.create([61, 63, -56, -44, -51], 27),
            18: Surface.create([-44, -47, -42, -43], 18),
            20: Surface.create([47, -51, -52, -53], 20),
            28: Surface.create([58, 60, 62, 63], 28),
        }
        expected_volume_sleeper = {
            2: Volume.create([-10, -11, -12, -13, 4, 6, 14], 2),
            3: Volume.create([-24, -25, -26, -27, 18, 20, 28], 3),
        }
        expected_sleeper_geometry = Geometry(expected_sleeper_points, expected_sleeper_lines, expected_surfaces_sleeper,
                                             expected_volume_sleeper)

        sleeper_model_part = model.body_model_parts[2]
        calculated_sleeper_geometry = sleeper_model_part.geometry

        TestUtils.assert_almost_equal_geometries(expected_sleeper_geometry, calculated_sleeper_geometry)

        # check the rail pads
        expected_rail_pad_points = {
            41: Point.create([origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness, origin_point[2]],
                             41),
            43: Point.create([origin_point[0], origin_point[1] + sleeper_height, origin_point[2]], 43),
            42: Point.create([
                origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness,
                origin_point[2] + sleeper_distance
            ], 42),
            44: Point.create([origin_point[0], origin_point[1] + sleeper_height, origin_point[2] + sleeper_distance],
                             44),
        }
        expected_rail_pad_lines = {65: Line.create([41, 43], 65), 66: Line.create([42, 44], 66)}

        expected_rail_pad_geometry = Geometry(expected_rail_pad_points, expected_rail_pad_lines)

        rail_pad_model_part = model.body_model_parts[3]
        calculated_rail_pad_geometry = rail_pad_model_part.geometry
        calculated_rail_pad_parameters = rail_pad_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_pad_geometry, calculated_rail_pad_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_pad_parameters.__dict__, calculated_rail_pad_parameters.__dict__)

    def test_generate_straight_track_3d_volume_sleeper_on_soil_x_direction(
            self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a straight track is generated correctly along the x-axis in a 3D space. The sleepers are modeled as
        volumes and placed on the soil layer.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """
        ndim = 3
        model = Model(ndim)
        model.extrusion_length = 19.5
        material_soil = create_default_3d_soil_material

        layer1_coordinates = [(0.0, 0.0, -5.0), (4.0, 0.0, -5.0), (4.0, 1.0, -5.0), (0.0, 1.0, -5.0)]
        model.add_soil_layer_by_coordinates(layer1_coordinates, material_soil, "soil_layer_1")

        rail_parameters = EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        sleeper_parameters = create_default_3d_soil_material
        origin_point = np.array([0.0, 1.0, 0.5])
        direction_vector = np.array([1, 0, 0])

        sleeper_height = 0.3
        rail_pad_thickness = 0.02
        sleeper_length = 2.6
        sleeper_width = 0.234
        sleeper_distance = 5.0
        sleeper_rail_pad_offset = sleeper_length / 2
        sleeper_dimensions = [sleeper_length, sleeper_width, sleeper_height]

        model.generate_straight_track(sleeper_distance, 2, rail_parameters, sleeper_parameters, rail_pad_parameters,
                                      rail_pad_thickness, origin_point, direction_vector, "track_x",
                                      sleeper_rail_pad_offset, sleeper_dimensions)
        rail_model_part = model.body_model_parts[1]
        sleeper_model_part = model.body_model_parts[2]
        rail_pad_model_part = model.body_model_parts[3]

        assert len(rail_model_part.geometry.points) == 2
        assert len(sleeper_model_part.geometry.volumes) == 2
        assert len(rail_pad_model_part.geometry.lines) == 2

        first_rail_point = rail_model_part.geometry.points[min(rail_model_part.geometry.points)]
        second_rail_point = rail_model_part.geometry.points[max(rail_model_part.geometry.points)]
        assert first_rail_point.coordinates[0] == origin_point[0]
        assert second_rail_point.coordinates[0] == origin_point[0] + sleeper_distance
        assert first_rail_point.coordinates[1] == origin_point[1] + sleeper_height + rail_pad_thickness

    def test_generate_straight_track_3d_volume_sleeper(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a straight track is generated correctly in a 3d space. A straight track is generated and added to the
        model. The geometry and material of the rails, sleepers and rail pads are checked. The sleepers are modelled as
        volumes.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3
        model = Model(ndim)

        rail_parameters = EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])

        sleeper_parameters = create_default_3d_soil_material

        origin_point = np.array([2.5, 1.0, 0.0])
        direction_vector = np.array([0, 0, 1])

        # dimensions of the sleeper
        sleeper_height = 0.3
        rail_pad_thickness = 0.02
        sleeper_length = 2.6
        sleeper_width = 0.234
        sleeper_distance = 5.0
        sleeper_rail_pad_offset = sleeper_length / 2
        sleeper_dimensions = [sleeper_length, sleeper_width, sleeper_height]
        # create a straight track with rails, sleepers and rail pads
        model.generate_straight_track(sleeper_distance, 2, rail_parameters, sleeper_parameters, rail_pad_parameters,
                                      rail_pad_thickness, origin_point, direction_vector, "track_1",
                                      sleeper_rail_pad_offset, sleeper_dimensions)

        # check geometry and material of the rail
        expected_rail_points = {
            17: Point.create([origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness, origin_point[2]],
                             17),  #
            18: Point.create([
                origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness,
                origin_point[2] + sleeper_distance
            ], 18),
        }
        expected_rail_lines = {25: Line.create([17, 18], 25)}

        expected_rail_geometry = Geometry(expected_rail_points, expected_rail_lines)

        # check rail model part
        rail_model_part = model.body_model_parts[0]
        calculated_rail_geometry = rail_model_part.geometry
        calculated_rail_parameters = rail_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_geometry, calculated_rail_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_parameters.__dict__, calculated_rail_parameters.__dict__)

        # check first sleeper

        expected_sleeper_points = {
            1: Point.create(
                [origin_point[0] + sleeper_length / 2, origin_point[1], origin_point[2] + sleeper_width / 2], 1),
            5: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2
            ], 5),
            6: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2
            ], 6),
            2: Point.create(
                [origin_point[0] + sleeper_length / 2, origin_point[1], origin_point[2] - sleeper_width / 2], 2),
            7: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2
            ], 7),
            3: Point.create(
                [origin_point[0] - sleeper_length / 2, origin_point[1], origin_point[2] - sleeper_width / 2], 3),
            8: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2
            ], 8),
            4: Point.create(
                [origin_point[0] - sleeper_length / 2, origin_point[1], origin_point[2] + sleeper_width / 2], 4),
            9: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1],
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 9),
            13: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 13),
            14: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 14),
            10: Point.create([
                origin_point[0] + sleeper_length / 2, origin_point[1],
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 10),
            15: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 15),
            11: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1],
                origin_point[2] - sleeper_width / 2 + sleeper_distance
            ], 11),
            16: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1] + sleeper_height,
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 16),
            12: Point.create([
                origin_point[0] - sleeper_length / 2, origin_point[1],
                origin_point[2] + sleeper_width / 2 + sleeper_distance
            ], 12),
        }
        expected_sleeper_lines = {
            5: Line.create([1, 5], 5),
            7: Line.create([5, 6], 7),
            6: Line.create([2, 6], 6),
            1: Line.create([1, 2], 1),
            9: Line.create([6, 7], 9),
            8: Line.create([3, 7], 8),
            2: Line.create([2, 3], 2),
            11: Line.create([7, 8], 11),
            10: Line.create([4, 8], 10),
            3: Line.create([3, 4], 3),
            12: Line.create([8, 5], 12),
            4: Line.create([4, 1], 4),
            17: Line.create([9, 13], 17),
            19: Line.create([13, 14], 19),
            18: Line.create([10, 14], 18),
            13: Line.create([9, 10], 13),
            21: Line.create([14, 15], 21),
            20: Line.create([11, 15], 20),
            14: Line.create([10, 11], 14),
            23: Line.create([15, 16], 23),
            22: Line.create([12, 16], 22),
            15: Line.create([11, 12], 15),
            24: Line.create([16, 13], 24),
            16: Line.create([12, 9], 16),
        }
        expected_surfaces_sleeper = {
            2: Surface.create([5, 7, -6, -1], 2),
            3: Surface.create([6, 9, -8, -2], 3),
            4: Surface.create([8, 11, -10, -3], 4),
            5: Surface.create([10, 12, -5, -4], 5),
            1: Surface.create([1, 2, 3, 4], 1),
            6: Surface.create([7, 9, 11, 12], 6),
            8: Surface.create([17, 19, -18, -13], 8),
            9: Surface.create([18, 21, -20, -14], 9),
            10: Surface.create([20, 23, -22, -15], 10),
            11: Surface.create([22, 24, -17, -16], 11),
            7: Surface.create([13, 14, 15, 16], 7),
            12: Surface.create([19, 21, 23, 24], 12),
        }
        expected_volume_sleeper = {
            1: Volume.create([-2, -3, -4, -5, -1, 6], 1),
            2: Volume.create([-8, -9, -10, -11, -7, 12], 2),
        }
        expected_sleeper_geometry = Geometry(expected_sleeper_points, expected_sleeper_lines, expected_surfaces_sleeper,
                                             expected_volume_sleeper)

        sleeper_model_part = model.body_model_parts[1]
        calculated_sleeper_geometry = sleeper_model_part.geometry

        TestUtils.assert_almost_equal_geometries(expected_sleeper_geometry, calculated_sleeper_geometry)

        # check the rail pads
        expected_rail_pad_points = {
            17: Point.create([origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness, origin_point[2]],
                             17),
            19: Point.create([origin_point[0], origin_point[1] + sleeper_height, origin_point[2]], 19),
            18: Point.create([
                origin_point[0], origin_point[1] + sleeper_height + rail_pad_thickness,
                origin_point[2] + sleeper_distance
            ], 18),
            20: Point.create([origin_point[0], origin_point[1] + sleeper_height, origin_point[2] + sleeper_distance],
                             20),
        }
        expected_rail_pad_lines = {26: Line.create([17, 19], 26), 27: Line.create([18, 20], 27)}

        expected_rail_pad_geometry = Geometry(expected_rail_pad_points, expected_rail_pad_lines)

        rail_pad_model_part = model.body_model_parts[2]
        calculated_rail_pad_geometry = rail_pad_model_part.geometry
        calculated_rail_pad_parameters = rail_pad_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_pad_geometry, calculated_rail_pad_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_pad_parameters.__dict__, calculated_rail_pad_parameters.__dict__)

    def test_generate_straight_track_3d(self):
        """
        Tests if a straight track is generated correctly in a 3d space. A straight track is generated and added to the
        model. The geometry and material of the rails, sleepers and rail pads are checked.
        """

        model = Model(3)

        rail_parameters = EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        sleeper_parameters = NodalConcentrated([1, 1, 1], 1, [1, 1, 1])

        origin_point = np.array([2.0, 3.0, 1.0])
        direction_vector = np.array([1, 1, -1])

        # create a straight track with rails, sleepers and rail pads
        model.generate_straight_track(0.6, 3, rail_parameters, sleeper_parameters, rail_pad_parameters, 0.02,
                                      origin_point, direction_vector, "track_1")

        distance_sleepers_xyz = 0.6 / 3**0.5

        # check geometry and material of the rail
        expected_rail_points = {
            4: Point.create([2.0, 3.02, 1.0], 4),
            5: Point.create([2.0 + distance_sleepers_xyz, 3.02 + distance_sleepers_xyz, 1.0 - distance_sleepers_xyz],
                            5),
            6: Point.create(
                [2.0 + 2 * distance_sleepers_xyz, 3.02 + 2 * distance_sleepers_xyz, 1.0 - 2 * distance_sleepers_xyz], 6)
        }
        expected_rail_lines = {3: Line.create([4, 5], 3), 4: Line.create([5, 6], 4)}

        expected_rail_geometry = Geometry(expected_rail_points, expected_rail_lines)

        # check rail model part
        rail_model_part = model.body_model_parts[0]
        calculated_rail_geometry = rail_model_part.geometry
        calculated_rail_parameters = rail_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_geometry, calculated_rail_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_parameters.__dict__, calculated_rail_parameters.__dict__)

        # check geometry and material of the sleepers
        expected_sleeper_points = {
            1: Point.create([2.0, 3.0, 1.0], 1),
            2: Point.create([2.0 + distance_sleepers_xyz, 3.0 + distance_sleepers_xyz, 1.0 - distance_sleepers_xyz], 2),
            3: Point.create(
                [2.0 + 2 * distance_sleepers_xyz, 3.0 + 2 * distance_sleepers_xyz, 1.0 - 2 * distance_sleepers_xyz], 3)
        }

        expected_sleeper_geometry = Geometry(expected_sleeper_points)

        sleeper_model_part = model.body_model_parts[1]
        calculated_sleeper_geometry = sleeper_model_part.geometry
        calculated_sleeper_parameters = sleeper_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_sleeper_geometry, calculated_sleeper_geometry)
        TestUtils.assert_dictionary_almost_equal(sleeper_parameters.__dict__, calculated_sleeper_parameters.__dict__)

        # check geometry and material of the rail pads
        rail_pad_model_part = model.body_model_parts[2]
        calculated_rail_pad_geometry = rail_pad_model_part.geometry
        calculated_rail_pad_parameters = rail_pad_model_part.material.material_parameters

        expected_rail_pad_points = {
            4: Point.create([2.0, 3.02, 1.0], 4),
            1: Point.create([2.0, 3.0, 1.0], 1),
            5: Point.create([2.0 + distance_sleepers_xyz, 3.02 + distance_sleepers_xyz, 1.0 - distance_sleepers_xyz],
                            5),
            2: Point.create([2.0 + distance_sleepers_xyz, 3.0 + distance_sleepers_xyz, 1.0 - distance_sleepers_xyz], 2),
            6: Point.create(
                [2.0 + 2 * distance_sleepers_xyz, 3.02 + 2 * distance_sleepers_xyz, 1.0 - 2 * distance_sleepers_xyz],
                6),
            3: Point.create(
                [2.0 + 2 * distance_sleepers_xyz, 3.0 + 2 * distance_sleepers_xyz, 1.0 - 2 * distance_sleepers_xyz], 3)
        }

        expected_rail_pad_lines = {5: Line.create([4, 1], 5), 6: Line.create([5, 2], 6), 7: Line.create([6, 3], 7)}

        expected_rail_pad_geometry = Geometry(expected_rail_pad_points, expected_rail_pad_lines)

        TestUtils.assert_almost_equal_geometries(expected_rail_pad_geometry, calculated_rail_pad_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_pad_parameters.__dict__, calculated_rail_pad_parameters.__dict__)

        # check rotation constrain model part
        rotation_constrain_model_part = model.process_model_parts[1]
        calculated_rotation_constrain_geometry = rotation_constrain_model_part.geometry
        calculated_rotation_constrain_parameters = rotation_constrain_model_part.parameters

        expected_rotation_constrain_points = {4: Point.create([2.0, 3.02, 1.0], 4)}
        expected_rotation_constrain_geometry = Geometry(expected_rotation_constrain_points)

        expected_rotation_constraint_parameters = RotationConstraint(value=[0, 0, 0],
                                                                     is_fixed=[True, True, True],
                                                                     active=[True, True, True])

        TestUtils.assert_almost_equal_geometries(expected_rotation_constrain_geometry,
                                                 calculated_rotation_constrain_geometry)
        TestUtils.assert_dictionary_almost_equal(expected_rotation_constraint_parameters.__dict__,
                                                 calculated_rotation_constrain_parameters.__dict__)

    def test_set_element_size_of_group(self, create_default_2d_soil_material: SoilMaterial):
        """
        Tests setting the element size of a group. A model is created with a soil layer. The element size of the
        process model part is set to half the size of the rest of the mesh. The mesh is visually checked in Gmsh if it
        is as expected. In this test the only performed checks are the amount of elements and nodes per model part.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """
        model = Model(2)

        # add soil material
        soil_material = create_default_2d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")

        # add process geometry
        gmsh_process_input = {"process_1d": {"coordinates": [[0, 1, 0], [1, 1, 0]], "ndim": 1}}
        model.gmsh_io.generate_geometry(gmsh_process_input, "")

        # create process model part
        process_model_part = ModelPart("process_1d")

        # set the geometry of the process model part
        process_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "process_1d")

        # add process geometry
        gmsh_process_input = {"process_1d_2": {"coordinates": [[0, 1, 0], [1, 1, 0]], "ndim": 1}}
        model.gmsh_io.generate_geometry(gmsh_process_input, "")

        # create process model part
        process_model_part2 = ModelPart("process_1d_2")

        # set the geometry of the process model part
        process_model_part2.get_geometry_from_geo_data(model.gmsh_io.geo_data, "process_1d_2")

        # add process model parts
        model.process_model_parts.append(process_model_part)
        model.process_model_parts.append(process_model_part2)

        # synchronise geometry and generate mesh
        model.synchronise_geometry()

        # set base mesh size
        model.set_mesh_size(1)

        # set element size of process model parts
        model.set_element_size_of_group(0.5, "process_1d")
        model.set_element_size_of_group(1, "process_1d_2")
        model.generate_mesh(open_gmsh_gui=False)

        # check mesh of body model part
        mesh_body = model.body_model_parts[0].mesh
        assert len(mesh_body.elements) == 13
        assert len(mesh_body.nodes) == 11

        # check process model parts, both process model parts should have the same amount of elements and nodes
        mesh_process = model.process_model_parts[0].mesh
        assert len(mesh_process.elements) == 2
        assert len(mesh_process.nodes) == 3

        mesh_process = model.process_model_parts[1].mesh
        assert len(mesh_process.elements) == 2
        assert len(mesh_process.nodes) == 3

    def test_set_element_size_of_group_expected_raise(self):
        """
        Tests if a ValueError is raised when the group name is not found while setting the element size of a group

        """

        model = Model(2)

        with pytest.raises(ValueError, match=f"Group name `non_existing_group` not found."):
            model.set_element_size_of_group(1, "non_existing_group")

    def test_split_body_model_part_3D(self, expected_geometry_two_layers_3D_extruded: Tuple[Geometry, Geometry],
                                      create_default_3d_soil_material: SoilMaterial):
        """
        Test if a body model part is split correctly in a 3D case. A model is created with two soil layers. The model is
        split in two parts and the geometry of the split model parts is checked.

        Args:
            - expected_geometry_two_layers_3D_extruded (Tuple[:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`]): expected geometry of the model
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 3

        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_3d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material1, "layer2")

        model.gmsh_io.geo_data["physical_groups"]["layer1"]["geometry_ids"] = [1, 2]
        model.gmsh_io.geo_data["physical_groups"].pop("layer2")

        model.body_model_parts[0].geometry = Geometry.create_geometry_from_gmsh_group(model.gmsh_io.geo_data, "layer1")
        model.body_model_parts.pop(1)

        model.split_model_part("layer1", "split_layer1", [1], soil_material1)

        # check geometry of the split model part
        TestUtils.assert_almost_equal_geometries(expected_geometry_two_layers_3D_extruded[0],
                                                 model.body_model_parts[1].geometry)
        TestUtils.assert_almost_equal_geometries(expected_geometry_two_layers_3D_extruded[1],
                                                 model.body_model_parts[0].geometry)

        # split body model part expected error
        with pytest.raises(ValueError,
                           match="New parameters must have the same material type as in the original "
                           "body model part."):
            model.split_model_part("layer1", "second_split", [2],
                                   StructuralMaterial("beam", EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)))

    def test_split_process_model_part_2D(self, expected_geometry_two_layers_2D: Tuple[Geometry, Geometry, Geometry],
                                         create_default_2d_soil_material: SoilMaterial):
        """
        Test if a process model part is split correctly in a 2D case. A model is created with two soil layers. The model
        is split in two parts and the geometry of the split model parts is checked.

        Args:
            - expected_geometry_two_layers_2D (Tuple[:class:`stem.geometry.Geometry`, \
                :class:`stem.geometry.Geometry`, :class:`stem.geometry.Geometry`]): expected geometry of the model
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        ndim = 2

        layer1_coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        layer2_coordinates = [(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)]

        # define soil materials
        soil_material1 = create_default_2d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")
        model.add_soil_layer_by_coordinates(layer2_coordinates, soil_material1, "layer2")

        # add process model part on the whole geometry
        model._Model__add_gravity_load()

        # split process model part
        model.split_model_part("gravity_load_2d", "gravity_layer_2", [2],
                               GravityLoad(value=[20, 20, 20], active=[True, True, True]))

        # check geometry and parameters of the original and split model part
        assert model.process_model_parts[0].name == "gravity_load_2d"
        assert model.process_model_parts[0].parameters.value == [0, -9.81, 0]
        TestUtils.assert_almost_equal_geometries(expected_geometry_two_layers_2D[0],
                                                 model.process_model_parts[0].geometry)

        assert model.process_model_parts[1].name == "gravity_layer_2"
        assert model.process_model_parts[1].parameters.value == [20, 20, 20]
        TestUtils.assert_almost_equal_geometries(expected_geometry_two_layers_2D[1],
                                                 model.process_model_parts[1].geometry)

        # split process model part expected error
        with pytest.raises(ValueError,
                           match="New parameters must have the same process parameter type as in the "
                           "original process model part."):
            model.split_model_part("gravity_load_2d", "gravity_layer_3", [1],
                                   LineLoad(value=[20, 20, 20], active=[True, True, True]))

    def test_split_model_part_expected_raise(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if a ValueErrors are raised when:
        - the group name is not found while splitting a model part |
        - the geometry is not defined in the model part |
        - the model part type and new parameters type do not match

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        model = Model(2)

        model.process_model_parts.append(ModelPart("process_2d"))

        with pytest.raises(ValueError, match=f"Model part: non_existing_group not found."):
            model.split_model_part("non_existing_group", "split_group", [1], create_default_2d_soil_material)

        with pytest.raises(ValueError, match=f"Geometry is not defined in the model part: process_2d."):
            model.split_model_part("process_2d", "split_group", [1], create_default_2d_soil_material)

        model.process_model_parts[0].geometry = Geometry(surfaces={1: Surface.create([1, 2, 3], 1)},
                                                         lines={
                                                             1: Line.create([1, 2], 1),
                                                             2: Line.create([2, 3], 2),
                                                             3: Line.create([3, 1], 3)
                                                         },
                                                         points={
                                                             1: Point.create([0, 0, 0], 1),
                                                             2: Point.create([1, 0, 0], 2),
                                                             3: Point.create([1, 1, 0], 3)
                                                         })

        with pytest.raises(ValueError, match=f"Model part type and new parameters type must match."):
            model.split_model_part("process_2d", "split_group", [1], create_default_2d_soil_material)

    def test_finalise_json_output_raises_errors(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test that finalisation raises error correctly.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): A default soil material.

        """

        # define layer coordinates
        ndim = 2
        layer1_coordinates = [(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)]

        # define soil materials
        soil_material1 = create_default_2d_soil_material
        soil_material1.name = "soil1"

        # create model
        model = Model(ndim)

        # add soil layers
        model.add_soil_layer_by_coordinates(layer1_coordinates, soil_material1, "layer1")

        # synchronise geometry and recalculates the ids
        model.synchronise_geometry()
        # Define nodal results
        nodal_results = [NodalOutput.ACCELERATION]
        # Define output coordinates
        output_coordinates = [(1.5, 1, 0), (1.5, 0.5, 0), (2.5, 0.5, 0), (2.5, 0, 0)]

        # add output settings
        model.add_output_settings_by_coordinates(output_coordinates,
                                                 part_name="nodal_accelerations",
                                                 output_name="json_nodal_accelerations",
                                                 output_dir="dir_test",
                                                 output_parameters=JsonOutputParameters(output_interval=100,
                                                                                        nodal_results=nodal_results))
        model.synchronise_geometry()
        model.generate_mesh()

        # set output name of json output to None
        model_copy = deepcopy(model)
        model_copy.output_settings[-1].output_name = None
        msg = "No name is specified for the json file."
        with pytest.raises(ValueError, match=msg):
            model_copy.finalise(input_folder="input_files")

        # set json filename to a wrong one
        model_copy = deepcopy(model)
        model_copy.output_settings[-1].output_name = "json_nodal_displacements"

        expected_path = Path(
            'input_dir') / model_copy.output_settings[-1].output_dir / model_copy.output_settings[-1].output_name
        expected_path = str(expected_path.with_suffix('.json'))
        msg = (f"No JSON file is found in the output directory for path: {expected_path}. "
               "Either the working folder is incorrectly specified or no simulation has been performed yet.")
        with pytest.raises(OSError, match=re.escape(msg)):
            model_copy.finalise(input_folder="input_dir")

        # set part name of the output settings to None
        model_copy = deepcopy(model)
        model_copy.output_settings[-1].part_name = None
        msg = "The output model part has no part name specified."
        with pytest.raises(ValueError, match=msg):
            model_copy.finalise(input_folder="input_files")

        # set part name of the output settings to non-existing part
        model_copy = deepcopy(model)
        model_copy.output_settings[-1].part_name = "part 404"
        msg = "No model part matches the part name specified in the output settings."
        with pytest.raises(ValueError, match=msg):
            model_copy.finalise(input_folder="input_files")

        # set part name of the output settings to non-existing part
        model_copy = deepcopy(model)
        model_copy.process_model_parts[-1].mesh = None
        msg = "process model part has not been meshed yet!"
        with pytest.raises(ValueError, match=msg):
            model_copy.finalise(input_folder="input_files")

    def test_add_boundary_condition_on_plane(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a boundary condition is added correctly on a plane. A model is created with two soil layers. A boundary
        condition is added on a plane and the nodes of the boundary condition are checked.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """

        model = Model(3)
        model.extrusion_length = 1

        # add soil material
        soil_material = create_default_3d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.add_soil_layer_by_coordinates([(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)], soil_material, "layer2")

        model.synchronise_geometry()

        no_displacement_boundary = DisplacementConstraint(active=[True, True, True],
                                                          is_fixed=[True, True, True],
                                                          value=[0, 0, 0])

        plane_coordinates = [(0, 0, 0), (0, 0, 1), (0, 1, 1)]
        model.add_boundary_condition_on_plane(plane_coordinates, no_displacement_boundary, "left_side_boundary")

        # check if the boundary condition is added to the model
        assert len(model.process_model_parts) == 1

        # check if the boundary condition is added to the correct model part
        assert model.process_model_parts[0].name == "left_side_boundary"
        assert model.process_model_parts[0].parameters == no_displacement_boundary

        # check if the boundary condition is added to the correct nodes
        expected_points = {
            7: Point.create([0, 1, 0], 7),
            8: Point.create([0, 1, 1], 8),
            2: Point.create([0, 0, 1], 2),
            1: Point.create([0, 0, 0], 1),
            11: Point.create([0, 2, 0], 11),
            12: Point.create([0, 2, 1], 12),
        }

        # get the generated points
        generated_points = model.process_model_parts[0].geometry.points

        # check if points are the same, these should be the all the points on the plane, i.e. points which are part
        # of layer 1 and layer 2
        for (generated_point_id, generated_point), (expected_point_id, expected_point) in \
                zip(generated_points.items(), expected_points.items()):
            assert generated_point_id == expected_point_id
            assert generated_point.id == expected_point.id
            npt.assert_allclose(generated_point.coordinates, expected_point.coordinates)

        # add the same boundary condition again, this should not raise an error
        model.add_boundary_condition_on_plane(plane_coordinates, no_displacement_boundary, "left_side_boundary")

        # check if the boundary condition is added to the model
        assert len(model.process_model_parts) == 1

        # check if the boundary condition is added to the correct model part
        assert model.process_model_parts[0].name == "left_side_boundary"
        assert model.process_model_parts[0].parameters == no_displacement_boundary

        # get the generated points
        generated_points = model.process_model_parts[0].geometry.points

        # check if points are the same, these should be the all the points on the plane, i.e. points which are part
        # of layer 1 and layer 2
        for (generated_point_id, generated_point), (expected_point_id, expected_point) in \
                zip(generated_points.items(), expected_points.items()):
            assert generated_point_id == expected_point_id
            assert generated_point.id == expected_point.id
            npt.assert_allclose(generated_point.coordinates, expected_point.coordinates)

        # expect it raises an error when the less than 3 points are given to define a plane
        with pytest.raises(ValueError, match="At least 3 vertices are required to define a plane."):
            model.add_boundary_condition_on_plane([(0, 0, 0), (0, 0, 1)], no_displacement_boundary, "wrong_plane")

    def test_add_boundary_condition_on_polygon(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if a boundary condition is added correctly on a polygon. A model is created with two soil layers. A boundary
        condition is added on a polygon and the nodes of the boundary condition are checked.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """

        model = Model(3)
        model.extrusion_length = 1

        # add soil material
        soil_material = create_default_3d_soil_material

        # add soil layers
        model.add_soil_layer_by_coordinates([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], soil_material, "layer1")
        model.add_soil_layer_by_coordinates([(1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0)], soil_material, "layer2")

        model.synchronise_geometry()

        no_displacement_boundary = DisplacementConstraint(active=[True, True, True],
                                                          is_fixed=[True, True, True],
                                                          value=[0, 0, 0])

        polygon_coordinates = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]
        model.add_boundary_condition_on_polygon(polygon_coordinates, no_displacement_boundary, "left_bottom_boundary")

        # check if the boundary condition is added to the model
        assert len(model.process_model_parts) == 1

        # check if the boundary condition is added to the correct model part
        assert model.process_model_parts[0].name == "left_bottom_boundary"
        assert model.process_model_parts[0].parameters == no_displacement_boundary

        # check if the boundary condition is added to the correct nodes
        expected_points = {
            7: Point.create([0, 1, 0], 7),
            8: Point.create([0, 1, 1], 8),
            2: Point.create([0, 0, 1], 2),
            1: Point.create([0, 0, 0], 1),
        }

        # Get the generated points
        generated_points = model.process_model_parts[0].geometry.points

        # check if points are the same, these should be only the points which are part of layer 1
        for (generated_point_id, generated_point), (expected_point_id, expected_point) in \
                zip(generated_points.items(), expected_points.items()):
            assert generated_point_id == expected_point_id
            assert generated_point.id == expected_point.id
            npt.assert_allclose(generated_point.coordinates, expected_point.coordinates)

    def test_get_bounding_box_soil(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if the bounding box of all soil volumes is correctly calculated. A model is created with a soil layer and
        the bounding box of the soil volume is checked.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """
        ndim = 3

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)]

        # define soil material
        soil_material = create_default_3d_soil_material

        # create model
        model = Model(ndim)
        model.extrusion_length = 4

        model.project_parameters = TestUtils.create_default_solver_settings()

        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material

        # run the tests
        min_coords, max_coords = model.get_bounding_box_soil()

        assert min_coords[0] == 0
        assert min_coords[1] == 0
        assert min_coords[2] == 0
        assert max_coords[0] == 1
        assert max_coords[1] == 2
        assert max_coords[2] == 4

    def test_get_bounding_box_soil_error(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if an error is raised when the model part has no geometry. A model is created with a soil layer and the
        bounding box of the soil volume is checked. The geometry of the model part is set to None and an error is
        expected.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """
        ndim = 3

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 2, 3), (0, 2, 3)]

        # define soil material
        soil_material = create_default_3d_soil_material

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        model.project_parameters = TestUtils.create_default_solver_settings()

        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material
        model.body_model_parts[0].geometry = None

        with pytest.raises(ValueError, match="Model part has no geometry."):
            model.get_bounding_box_soil()

    def test_get_points_outside_soil_volume(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if the points outside the soil volume are correctly found. A model is created with a soil layer and
        points outside the soil volume are checked. Some points are outside the soil volume and some are inside.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material

        """
        ndim = 3

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)]

        # define soil material
        soil_material = create_default_3d_soil_material

        # create model
        model = Model(ndim)
        model.extrusion_length = 4

        model.project_parameters = TestUtils.create_default_solver_settings()

        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material

        points_outside_test = [(1, 0, -1), (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 5), (1, 0, 6)]
        outside_name = f"points_outside"
        points_outside_settings = {outside_name: {"coordinates": points_outside_test, "ndim": 0}}
        model.gmsh_io.generate_geometry(points_outside_settings, "")
        points_outside_part = BodyModelPart(outside_name)
        points_outside_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, outside_name)
        rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 750e6, 1],
                                                  NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                                  NODAL_DAMPING_COEFFICIENT=[1, 750e3, 1],
                                                  NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])
        points_outside_part.material = StructuralMaterial(name=outside_name, material_parameters=rail_pad_parameters)
        model.body_model_parts.append(points_outside_part)
        # run the test
        points_outside_volume = model.get_points_outside_soil(outside_name)
        # check if the points are outside the soil volume
        assert len(points_outside_volume) == 3
        assert points_outside_volume[0].coordinates == [1., 0., -1.]
        assert points_outside_volume[1].coordinates == [1., 0., 5.]
        assert points_outside_volume[2].coordinates == [1., 0., 6.]

    def test_get_points_outside_soil_volume_error(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if an error is raised when the model part is not found. A model is created with a soil layer and points
        outside the soil volume are checked. The model part is not found and an error is expected.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """
        ndim = 3

        layer_coordinates = [(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)]

        # define soil material
        soil_material = create_default_3d_soil_material

        # create model
        model = Model(ndim)
        model.extrusion_length = 4

        model.project_parameters = TestUtils.create_default_solver_settings()

        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"
        assert model.body_model_parts[0].material == soil_material

        points_outside_test = [(1, 0, -1), (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 5), (1, 0, 6)]
        outside_name = f"fake_name"
        with pytest.raises(ValueError, match="Model part fake_name not found."):
            model.get_points_outside_soil(outside_name)

    def test_get_points_outside_soil_volume_error_geometry(self, create_default_3d_soil_material: SoilMaterial):
        """
        Test if an error is raised when the model part has no geometry. A model is created with a soil layer and points
        outside the soil volume are checked. The geometry of the model part is set to None and an error is expected.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """
        ndim = 3

        # create model
        model = Model(ndim)
        model.extrusion_length = 1

        model.project_parameters = TestUtils.create_default_solver_settings()

        beam = StructuralMaterial(name="soil1", material_parameters=EulerBeam(2, 1, 1, 1, 1, 1))
        beam_part = BodyModelPart("soil1")
        beam_part.material = beam
        model.body_model_parts.append(beam_part)

        # check if layer is added correctly
        assert len(model.body_model_parts) == 1
        assert model.body_model_parts[0].name == "soil1"

        outside_name = f"soil1"
        with pytest.raises(ValueError, match="Model part soil1 has no geometry."):
            model.get_points_outside_soil(outside_name)

    def test_generate_extended_straight_track_2d(self, create_default_2d_soil_material: SoilMaterial):
        """
        Test if a straight track is generated correctly in a 2d space. A straight track is generated and added to the
        model. The geometry and material of the rails, sleepers and rail pads are checked.

        Args:
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """

        # initialise model
        model = Model(2)
        # add soil material 2d
        soil_material = create_default_2d_soil_material
        layer_coordinates = [(2, 3, 0), (4, 3, 0), (4, 1, 0), (2, 1, 0)]
        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        rail_parameters = EulerBeam(2, 1, 1, 1, 1, 1)
        extended_soil_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        sleeper_parameters = NodalConcentrated([1, 1, 1], 1, [1, 1, 1])

        origin_point = np.array([1.0, 3.0, -1])
        direction_vector = np.array([1, 0, 0])

        # create a straight track with rails, sleepers and rail pads
        model.generate_extended_straight_track(0.6, 8, rail_parameters, sleeper_parameters, rail_pad_parameters, 0.02,
                                               origin_point, extended_soil_parameters, 5, direction_vector, "track_1")

        # check geometry and material of the rail
        expected_rail_points = {
            13: Point.create([1.0, 3.02, -1.0], 13),
            14: Point.create([1.6, 3.02, -1.0], 14),
            15: Point.create([2.2, 3.02, -1.0], 15),
            16: Point.create([2.8, 3.02, -1.0], 16),
            17: Point.create([3.4, 3.02, -1.0], 17),
            18: Point.create([4.0, 3.02, -1.0], 18),
            19: Point.create([4.6, 3.02, -1.0], 19),
            20: Point.create([5.2, 3.02, -1.0], 20)
        }
        expected_rail_lines = {
            12: Line.create([13, 14], 12),
            13: Line.create([14, 15], 13),
            14: Line.create([15, 16], 14),
            15: Line.create([16, 17], 15),
            16: Line.create([17, 18], 16),
            17: Line.create([18, 19], 17),
            18: Line.create([19, 20], 18),
        }

        expected_rail_geometry = Geometry(expected_rail_points, expected_rail_lines)

        # check rail model part
        rail_model_part = model.body_model_parts[1]
        calculated_rail_geometry = rail_model_part.geometry
        calculated_rail_parameters = rail_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_geometry, calculated_rail_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_parameters.__dict__, calculated_rail_parameters.__dict__)

        # check geometry and material of the sleepers
        expected_sleeper_points = {
            5: Point.create([1.0, 3.00, -1.0], 5),
            6: Point.create([1.6, 3.00, -1.0], 6),
            7: Point.create([2.2, 3.00, -1.0], 7),
            8: Point.create([2.8, 3.00, -1.0], 8),
            9: Point.create([3.4, 3.00, -1.0], 9),
            10: Point.create([4.0, 3.00, -1.0], 10),
            11: Point.create([4.6, 3.00, -1.0], 11),
            12: Point.create([5.2, 3.00, -1.0], 12)
        }
        expected_sleeper_geometry = Geometry(expected_sleeper_points)

        sleeper_model_part = model.body_model_parts[2]
        calculated_sleeper_geometry = sleeper_model_part.geometry
        calculated_sleeper_parameters = sleeper_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_sleeper_geometry, calculated_sleeper_geometry)
        TestUtils.assert_dictionary_almost_equal(sleeper_parameters.__dict__, calculated_sleeper_parameters.__dict__)

        # check geometry and material of the rail pads
        rail_pad_model_part = model.body_model_parts[3]
        calculated_rail_pad_geometry = rail_pad_model_part.geometry
        calculated_rail_pad_parameters = rail_pad_model_part.material.material_parameters

        expected_rail_pad_points = {
            13: Point.create([1.0, 3.02, -1.0], 13),
            5: Point.create([1.0, 3.0, -1.0], 5),
            14: Point.create([1.6, 3.02, -1.0], 14),
            6: Point.create([1.6, 3.0, -1.0], 6),
            15: Point.create([2.2, 3.02, -1.0], 15),
            7: Point.create([2.2, 3.0, -1.0], 7),
            16: Point.create([2.8, 3.02, -1.0], 16),
            8: Point.create([2.8, 3.0, -1.0], 8),
            17: Point.create([3.4, 3.02, -1.0], 17),
            9: Point.create([3.4, 3.0, -1.0], 9),
            18: Point.create([4.0, 3.02, -1.0], 18),
            10: Point.create([4.0, 3.0, -1.0], 10),
            19: Point.create([4.6, 3.02, -1.0], 19),
            11: Point.create([4.6, 3.0, -1.0], 11),
            20: Point.create([5.2, 3.02, -1.0], 20),
            12: Point.create([5.2, 3.0, -1.0], 12),
        }

        expected_rail_pad_lines = {
            19: Line.create([13, 5], 19),
            20: Line.create([14, 6], 20),
            21: Line.create([15, 7], 21),
            22: Line.create([16, 8], 22),
            23: Line.create([17, 9], 23),
            24: Line.create([18, 10], 24),
            25: Line.create([19, 11], 25),
            26: Line.create([20, 12], 26),
        }

        expected_rail_pad_geometry = Geometry(expected_rail_pad_points, expected_rail_pad_lines)

        TestUtils.assert_almost_equal_geometries(expected_rail_pad_geometry, calculated_rail_pad_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_pad_parameters.__dict__, calculated_rail_pad_parameters.__dict__)

        # check geometry and material of soil equivalent
        soil_equivalent_model_part = model.body_model_parts[4]
        calculated_soil_equivalent_geometry = soil_equivalent_model_part.geometry
        calculated_soil_equivalent_parameters = soil_equivalent_model_part.material.material_parameters

        expected_soil_equivalent_points = {
            5: Point.create([1.0, 3.0, -1.0], 5),
            21: Point.create([1.0, -2.0, -1.0], 21),
            6: Point.create([1.6, 3.0, -1.0], 6),
            22: Point.create([1.6, -2.0, -1.0], 22),
            11: Point.create([4.6, 3.0, -1.0], 11),
            23: Point.create([4.6, -2.0, -1.0], 23),
            12: Point.create([5.2, 3.0, -1.0], 12),
            24: Point.create([5.2, -2.0, -1.0], 24),
        }

        expected_soil_equivalent_lines = {
            27: Line.create([5, 21], 27),
            28: Line.create([6, 22], 28),
            29: Line.create([11, 23], 29),
            30: Line.create([12, 24], 30),
        }

        expected_soil_equivalent_geometry = Geometry(expected_soil_equivalent_points, expected_soil_equivalent_lines)

        TestUtils.assert_almost_equal_geometries(expected_soil_equivalent_geometry, calculated_soil_equivalent_geometry)
        TestUtils.assert_dictionary_almost_equal(extended_soil_parameters.__dict__,
                                                 calculated_soil_equivalent_parameters.__dict__)

    def test_generate_extended_straight_track_3d(self, create_default_3d_soil_material: SoilMaterial):
        """
        Tests if a straight track is generated correctly in a 3d space. A straight track is generated and added to the
        model. The geometry and material of the rails, sleepers and rail pads are checked.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """

        model = Model(3)
        model.extrusion_length = 1
        # add soil material 2d
        soil_material = create_default_3d_soil_material
        layer_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 3.0, 0.0), (0.0, 3.0, 0.0)]
        # add soil layer
        model.add_soil_layer_by_coordinates(layer_coordinates, soil_material, "soil1")

        rail_parameters = EulerBeam(3, 1, 1, 1, 1, 1, 1, 1)
        rail_pad_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        extended_soil_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        sleeper_parameters = NodalConcentrated([1, 1, 1], 1, [1, 1, 1])

        origin_point = np.array([2.0, 3.0, -0.5])
        direction_vector = np.array([0, 0, 1])

        # create a straight track with rails, sleepers and rail pads
        # create a straight track with rails, sleepers and rail pads
        model.generate_extended_straight_track(0.5, 5, rail_parameters, sleeper_parameters, rail_pad_parameters, 0.02,
                                               origin_point, extended_soil_parameters, 5, direction_vector, "track_1")

        # check geometry and material of the rail
        expected_rail_points = {
            14: Point.create([2.0, 3.02, -0.5], 14),
            15: Point.create([2.0, 3.02, 0.0], 15),
            16: Point.create([2.0, 3.02, 0.5], 16),
            17: Point.create([2.0, 3.02, 1.0], 17),
            18: Point.create([2.0, 3.02, 1.5], 18),
        }
        expected_rail_lines = {
            21: Line.create([14, 15], 21),
            22: Line.create([15, 16], 22),
            23: Line.create([16, 17], 23),
            24: Line.create([17, 18], 24),
        }

        expected_rail_geometry = Geometry(expected_rail_points, expected_rail_lines)

        # check rail model part
        rail_model_part = model.body_model_parts[1]
        calculated_rail_geometry = rail_model_part.geometry
        calculated_rail_parameters = rail_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_rail_geometry, calculated_rail_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_parameters.__dict__, calculated_rail_parameters.__dict__)

        # check geometry and material of the sleepers
        expected_sleeper_points = {
            9: Point.create([2.0, 3.0, -0.5], 9),
            10: Point.create([2.0, 3.0, 0.0], 10),
            11: Point.create([2.0, 3.0, 0.5], 11),
            12: Point.create([2.0, 3.0, 1.0], 12),
        }

        expected_sleeper_geometry = Geometry(expected_sleeper_points)

        sleeper_model_part = model.body_model_parts[2]
        calculated_sleeper_geometry = sleeper_model_part.geometry
        calculated_sleeper_parameters = sleeper_model_part.material.material_parameters

        TestUtils.assert_almost_equal_geometries(expected_sleeper_geometry, calculated_sleeper_geometry)
        TestUtils.assert_dictionary_almost_equal(sleeper_parameters.__dict__, calculated_sleeper_parameters.__dict__)

        # check geometry and material of the rail pads
        rail_pad_model_part = model.body_model_parts[3]
        calculated_rail_pad_geometry = rail_pad_model_part.geometry
        calculated_rail_pad_parameters = rail_pad_model_part.material.material_parameters

        expected_rail_pad_points = {
            14: Point.create([2.0, 3.02, -0.5], 14),
            9: Point.create([2.0, 3.0, -0.5], 9),
            15: Point.create([2.0, 3.02, 0.0], 15),
            10: Point.create([2.0, 3.0, 0.0], 10),
            16: Point.create([2.0, 3.02, 0.5], 16),
            11: Point.create([2.0, 3.0, 0.5], 11),
            17: Point.create([2.0, 3.02, 1.0], 17),
            12: Point.create([2.0, 3.0, 1.0], 12),
            18: Point.create([2.0, 3.02, 1.5], 18),
            13: Point.create([2.0, 3.0, 1.5], 13),
        }

        expected_rail_pad_lines = {
            25: Line.create([14, 9], 25),
            26: Line.create([15, 10], 26),
            27: Line.create([16, 11], 27),
            28: Line.create([17, 12], 28),
            29: Line.create([18, 13], 29),
        }

        expected_rail_pad_geometry = Geometry(expected_rail_pad_points, expected_rail_pad_lines)

        TestUtils.assert_almost_equal_geometries(expected_rail_pad_geometry, calculated_rail_pad_geometry)
        TestUtils.assert_dictionary_almost_equal(rail_pad_parameters.__dict__, calculated_rail_pad_parameters.__dict__)

        # check rotation constrain model part
        rotation_constrain_model_part = model.process_model_parts[1]
        calculated_rotation_constrain_geometry = rotation_constrain_model_part.geometry
        calculated_rotation_constrain_parameters = rotation_constrain_model_part.parameters

        expected_rotation_constrain_points = {14: Point.create([2.0, 3.02, -0.5], 14)}
        expected_rotation_constrain_geometry = Geometry(expected_rotation_constrain_points)

        expected_rotation_constraint_parameters = RotationConstraint(value=[0, 0, 0],
                                                                     is_fixed=[True, True, True],
                                                                     active=[True, True, True])

        TestUtils.assert_almost_equal_geometries(expected_rotation_constrain_geometry,
                                                 calculated_rotation_constrain_geometry)
        TestUtils.assert_dictionary_almost_equal(expected_rotation_constraint_parameters.__dict__,
                                                 calculated_rotation_constrain_parameters.__dict__)

        # check geometry and material of soil equivalent
        soil_equivalent_model_part = model.body_model_parts[4]
        calculated_soil_equivalent_geometry = soil_equivalent_model_part.geometry
        calculated_soil_equivalent_parameters = soil_equivalent_model_part.material.material_parameters

        expected_soil_equivalent_points = {
            9: Point.create([2.0, 3.0, -0.5], 9),
            19: Point.create([2.0, -2.0, -0.5], 19),
            13: Point.create([2.0, 3.0, 1.5], 13),
            20: Point.create([2.0, -2.0, 1.5], 20),
        }

        expected_soil_equivalent_lines = {
            30: Line.create([9, 19], 30),
            31: Line.create([13, 20], 31),
        }

        expected_soil_equivalent_geometry = Geometry(expected_soil_equivalent_points, expected_soil_equivalent_lines)

        TestUtils.assert_almost_equal_geometries(expected_soil_equivalent_geometry, calculated_soil_equivalent_geometry)
        TestUtils.assert_dictionary_almost_equal(extended_soil_parameters.__dict__,
                                                 calculated_soil_equivalent_parameters.__dict__)

    def test_add_hinge_on_beam(self, create_default_3d_beam):
        """
        Test if a hinge is added correctly on a beam. A model is created with a beam and a hinge is added on the beam.
        This test also checks for errors.
        """

        model = Model(3)
        beam_material = create_default_3d_beam
        # Specify the coordinates for the beam: x:1m x y:0m
        beam_coordinates = [(0, 0, 0), (1, 0, 0)]
        # Create the beam
        gmsh_input = {beam_material.name: {"coordinates": beam_coordinates, "ndim": 1}}

        # check if extrusion length is specified in 3D
        model.gmsh_io.generate_geometry(gmsh_input, "")
        #
        # create body model part
        body_model_part = BodyModelPart(beam_material.name)
        body_model_part.material = beam_material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, beam_material.name)
        model.body_model_parts.append(body_model_part)

        hinge_parameters = HingeParameters(1e9, 1e9)

        model.add_hinge_on_beam(beam_material.name, [(0.2, 0.0, 0.0)], hinge_parameters, "hinge_1")

        # check if hinge is added to the model
        assert model.process_model_parts[0].name == "hinge_1"
        TestUtils.assert_almost_equal_geometries(model.process_model_parts[0].geometry,
                                                 Geometry({3: Point.create([0.2, 0.0, 0.0], 3)}))
        assert model.process_model_parts[0].parameters == hinge_parameters

        # check that the hinge node is part of the beam model part
        assert 3 in model.body_model_parts[0].geometry.points.keys()

        # try to add hinge outside of beam
        with pytest.raises(ValueError, match="The hinge points are not part of the beam model part `beam`."):
            model.add_hinge_on_beam(beam_material.name, [(0.2, 0.2, 0.0)], hinge_parameters, "hinge_2")

        # try to add hinge on a spring damper
        spring_damper_parameters = ElasticSpringDamper([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
        spring_damper_material = StructuralMaterial(name="spring_damper", material_parameters=spring_damper_parameters)
        gmsh_input = {spring_damper_material.name: {"coordinates": [(-1, 0, 0), (0, 0, 0)], "ndim": 1}}
        model.gmsh_io.generate_geometry(gmsh_input, "")

        spring_damper_part = BodyModelPart("spring_damper")
        spring_damper_part.material = spring_damper_material
        spring_damper_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, spring_damper_material.name)
        model.body_model_parts.append(spring_damper_part)

        with pytest.raises(ValueError, match="Hinges can only be applied to beam model parts"):
            model.add_hinge_on_beam(spring_damper_part.name, [(-0.2, 0.0, 0.0)], hinge_parameters, "hinge_3")

        # try to add hinge on non existing model part
        with pytest.raises(ValueError, match="Model part `non_existing_part` not found."):
            model.add_hinge_on_beam("non_existing_part", [(-0.2, 0.0, 0.0)], hinge_parameters, "hinge_4")

        # try to add hinge on 2D model
        model.ndim = 2
        with pytest.raises(NotImplementedError, match="Hinges can only be applied in 3D models"):
            model.add_hinge_on_beam(beam_material.name, [(-0.2, 0.0, 0.0)], hinge_parameters, "hinge_5")

    def test_create_rail_model_part(self):
        """
        Tests if the rail model part is correctly created.

        """
        model = Model(3)
        rail_name = "rail1"
        # add rail geometry
        rail_geo_settings = {rail_name: {"coordinates": [(0, 0, 0), (0, 2, 0), (0, 2, 1)], "ndim": 1}}
        model.gmsh_io.generate_geometry(rail_geo_settings, "")
        # set up materials
        rail_params = EulerBeam(1, 1, 1, 1, 1, 1)
        rail_part = model._Model__create_rail_model_part(rail_name, rail_params)
        # Check that the geometry was set by our dummy function.
        assert rail_part.geometry is not None
        # Check that material was set properly.
        TestUtils.assert_dictionary_almost_equal(rail_params.__dict__, rail_part.material.material_parameters.__dict__)
        assert rail_part.material.name == rail_name

    def test_create_sleeper_model_parts_nodal(self):
        """
        Tests if the sleeper model part is correctly created when the sleeper is a nodal concentrated type.

        """
        model = Model(3)
        sleeper_name = "sleeper1"
        model.gmsh_io.generate_geometry(
            {sleeper_name: {
                "coordinates": [(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)],
                "ndim": 0
            }}, "")
        sleeper_params = NodalConcentrated(1, 1, 1)
        part = model._Model__create_sleeper_model_parts(sleeper_name, sleeper_params)
        assert part.geometry is not None
        assert part.name == sleeper_name
        assert part.material.material_parameters == sleeper_params

    def test_create_sleeper_model_parts_soil(self, create_default_3d_soil_material: SoilMaterial):
        """
        Tests if the sleeper model part is correctly created when the sleeper is a soil material.

        Args:
            - create_default_3d_soil_material (:class:`stem.soil_material.SoilMaterial`): default soil material
        """
        model = Model(3)
        sleeper_name = "sleeper1"
        model.gmsh_io.generate_geometry(
            {
                sleeper_name: {
                    "coordinates": [(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)],
                    "ndim": 3,
                    "extrusion_length": [1, 0, 0]
                }
            }, "")
        sleeper_params = create_default_3d_soil_material
        part = model._Model__create_sleeper_model_parts(sleeper_name, sleeper_params)
        assert part.geometry is not None
        assert part.name == sleeper_name
        assert part.material == sleeper_params

    def test_create_rail_pads_model_part(self):
        """
        Tests if the rail pads model part is correctly created.

        """
        model = Model(3)
        rail_pads_name = "rail_pads1"
        model.gmsh_io.generate_geometry(
            {rail_pads_name: {
                "coordinates": [(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)],
                "ndim": 1
            }}, "")
        pad_params = ElasticSpringDamper(1, 1, 1, 1)
        pads_part = model._Model__create_rail_pads_model_part(rail_pads_name, pad_params)
        assert pads_part.geometry is not None
        assert pads_part.name == rail_pads_name
        assert pads_part.material.material_parameters == pad_params

    def test_create_rail_constraint_model_part(self):
        """
        Tests if the constraint model part is correctly created.

        """
        model = Model(3)
        rail_name = "track1"
        model.gmsh_io.generate_geometry(
            {rail_name: {
                "coordinates": [(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)],
                "ndim": 1
            }}, "")
        constraint_params = DisplacementConstraint([True, False, True], [True, False, True], [0.0, 0.0, 0.0])
        constraint_part = model._Model__create_rail_constraint_model_part(rail_name)
        assert constraint_part.geometry is not None
        assert constraint_part.name == "constraint_" + rail_name
        TestUtils.assert_dictionary_almost_equal(constraint_params.__dict__, constraint_part.parameters.__dict__)

    def test_create_no_rotation_model_part(self):
        """
        Tests if the no rotation model part is correctly created.

        """
        model = Model(3)
        rail_name = "track1"
        global_rail_coords = np.array([(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)])
        model.gmsh_io.generate_geometry(
            {rail_name: {
                "coordinates": [(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)],
                "ndim": 1
            }}, "")
        no_rotation_params = RotationConstraint([True, True, True], [True, True, True], [0.0, 0.0, 0.0])
        no_rotation_part = model._Model__create_rail_no_rotation_model_part(rail_name, global_rail_coords)
        assert no_rotation_part.geometry is not None
        assert no_rotation_part.name == "rotation_constraint_" + rail_name
        TestUtils.assert_dictionary_almost_equal(no_rotation_params.__dict__, no_rotation_part.parameters.__dict__)

    def test_generate_sleeper_base_coordinates_at_origin(self):
        """
        Test the creation of sleeper base coordinates at the origin when the sleeper is not rotated.

        This test ensures that when the direction vector is [1, 0, 0] (i.e. aligned with the
        local x-axis), the generated sleeper base coordinates remain as defined in the local system,
        translated by the global origin.

        """
        sleeper_rail_pad_offset = 0.5
        global_coord = [0.0, 0.0, 0.0]
        sleeper_dimensions = [2.0, 4.0, 1.0]  # length, width, height
        direction_vector = [1.0, 0.0, 0.0]  # rotated 90 degrees
        expected = np.array([[2.0, 0.0, -1.5], [-2.0, 0.0, -1.5], [-2.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        result = Model._Model__generate_sleeper_base_coordinates(global_coord, sleeper_dimensions,
                                                                 sleeper_rail_pad_offset, direction_vector)
        np.testing.assert_array_almost_equal(result, expected)

    def test_generate_sleeper_base_coordinates_nonzero_origin(self):
        """
        Test the creation of a sleeper base coordinates with a non-zero origin.

        This test ensures that the function correctly calculates the sleeper base coordinates
        when given a non-zero origin. The sleeper is not rotated.

        """
        local_coord = [1.0, -1.0, 0.5]
        sleeper_dimensions = [3.0, 2.0, 0.5]  # length, width, height
        sleeper_rail_pad_offset = 0.5
        direction_vector = [0.0, 0.0, 1.0]  # no rotation

        expected = np.array([[3.5, -1.0, 1.5], [3.5, -1.0, -0.5], [0.5, -1.0, -0.5], [0.5, -1.0, 1.5]])

        result = Model._Model__generate_sleeper_base_coordinates(local_coord, sleeper_dimensions,
                                                                 sleeper_rail_pad_offset, direction_vector)
        np.testing.assert_array_almost_equal(result, expected)

    def test_generate_sleeper_base_coordinates_with_negative_dimensions(self):
        """
        Test the creation of sleeper base coordinates with negative dimensions.

        While negative dimensions might not be physically meaningful,
        this test ensures that the function handles them consistently.

        The test checks if the function correctly calculates the sleeper base coordinates
        when given negative dimensions.

        Asserts that the result matches the expected output.
        """
        local_coord = [2.0, 3.0, 4.0]
        sleeper_dimensions = [-2.0, -4.0, 1.0]
        sleeper_rail_pad_offset = 0.5
        expected = np.array([[-0.5, 3.0, 2.0], [-0.5, 3.0, 6.0], [1.5, 3.0, 6.0], [1.5, 3.0, 2.0]])
        direction_vector = [0.0, 0.0, 1.0]  # no rotation
        result = Model._Model__generate_sleeper_base_coordinates(local_coord, sleeper_dimensions,
                                                                 sleeper_rail_pad_offset, direction_vector)
        np.testing.assert_array_almost_equal(result, expected)

    def test_split_second_order_elements_with_beam_and_load(self, create_default_3d_beam: StructuralMaterial,
                                                            create_default_moving_load_parameters: MovingLoad):
        """
        Test if second order elements are split correctly when a beam and a load are present in the model.

        Args:
            - create_default_3d_beam (:class:`stem.structural_material.StructuralMaterial`): default beam material
            - create_default_moving_load_parameters (:class:`stem.load.MovingLoad`): default moving load parameters

        """

        model = Model(3)

        # pre-set gmsh mesh data of all parts
        model.gmsh_io._GmshIO__mesh_data = {
            'elements': {
                'LINE_3N': {
                    1: [1, 2, 3]
                },
                'POINT_1N': {
                    2: [1],
                    3: [2]
                }
            },
            'ndim': 1,
            'nodes': {
                1: [0.0, 0.0, 0.0],
                2: [1.0, 0.0, 0.0],
                3: [0.5, 0.0, 0.0]
            },
            'physical_groups': {
                'beam': {
                    'element_ids': [1],
                    'element_type': 'LINE_3N',
                    'ndim': 1,
                    'node_ids': [1, 2, 3]
                },
                'load': {
                    'element_ids': [1],
                    'element_type': 'LINE_3N',
                    'ndim': 1,
                    'node_ids': [1, 2, 3]
                }
            }
        }

        beam_material = create_default_3d_beam
        # Specify the coordinates for the beam: x:1m x y:0m
        beam_coordinates = [(0, 0, 0), (1, 0, 0)]
        # Create the beam
        gmsh_input = {beam_material.name: {"coordinates": beam_coordinates, "ndim": 1}}

        # check if extrusion length is specified in 3D
        model.gmsh_io.generate_geometry(gmsh_input, "")
        #
        # create body model part
        body_model_part = BodyModelPart(beam_material.name)
        body_model_part.material = beam_material

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, beam_material.name)
        model.body_model_parts.append(body_model_part)

        body_model_part.mesh = Mesh.create_mesh_from_gmsh_group(model.gmsh_io.mesh_data, beam_material.name)

        # add load on top of beam
        load_parameters = create_default_moving_load_parameters

        load_model_part = ModelPart("load")
        load_model_part.parameters = load_parameters
        model.gmsh_io.add_physical_group("load", 1, [1])

        load_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, load_model_part.name)
        model.process_model_parts.append(load_model_part)

        # set the mesh of the load model part
        load_model_part.mesh = Mesh.create_mesh_from_gmsh_group(model.gmsh_io.mesh_data, load_model_part.name)

        # perform test
        model._Model__split_second_order_elements()

        expected_gmsh_mesh = {
            'elements': {
                'LINE_2N': {
                    4: [1, 3],
                    5: [3, 2]
                }
            },
            'ndim': 1,
            'nodes': {
                1: [0.0, 0.0, 0.0],
                2: [1.0, 0.0, 0.0],
                3: [0.5, 0.0, 0.0]
            },
            'physical_groups': {
                'beam': {
                    'element_ids': [4, 5],
                    'element_type': 'LINE_2N',
                    'ndim': 1,
                    'node_ids': [1, 2, 3]
                },
                'load': {
                    'element_ids': [4, 5],
                    'element_type': 'LINE_2N',
                    'ndim': 1,
                    'node_ids': [1, 2, 3]
                }
            }
        }

        expected_model_part_mesh = Mesh(1)
        expected_model_part_mesh.nodes = {
            1: Node(1, [0.0, 0.0, 0.0]),
            2: Node(2, [1.0, 0.0, 0.0]),
            3: Node(3, [0.5, 0.0, 0.0])
        }
        expected_model_part_mesh.elements = {4: Element(4, "LINE_2N", [1, 3]), 5: Element(5, "LINE_2N", [3, 2])}

        # both the beam and the load should have the same 1st order mesh
        TestUtils.assert_dictionary_almost_equal(expected_gmsh_mesh, model.gmsh_io.mesh_data)

        assert expected_model_part_mesh == model.body_model_parts[0].mesh
        assert expected_model_part_mesh == model.process_model_parts[0].mesh

    def test_split_second_order_elements_with_beam_load_and_soil(self, create_default_3d_beam: StructuralMaterial,
                                                                 create_default_moving_load_parameters: MovingLoad,
                                                                 create_default_2d_soil_material: SoilMaterial):
        """
        Test if second order elements are split correctly when a beam, a load and a soil are present in the model.

        Args:
            - create_default_3d_beam (:class:`stem.structural_material.StructuralMaterial`): default beam material
            - create_default_moving_load_parameters (:class:`stem.load.MovingLoad`): default moving load parameters
            - create_default_2d_soil_material (:class:`stem.soil_material.SoilMaterial`): default 2D soil material
        """

        # set dim of beam to 2D
        beam_parameters = create_default_3d_beam
        beam_parameters.ndim = 2

        model = Model(2)

        # pre-set gmsh mesh data of all parts
        model.gmsh_io._GmshIO__mesh_data = {
            'elements': {
                'LINE_3N': {
                    1: [1, 2, 3],
                    2: [2, 4, 5]
                },
                "TRIANGLE_6N": {
                    3: [6, 2, 1, 8, 3, 7],
                    4: [9, 4, 2, 10, 5, 11],
                    5: [6, 9, 2, 12, 11, 8]
                }
            },
            'ndim': 1,
            'nodes': {
                1: [0.0, 0.0, 0.0],
                2: [1.0, 0.0, 0.0],
                3: [0.5, 0.0, 0.0],
                4: [2.0, 0.0, 0.0],
                5: [1.5, 0.0, 0.0],
                6: [0.0, -1.0, 0.0],
                7: [0.0, -0.5, 0.0],
                8: [0.5, -0.5, 0.0],
                9: [2.0, -1.0, 0.0],
                10: [2.0, -0.5, 0.0],
                11: [1.5, -0.5, 0.0],
                12: [1.0, -1.0, 0.0]
            },
            'physical_groups': {
                'beam': {
                    'element_ids': [1],
                    'element_type': 'LINE_3N',
                    'ndim': 1,
                    'node_ids': [1, 2, 3]
                },
                'load': {
                    'element_ids': [1, 2],
                    'element_type': 'LINE_3N',
                    'ndim': 1,
                    'node_ids': [1, 2, 3, 4, 5]
                },
                'soil': {
                    'element_ids': [3, 4, 5],
                    'element_type': 'TRIANGLE_6N',
                    'ndim': 2,
                    'node_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                }
            }
        }

        # create beam body model part
        beam_model_part = BodyModelPart(beam_parameters.name)
        beam_model_part.material = beam_parameters
        beam_model_part.mesh = Mesh.create_mesh_from_gmsh_group(model.gmsh_io.mesh_data, beam_model_part.name)
        model.body_model_parts.append(beam_model_part)

        # create load process model part
        load_parameters = create_default_moving_load_parameters
        load_model_part = ModelPart("load")
        load_model_part.parameters = load_parameters
        load_model_part.mesh = Mesh.create_mesh_from_gmsh_group(model.gmsh_io.mesh_data, load_model_part.name)
        model.process_model_parts.append(load_model_part)

        # create soil body model part
        soil_model_part = BodyModelPart(create_default_2d_soil_material.name)
        soil_model_part.material = create_default_2d_soil_material
        soil_model_part.mesh = Mesh.create_mesh_from_gmsh_group(model.gmsh_io.mesh_data, soil_model_part.name)

        model.body_model_parts.append(soil_model_part)

        # perform test
        model._Model__split_second_order_elements()

        # expected beam mesh is a split of the original LINE_3N element into two LINE_2N elements
        expected_beam_mesh = Mesh(1)
        expected_beam_mesh.nodes = {
            1: Node(1, [0.0, 0.0, 0.0]),
            2: Node(2, [1.0, 0.0, 0.0]),
            3: Node(3, [0.5, 0.0, 0.0])
        }
        expected_beam_mesh.elements = {6: Element(6, "LINE_2N", [1, 3]), 7: Element(7, "LINE_2N", [3, 2])}

        # expected load mesh is at the location of the beam, a split of the original LINE_3N element into two LINE_2N elements
        # and the second LINE_3N element is kept as it is
        expected_load_mesh = Mesh(1)
        expected_load_mesh.nodes = {
            1: Node(1, [0.0, 0.0, 0.0]),
            2: Node(2, [1.0, 0.0, 0.0]),
            3: Node(3, [0.5, 0.0, 0.0]),
            4: Node(4, [2.0, 0.0, 0.0]),
            5: Node(5, [1.5, 0.0, 0.0])
        }
        expected_load_mesh.elements = {
            2: Element(2, "LINE_3N", [2, 4, 5]),
            6: Element(6, "LINE_2N", [1, 3]),
            7: Element(7, "LINE_2N", [3, 2])
        }

        # expected soil mesh is kept as it is
        expected_soil_mesh = Mesh(2)
        expected_soil_mesh.nodes = {
            1: Node(1, [0.0, 0.0, 0.0]),
            2: Node(2, [1.0, 0.0, 0.0]),
            3: Node(3, [0.5, 0.0, 0.0]),
            4: Node(4, [2.0, 0.0, 0.0]),
            5: Node(5, [1.5, 0.0, 0.0]),
            6: Node(6, [0.0, -1.0, 0.0]),
            7: Node(7, [0.0, -0.5, 0.0]),
            8: Node(8, [0.5, -0.5, 0.0]),
            9: Node(9, [2.0, -1.0, 0.0]),
            10: Node(10, [2.0, -0.5, 0.0]),
            11: Node(11, [1.5, -0.5, 0.0]),
            12: Node(12, [1.0, -1.0, 0.0])
        }
        expected_soil_mesh.elements = {
            3: Element(3, "TRIANGLE_6N", [6, 2, 1, 8, 3, 7]),
            4: Element(4, "TRIANGLE_6N", [9, 4, 2, 10, 5, 11]),
            5: Element(5, "TRIANGLE_6N", [6, 9, 2, 12, 11, 8])
        }

        # check if the meshes are split correctly
        assert expected_beam_mesh == model.body_model_parts[0].mesh
        assert expected_load_mesh == model.process_model_parts[0].mesh
        assert expected_soil_mesh == model.body_model_parts[1].mesh

    def test_reorder_gmsh_to_kratos_order(self):
        """
        Tests if the GMSH mesh data is reordered to match the Kratos order for tetrahedron and hexahedron elements.
        """

        model = Model(2)

        # set gmsh mesh data manually
        model.gmsh_io.mesh_data["elements"] = {
            "TETRAHEDRON_10N": {
                1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            "HEXAHEDRON_20N": {
                2: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            }
        }

        # perform the reordering
        model._Model__reorder_gmsh_to_kratos_order()

        expected_tetrahedron = {1: [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]}
        expected_hexahedron = {2: [11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 20, 21, 23, 25, 26, 27, 29, 30, 28]}

        # Check if the elements are reordered correctly
        assert model.gmsh_io.mesh_data["elements"]["TETRAHEDRON_10N"] == expected_tetrahedron
        assert model.gmsh_io.mesh_data["elements"]["HEXAHEDRON_20N"] == expected_hexahedron

    def test_set_mesh_constraints_volume(self, expected_geometry_single_layer_3D: Geometry):
        """
        Tests if the mesh constraints for a volume are set correctly.

        Args:
            expected_geometry_single_layer_3D (:class:`stem.geometry.Geometry`): A 3D block geometry.

        """

        model = Model(3)

        # transform stem geometry to gmsh geo data
        model.gmsh_io.geo_data["points"] = {
            key: value.coordinates
            for key, value in expected_geometry_single_layer_3D.points.items()
        }
        model.gmsh_io.geo_data["lines"] = {
            key: value.point_ids
            for key, value in expected_geometry_single_layer_3D.lines.items()
        }
        model.gmsh_io.geo_data["surfaces"] = {
            key: value.line_ids
            for key, value in expected_geometry_single_layer_3D.surfaces.items()
        }
        model.gmsh_io.geo_data["volumes"] = {
            key: value.surface_ids
            for key, value in expected_geometry_single_layer_3D.volumes.items()
        }

        # set constraints for volume
        model.mesh_settings.constraints["transfinite_volume"][1] = [2, 3, 4]

        # perform test
        model._Model__set_mesh_constraints_for_structured_mesh()

        # expected constraints dictionary for the volume
        expected_constraint_dict = {
            'transfinite_curve': {
                1: {
                    'n_points': 2
                },
                2: {
                    'n_points': 3
                },
                3: {
                    'n_points': 2
                },
                4: {
                    'n_points': 3
                },
                5: {
                    'n_points': 4
                },
                6: {
                    'n_points': 4
                },
                7: {
                    'n_points': 2
                },
                8: {
                    'n_points': 4
                },
                9: {
                    'n_points': 3
                },
                10: {
                    'n_points': 4
                },
                11: {
                    'n_points': 2
                },
                12: {
                    'n_points': 3
                }
            },
            'transfinite_surface': {
                1: {
                    'corner_node_ids': [1, 2, 3, 4],
                    'n_points': [2, 3, 4]
                },
                2: {
                    'corner_node_ids': [1, 5, 6, 2],
                    'n_points': [2, 3, 4]
                },
                3: {
                    'corner_node_ids': [2, 6, 7, 3],
                    'n_points': [2, 3, 4]
                },
                4: {
                    'corner_node_ids': [3, 7, 8, 4],
                    'n_points': [2, 3, 4]
                },
                5: {
                    'corner_node_ids': [4, 8, 5, 1],
                    'n_points': [2, 3, 4]
                },
                6: {
                    'corner_node_ids': [5, 6, 7, 8],
                    'n_points': [2, 3, 4]
                }
            },
            'transfinite_volume': {
                1: {
                    'corner_node_ids': [1, 2, 3, 4, 5, 6, 7, 8],
                    'n_points': [2, 3, 4]
                }
            }
        }

        # assert that the constraints dictionary is set correctly
        assert model.gmsh_io.geo_data["constraints"] == expected_constraint_dict

    def test_set_mesh_constraints_surface(self, expected_geometry_single_layer_3D: Geometry):
        """
        Tests if the mesh constraints for a surface are set correctly.

        Args:
            expected_geometry_single_layer_3D (:class:`stem.geometry.Geometry`): A 3D block geometry.
        """

        model = Model(3)

        # transform stem geometry to gmsh geo data
        model.gmsh_io.geo_data["points"] = {
            key: value.coordinates
            for key, value in expected_geometry_single_layer_3D.points.items()
        }
        model.gmsh_io.geo_data["lines"] = {
            key: value.point_ids
            for key, value in expected_geometry_single_layer_3D.lines.items()
        }
        model.gmsh_io.geo_data["surfaces"] = {
            key: value.line_ids
            for key, value in expected_geometry_single_layer_3D.surfaces.items()
        }
        model.gmsh_io.geo_data["volumes"] = {
            key: value.surface_ids
            for key, value in expected_geometry_single_layer_3D.volumes.items()
        }

        # set constraints for surface in the settings
        model.mesh_settings.constraints["transfinite_surface"][1] = [2, 3, 4]

        # perform test
        model._Model__set_mesh_constraints_for_structured_mesh()

        # expected constraints dictionary for the surface
        expected_constraint_dict = {
            'transfinite_curve': {
                1: {
                    'n_points': 2
                },
                2: {
                    'n_points': 3
                },
                3: {
                    'n_points': 2
                },
                4: {
                    'n_points': 3
                }
            },
            'transfinite_surface': {
                1: {
                    'corner_node_ids': [1, 2, 3, 4],
                    'n_points': [2, 3, 4]
                }
            }
        }

        # assert that the constraints dictionary is set correctly
        assert model.gmsh_io.geo_data["constraints"] == expected_constraint_dict

    def test_set_mesh_constraints_curve(self, expected_geometry_single_layer_3D: Geometry):
        """
        Tests if the mesh constraints for a curve are set correctly.

        Args:
            expected_geometry_single_layer_3D (:class:`stem.geometry.Geometry`): A 3D block geometry.
        """

        model = Model(3)

        # transform stem geometry to gmsh geo data
        model.gmsh_io.geo_data["points"] = {
            key: value.coordinates
            for key, value in expected_geometry_single_layer_3D.points.items()
        }
        model.gmsh_io.geo_data["lines"] = {
            key: value.point_ids
            for key, value in expected_geometry_single_layer_3D.lines.items()
        }
        model.gmsh_io.geo_data["surfaces"] = {
            key: value.line_ids
            for key, value in expected_geometry_single_layer_3D.surfaces.items()
        }
        model.gmsh_io.geo_data["volumes"] = {
            key: value.surface_ids
            for key, value in expected_geometry_single_layer_3D.volumes.items()
        }

        # set constraints for a curve in the settings
        model.mesh_settings.constraints["transfinite_curve"][1] = 2

        # perform test
        model._Model__set_mesh_constraints_for_structured_mesh()

        # expected constraints dictionary for the curve
        expected_constraint_dict = {'transfinite_curve': {1: {'n_points': 2}}}

        # assert that the constraints dictionary is set correctly
        assert model.gmsh_io.geo_data["constraints"] == expected_constraint_dict

    def test_update_node_ids_3d(self, model_setup_3d_with_interface: Dict[str, Any]):
        """
        Test updating node IDs with a mapping in a 3D model

        Args:
            - model_setup_3d_with_interface (Dict[str, Any]): Dictionary containing the 3D model and other test data.
        """
        # Create a mapping for node IDs 1, 2 and 3 (common nodes)
        map_new_node_ids = {1: 100, 2: 101, 3: 102}

        # Test the static method directly
        original_nodes = {k: v for k, v in model_setup_3d_with_interface["changing_part"].mesh.nodes.items()}
        updated_nodes = Model._Model__update_node_ids(original_nodes, map_new_node_ids)

        # Verify that nodes 1 and 2 have been updated in the result
        assert 100 in updated_nodes
        assert 101 in updated_nodes
        assert 1 not in updated_nodes
        assert 2 not in updated_nodes
        assert 102 in updated_nodes
        assert 3 not in updated_nodes

        # Verify that node 5 remains unchanged
        assert 5 in updated_nodes

        # Verify that the node objects have their IDs updated
        assert updated_nodes[100].id == 100
        assert updated_nodes[101].id == 101
        assert updated_nodes[102].id == 102

        # Verify coordinates are preserved
        for i, node_id in enumerate([100, 101, 102]):
            assert updated_nodes[node_id].coordinates == model_setup_3d_with_interface["coords"][i]

    def test_update_node_ids(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test updating node IDs with a mapping

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 100, 3: 101}

        # Test the static method directly
        original_nodes = {k: v for k, v in model_2d_with_interface["changing_part"].mesh.nodes.items()}
        updated_nodes = Model._Model__update_node_ids(original_nodes, map_new_node_ids)

        # Verify that nodes 2 and 3 have been updated in the result
        assert 100 in updated_nodes
        assert 101 in updated_nodes
        assert 2 not in updated_nodes
        assert 3 not in updated_nodes

        # Verify that node 4 remains unchanged
        assert 4 in updated_nodes

        # Verify that the node objects have their IDs updated
        assert updated_nodes[100].id == 100
        assert updated_nodes[101].id == 101

        # Verify coordinates are preserved
        for i, node_id in enumerate([100, 101], start=1):
            assert updated_nodes[node_id].coordinates == model_2d_with_interface["coords"][i]

    def test_update_elements_with_new_node_ids_3d(self, model_setup_3d_with_interface: Dict[str, Any]):
        """
        Test updating elements with new node IDs in a 3D model

        Args:
            - model_setup_3d_with_interface (Dict[str, Any]): Dictionary containing the 3D model and other test data.
        """
        # Create a mapping for node IDs 1 and 2 (common nodes)
        map_new_node_ids = {1: 100, 2: 101, 3: 102}

        # Create node_to_elements mapping
        node_to_elements = {
            100: [2],  # Element 1 is connected to node 1 (now 100)
            101: [2],  # Element 1 is connected to node 2 (now 101)
            102: [2]  # Element 2 is connected to node 3 (now 102)
        }

        # Original elements
        original_elements = {k: v for k, v in model_setup_3d_with_interface["changing_part"].mesh.elements.items()}

        # Test the static method
        updated_elements = Model._Model__update_elements_with_new_node_ids(original_elements, node_to_elements,
                                                                           map_new_node_ids)

        # Verify element 2 now references the new node IDs
        assert updated_elements[2].node_ids == [100, 101, 102, 5]

    def test_update_elements_with_new_node_ids(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test updating elements with new node IDs

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 100, 3: 101}

        # Create node_to_elements mapping
        node_to_elements = {
            100: [2],  # Element 2 is connected to node 2 (now 100)
            101: [2]  # Element 2 is connected to node 3 (now 101)
        }

        # Original elements
        original_elements = {k: v for k, v in model_2d_with_interface["changing_part"].mesh.elements.items()}

        # Test the static method
        updated_elements = Model._Model__update_elements_with_new_node_ids(original_elements, node_to_elements,
                                                                           map_new_node_ids)

        # Verify element 2 now references the new node IDs
        assert updated_elements[2].node_ids == [100, 101, 4]

    def test_get_interface_config_2d(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test getting interface configuration based on dimensions and element type for 2D model

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]

        # Test 2D configuration
        n_nodes, element_type = model._Model__get_interface_config()
        assert n_nodes == 4
        assert element_type == "QUADRANGLE_4N"

    def test_get_interface_config_3d(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test getting interface configuration based on dimensions and element type for 3D model

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        model.ndim = 3

        # Test 2D configuration
        n_nodes, element_type = model._Model__get_interface_config()
        assert n_nodes == 6
        assert element_type == "PRISM_6N"

        # Reset to 2D for other tests
        model.ndim = 2

    def test_get_interface_config_unknown_dim(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test getting interface configuration based on dimensions and element type for unknown
        dimensions. Should raise ValueError.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        # Test 4D configuration (should raise ValueError)
        model.ndim = 4

        with pytest.raises(ValueError, match="Unsupported number of dimensions: 4"):
            model._Model__get_interface_config()

    def test_update_changing_parts_3d(self, model_setup_3d_with_interface: Dict[str, Any]):
        """
        Test updating changing parts with new node IDs in a 3D model

        Args:
            - model_setup_3d_with_interface (Dict[str, Any]): Dictionary containing the 3D model and other test data.
        """
        model = model_setup_3d_with_interface["model"]
        changing_part = model_setup_3d_with_interface["changing_part"]

        # Create common nodes set (nodes 1 and 2)
        common_nodes = {1, 2, 3}

        # Create mapping
        map_new_node_ids = {1: 100, 2: 101, 3: 102}

        # Get index of changing part
        indexes_changing_parts = [1]
        # Update changing parts
        model._Model__update_changing_parts([changing_part], indexes_changing_parts, common_nodes, map_new_node_ids, {})
        # Verify nodes in changing part have been updated
        updated_nodes = model.body_model_parts[1].mesh.nodes
        assert 100 in updated_nodes
        assert 101 in updated_nodes
        assert 102 in updated_nodes
        assert 1 not in updated_nodes
        assert 2 not in updated_nodes
        assert 3 not in updated_nodes

        # Verify element node IDs have been updated
        updated_element = model.body_model_parts[1].mesh.elements[2]
        expected_node_ids = [100, 101, 102, 5]  # Updated from [1, 2, 3, 5]
        assert updated_element.node_ids == expected_node_ids

    def test_update_changing_parts(self, model_2d_with_interface: Dict[str, Any]):
        """

        Test updating changing parts with new node IDs

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        changing_part = model_2d_with_interface["changing_part"]

        # Create common nodes set (nodes 2 and 3)
        common_nodes = {2, 3}

        # Create mapping
        map_new_node_ids = {2: 100, 3: 101}

        # Get index of changing part
        indexes_changing_parts = [1]  # Index 1 in model.body_model_parts

        # Update changing parts
        model._Model__update_changing_parts([changing_part], indexes_changing_parts, common_nodes, map_new_node_ids, {})

        # Verify nodes in changing part have been updated
        updated_nodes = model.body_model_parts[1].mesh.nodes
        assert 100 in updated_nodes
        assert 101 in updated_nodes
        assert 2 not in updated_nodes
        assert 3 not in updated_nodes

        # Verify element node IDs have been updated
        updated_element = model.body_model_parts[1].mesh.elements[2]
        expected_node_ids = [100, 101, 4]  # Updated from [2, 5, 6, 3]
        assert updated_element.node_ids == expected_node_ids

    def test_create_interface_elements_TRIANGLE_3N(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test creating interface elements from nodes for 2D model TRIANGLE_3N that
        raise ValueError

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]

        # Set up test data
        test_nodes = {
            3: Node(3, [1.0, 0.0, 0.0]),
            5: Node(5, [1.0, 0.0, 0.0]),  # Mapped from node 2
            6: Node(6, [1.0, 1.0, 0.0]),  # Mapped from node 3
            2: Node(2, [1.0, 1.0, 0.0])
        }

        # Mapped node IDs
        map_new_node_ids = {2: 5, 3: 6}

        # Nodes from stable parts
        nodes_stable_parts = [3, 2]

        with pytest.raises(ValueError, match="Element type TRIANGLE_3N is not supported, for interface elements."):
            model._Model__create_interface_elements(test_nodes, "TRIANGLE_3N", nodes_stable_parts, map_new_node_ids)

    def test_create_interface_elements(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test creating interface elements from nodes

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]

        # Set up test data
        test_nodes = {
            3: Node(3, [1.0, 0.0, 0.0]),
            5: Node(5, [1.0, 0.0, 0.0]),  # Mapped from node 2
            6: Node(6, [1.0, 1.0, 0.0]),  # Mapped from node 3
            2: Node(2, [1.0, 1.0, 0.0])
        }

        # Nodes from stable parts
        nodes_stable_parts = [3, 2]

        # Mapped node IDs
        map_new_node_ids = {2: 5, 3: 6}

        # the element nodes are arleady changed in the changing part
        model.body_model_parts[1].mesh.nodes[5] = Node(5, [1.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[6] = Node(6, [1.0, 1.0, 0.0])
        model.body_model_parts[1].mesh.nodes.pop(2)
        model.body_model_parts[1].mesh.nodes.pop(3)
        # also update the elements in the changing part
        model.body_model_parts[1].mesh.elements[2] = Element(2, "TRIANGLE_3N", [5, 6, 4])

        # Test creating interface elements
        interface_elements = model._Model__create_interface_elements(test_nodes, "QUADRANGLE_4N", nodes_stable_parts,
                                                                     map_new_node_ids)

        # Verify an element was created
        assert len(interface_elements) == 1

        # Get the created element
        element_id = list(interface_elements.keys())[0]
        created_element = interface_elements[element_id]

        # Verify element properties
        assert created_element.element_type == "QUADRANGLE_4N"

        # Verify node order (should follow stable-changing-stable-changing pattern)
        node_ids = created_element.node_ids
        assert node_ids == [2, 3, 5, 6]

    def test_create_interface_elements_skips_part_without_mesh(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test that __create_interface_elements correctly skips a body model part
        if its mesh is None.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]

        # Create a new body model part without a mesh and add it to the model
        part_without_mesh = BodyModelPart("no_mesh_part")
        part_without_mesh.mesh = None
        model.body_model_parts.append(part_without_mesh)

        # Set up test data similar to test_create_interface_elements
        test_nodes = {
            3: Node(3, [1.0, 0.0, 0.0]),
            5: Node(5, [1.0, 0.0, 0.0]),
            6: Node(6, [1.0, 1.0, 0.0]),
            2: Node(2, [1.0, 1.0, 0.0])
        }
        nodes_stable_parts = {2, 3}
        map_new_node_ids = {2: 5, 3: 6}

        # Modify the changing part to have the correct nodes and elements for the interface
        changing_part = model.body_model_parts[1]
        changing_part.mesh.nodes[5] = Node(5, [1.0, 0.0, 0.0])
        changing_part.mesh.nodes[6] = Node(6, [1.0, 1.0, 0.0])
        changing_part.mesh.nodes.pop(2)
        changing_part.mesh.nodes.pop(3)
        changing_part.mesh.elements[2] = Element(2, "TRIANGLE_3N", [5, 6, 4])

        # Call the method. It should run without error, skipping the part with no mesh.
        interface_elements = model._Model__create_interface_elements(test_nodes, "QUADRANGLE_4N", nodes_stable_parts,
                                                                     map_new_node_ids)

        # Verify that an interface element was still created from the valid parts
        assert len(interface_elements) == 1
        created_element = list(interface_elements.values())[0]
        assert created_element.element_type == "QUADRANGLE_4N"
        assert created_element.node_ids == [2, 3, 5, 6]

    def test_create_interface_elements_3d(self, model_setup_3d_with_interface: Dict[str, Any]):
        """
        Test creating interface elements from nodes for 3D model.

        Args:
            - model_setup_3d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data for 3D.
        """
        model = model_setup_3d_with_interface["model"]

        # Set up test data
        test_nodes = {
            1: Node(id=1, coordinates=[0.0, 0.0, 0.0]),
            2: Node(id=2, coordinates=[1.0, 0.0, 0.0]),
            3: Node(id=3, coordinates=[0.0, 1.0, 0.0]),
            6: Node(id=6, coordinates=[0.0, 0.0, 0.0]),
            7: Node(id=7, coordinates=[1.0, 0.0, 0.0]),
            8: Node(id=8, coordinates=[0.0, 1.0, 0.0])
        }

        # Mapped node IDs
        map_new_node_ids = {1: 6, 2: 7, 3: 8}

        # Nodes from stable parts
        nodes_stable_parts = [1, 2, 3, 4]

        # the element nodes are already changed in the changing part
        model.body_model_parts[1].mesh.nodes[6] = Node(6, [0.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[7] = Node(7, [1.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[8] = Node(8, [0.0, 1.0, 0.0])
        model.body_model_parts[1].mesh.nodes.pop(1)
        model.body_model_parts[1].mesh.nodes.pop(2)
        model.body_model_parts[1].mesh.nodes.pop(3)
        # also update the elements in the changing part
        model.body_model_parts[1].mesh.elements[2] = Element(2, "TETRAHEDRON_4N", [6, 7, 8, 5])

        # Test creating interface elements
        interface_elements = model._Model__create_interface_elements(test_nodes, "PRISM_6N", nodes_stable_parts,
                                                                     map_new_node_ids)

        # Verify an element was created
        assert len(interface_elements) == 1
        # Compare the created element with expected properties
        assert list(interface_elements.keys())[0] == 3
        assert interface_elements[3].element_type == "PRISM_6N"
        assert interface_elements[3].node_ids == [1, 2, 3, 6, 7, 8]
        assert interface_elements[3].id == 3

    def test_create_interface_body_model_part_2d(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test creating an interface body model part

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]

        # Nodes from stable parts
        nodes_stable_parts = [3, 2]
        common_nodes = [2, 3]
        map_new_node_ids = {2: 6, 3: 5}
        nodes_stable_parts = [1, 2, 3]
        n_interface_nodes = 4
        element_type_gmsh = "QUADRANGLE_4N"
        # add the new nodes to the model
        model.body_model_parts[1].mesh.nodes[5] = Node(5, [1.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[6] = Node(6, [1.0, 1.0, 0.0])
        # pop the nodes from the changing part
        model.body_model_parts[1].mesh.nodes.pop(2)
        model.body_model_parts[1].mesh.nodes.pop(3)
        # also update the elements in the changing part
        model.body_model_parts[1].mesh.elements[2] = Element(2, "TRIANGLE_3N", [5, 6, 4])

        # Create interface body model part
        interface_part = model._Model__create_interface_body_model_part("test_interface",
                                                                        model_2d_with_interface["interface_material"],
                                                                        common_nodes, map_new_node_ids,
                                                                        element_type_gmsh, nodes_stable_parts)

        # Verify basic properties
        assert interface_part.name == "test_interface"
        assert interface_part.material == model_2d_with_interface["interface_material"]

        # Verify nodes in the mesh
        mesh_nodes = interface_part.mesh.nodes
        assert len(mesh_nodes) == 4  # 2 original + 2 mapped nodes
        assert [2, 3, 6, 5] == list(mesh_nodes.keys())

        # Verify elements were created
        assert len(interface_part.mesh.elements) > 0

    def test_create_interface_body_model_part_2d_update(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test creating an interface body model part when the element type already exists in mesh_data.
        This specifically targets the .update() call for elements in gmsh_io.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]

        # Nodes from stable parts
        nodes_stable_parts = [3, 2]
        common_nodes = [2, 3]
        map_new_node_ids = {2: 6, 3: 5}
        nodes_stable_parts = [1, 2, 3]
        n_interface_nodes = 4
        element_type_gmsh = "QUADRANGLE_4N"
        # add the new nodes to the model
        model.body_model_parts[1].mesh.nodes[5] = Node(5, [1.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[6] = Node(6, [1.0, 1.0, 0.0])
        # pop the nodes from the changing part
        model.body_model_parts[1].mesh.nodes.pop(2)
        model.body_model_parts[1].mesh.nodes.pop(3)
        # also update the elements in the changing part
        model.body_model_parts[1].mesh.elements[2] = Element(2, "TRIANGLE_3N", [5, 6, 4])

        # Pre-populate mesh_data with the element type and a dummy element to ensure the 'else' block is hit
        dummy_element_id = 999
        model.gmsh_io.mesh_data["elements"][element_type_gmsh] = {dummy_element_id: [9, 9, 9, 9]}

        # Create interface body model part
        interface_part = model._Model__create_interface_body_model_part("test_interface",
                                                                        model_2d_with_interface["interface_material"],
                                                                        common_nodes, map_new_node_ids,
                                                                        element_type_gmsh, nodes_stable_parts)

        # Verify that the new interface elements were added and the dummy element is still there
        gmsh_elements = model.gmsh_io.mesh_data["elements"][element_type_gmsh]
        assert dummy_element_id in gmsh_elements
        assert len(gmsh_elements) > 1  # Dummy element + new interface elements

        # Verify basic properties
        assert interface_part.name == "test_interface"
        assert interface_part.material == model_2d_with_interface["interface_material"]

        # Verify nodes in the mesh
        mesh_nodes = interface_part.mesh.nodes
        assert len(mesh_nodes) == 4  # 2 original + 2 mapped nodes
        assert [2, 3, 6, 5] == list(mesh_nodes.keys())

        # Verify elements were created
        assert len(interface_part.mesh.elements) > 0

    def test_create_interface_body_model_part_3d(self, model_setup_3d_with_interface: Dict[str, Any]):
        """
        Test creating an interface body model part

        Args:
            - model_setup_3d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data for 3D.
        """

        model = model_setup_3d_with_interface["model"]

        # Nodes from stable parts
        nodes_stable_parts = [1, 2, 3, 4]
        common_nodes = [1, 2, 3]
        map_new_node_ids = {1: 6, 2: 7, 3: 8}
        n_interface_nodes = 6
        element_type_gmsh = "PRISM_6N"
        # add the new nodes to the model
        model.body_model_parts[1].mesh.nodes[6] = Node(6, [0.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[7] = Node(7, [1.0, 0.0, 0.0])
        model.body_model_parts[1].mesh.nodes[8] = Node(8, [0.0, 1.0, 0.0])
        # pop the nodes from the changing part
        model.body_model_parts[1].mesh.nodes.pop(1)
        model.body_model_parts[1].mesh.nodes.pop(2)
        model.body_model_parts[1].mesh.nodes.pop(3)
        # also update the elements in the changing part
        model.body_model_parts[1].mesh.elements[2] = Element(2, "TETRAHEDRON_4N", [6, 7, 8, 5])

        # Create interface body model part
        interface_part = model._Model__create_interface_body_model_part(
            "test_interface", model_setup_3d_with_interface["interface_material"], common_nodes, map_new_node_ids,
            element_type_gmsh, nodes_stable_parts)

        # Verify basic properties
        assert interface_part.name == "test_interface"
        assert interface_part.material == model_setup_3d_with_interface["interface_material"]

        # Verify nodes in the mesh
        mesh_nodes = interface_part.mesh.nodes
        assert len(mesh_nodes) == 6

    def test_adjust_interface_elements_3d(self, model_setup_3d_with_interface: Dict[str, Any]):
        """
        Test the full interface element adjustment process for 3D models

        Args:
            - model_setup_3d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_setup_3d_with_interface["model"]
        model.interfaces["interface_part_1_part_2"] = {
            "interface_part_1": [model.body_model_parts[0]],
            "interface_part_2": [model.body_model_parts[1]],
            "material": model_setup_3d_with_interface["interface_material"],
            "connected_process_definition": {}
        }
        # Get original state
        original_element_count = sum(len(part.mesh.elements) for part in model.body_model_parts)
        original_part_count = len(model.body_model_parts)

        # Run the adjustment method
        model._Model__adjust_interface_elements()
        # Verify a new body model part was added
        assert len(model.body_model_parts) == original_part_count + 1
        # Check the interface part
        interface_part = model.body_model_parts[-1]
        assert interface_part.name == "interface_part_1_part_2"
        assert interface_part.material.name == "interface"
        # Verify node mapping in changing part
        changing_part = model_setup_3d_with_interface["changing_part"]
        assert 1 not in changing_part.mesh.nodes
        assert 2 not in changing_part.mesh.nodes
        assert 3 not in changing_part.mesh.nodes

        # Verify elements in interface part were created
        assert len(interface_part.mesh.elements) > 0

        # Verify total element count increased
        new_element_count = sum(len(part.mesh.elements) for part in model.body_model_parts)
        assert new_element_count > original_element_count

    def test_adjust_interface_elements(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test the full interface element adjustment process

        Args:
            - model_setup (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        model.interfaces["interface_part_1_part_2"] = {
            "interface_part_1": [model.body_model_parts[0]],
            "interface_part_2": [model.body_model_parts[1]],
            "material": model_2d_with_interface["interface_material"],
            "connected_process_definition": {}
        }

        # Get original state
        original_node_count = sum(len(part.mesh.nodes) for part in model.body_model_parts)
        original_element_count = sum(len(part.mesh.elements) for part in model.body_model_parts)
        original_part_count = len(model.body_model_parts)

        # Run the adjustment method
        model._Model__adjust_interface_elements()

        # Verify a new body model part was added
        assert len(model.body_model_parts) == original_part_count + 1

        # Check the interface part
        interface_part = model.body_model_parts[-1]
        assert interface_part.name == "interface_part_1_part_2"
        assert interface_part.material.name == "interface"

        # Verify node mapping in changing part
        changing_part = model.body_model_parts[1]
        assert 2 not in changing_part.mesh.nodes
        assert 3 not in changing_part.mesh.nodes

        # Verify elements in interface part were created
        assert len(interface_part.mesh.elements) > 0

        # Verify total element count increased
        new_element_count = sum(len(part.mesh.elements) for part in model.body_model_parts)
        assert new_element_count > original_element_count

    def test_update_changing_parts_without_mesh(self, model_2d_with_interface: Dict[str, Any]):
        """

        Test updating changing parts with new node IDs, but without a mesh in the changing part.
        Error should be raised.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        # remove mesh from changing part
        model.body_model_parts[1].mesh = None
        changing_part = model_2d_with_interface["changing_part"]

        # Create common nodes set (nodes 2 and 3)
        common_nodes = {2, 3}

        # Create mapping
        map_new_node_ids = {2: 100, 3: 101}

        # Get index of changing part
        indexes_changing_parts = [1]  # Index 1 in model.body_model_parts

        # check that error is raised
        with pytest.raises(ValueError, match="Part `changing_part` has no mesh. Please generate the mesh first."):
            model._Model__update_changing_parts([changing_part], indexes_changing_parts, common_nodes, map_new_node_ids,
                                                {})

    def test_set_interface_success(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test setting an interface between two valid model parts.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.

        """
        model = model_2d_with_interface["model"]
        part_1_name = ["interface_part_1"]
        part_2_name = ["interface_part_2"]
        material = model_2d_with_interface["interface_material"]

        # Mock the model parts
        model.get_model_part_by_name = (lambda name: name if name in part_1_name + part_2_name else None)

        # Call the method
        model.set_interface_between_model_parts(part_1_name, part_2_name, material, {})

        # Verify the interface was set
        interface_name = "interface_interface_part_1_interface_part_2"
        assert interface_name in model.interfaces
        assert model.interfaces[interface_name]["interface_part_1"] == part_1_name
        assert model.interfaces[interface_name]["interface_part_2"] == part_2_name
        assert model.interfaces[interface_name]["material"] == material

    def test_set_interface_part_1_not_found(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test setting an interface raises ValueError when part_1_name is not found.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        part_1_name = ["nonexistent_part_1"]
        part_2_name = ["part_2"]
        material = model_2d_with_interface["interface_material"]

        # Mock the model parts
        model.get_model_part_by_name = (lambda name: name if name in part_2_name else None)

        # Verify the error is raised
        with pytest.raises(
                ValueError,
                match="One or more model parts for the interface are not found. Please check the model part names."):
            model.set_interface_between_model_parts(part_1_name, part_2_name, material, {})

    def test_set_interface_part_2_not_found(self, model_2d_with_interface: Dict[str, Any]):
        """
        Test setting an interface raises ValueError when part_2_name is not found.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        model = model_2d_with_interface["model"]
        part_1_name = ["part_1"]
        part_2_name = ["nonexistent_part_2"]
        material = model_2d_with_interface["interface_material"]

        # Mock the model parts
        model.get_model_part_by_name = (lambda name: name if name in part_1_name else None)

        # Verify the error is raised
        with pytest.raises(
                ValueError,
                match="One or more model parts for the interface are not found. Please check the model part names.",
        ):
            model.set_interface_between_model_parts(part_1_name, part_2_name, material, {})

    def test_update_process_model_parts_applied_both_parts(self, model_setup_large_2d: Model):
        """
        Test updating process model parts with new node IDs. This test checks that the
        method correctly updates the process model parts. The process model part is applied to both parts,
        so both parts should be updated with the new node IDs.

        Args:
            - model_setup_large_2d (:class:`stem.model.Model`): Model instance set up for testing.
        """
        model = model_setup_large_2d
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 8, 3: 10, 5: 9}

        # let's add a line load in the process model part
        line_load = LineLoad(value=[1, 0], active=[True, True])
        load_coordinates = [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)]  # Coordinates for the load
        # Add the load to the model
        model.add_load_by_coordinates(load_coordinates, line_load, "load")
        # add the mesh manually to the process model part
        nodes = {
            1: Node(1, [0.0, 0.0, 0.0]),
            2: Node(2, [0.0, 1.0, 0.0]),  # This will be updated to 8
            7: Node(7, [0.0, 2.0, 0.0])
        }
        elements = {1: Element(1, "LINE_2N", [2, 1]), 2: Element(2, "LINE_2N", [2, 7])}
        # Create a dummy mesh and assign nodes and elements
        mesh = Mesh(ndim=2)
        mesh.nodes = nodes
        mesh.elements = elements
        model.process_model_parts[0].mesh = mesh
        model.gmsh_io.mesh_data["elements"]["LINE_2N"] = {1: [2, 1], 2: [2, 7]}
        # also in the physical groups
        model.gmsh_io.mesh_data["physical_groups"][model.process_model_parts[0].name] = {
            "node_ids": list(nodes.keys()),
            "element_ids": list(elements.keys()),
            "ndim": 2,
            "element_type": "LINE_2N"
        }

        # Prepare a mapping to update node id 2 to 10.
        map_new_node_ids = {2: 8}
        # define the dictionary of connections
        connections = {"load": [True, True]}
        # Call the private update method using name mangling.
        model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, model.body_model_parts[1],
                                                                connections)

        # Retrieve the updated process model part.
        updated_mp = model.process_model_parts[0]
        updated_nodes = updated_mp.mesh.nodes
        updated_element = updated_mp.mesh.elements

        # Check that the node with id 2 is updated to 10
        assert list(updated_nodes.keys()) == [1, 8, 7, 2]
        assert updated_mp.mesh.elements[1].node_ids == [2, 1]
        assert updated_mp.mesh.elements[2].node_ids == [8, 7]

    def test_update_process_model_parts_applied_both_parts_3d(self, model_setup_large_3d_custom: Model):
        """
        Test updating process model parts with new node IDs. This test checks that the
        method correctly updates the process model parts. The process model part is applied to both parts,
        so both parts should be updated with the new node IDs.

        Args:
            - model_setup_large_3d_custom (:class:`stem.model.Model`): Model instance set up for testing.
        """
        model = model_setup_large_3d_custom

        # let's add a surface load in the process model part
        surface_load = SurfaceLoad(value=[1, 0], active=[True, True, True])
        load_coordinates = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 0.0, 0.0]]

        # Add the load to the model
        model.add_load_by_coordinates(load_coordinates, surface_load, "load")
        # add the mesh manually to the process model part
        nodes = {
            3: Node(3, [1.0, 1.0, 1.0]),
            7: Node(7, [2.0, 1.0, 1.0]),
            10: Node(10, [1.0, 0.0, 1.0]),
            12: Node(12, [2.0, 0.0, 1.0]),
            24: Node(24, [1.5, 0.5, 1.0])
        }

        elements = {
            20: Element(20, 'TRIANGLE_3N', [3, 24, 7]),
            21: Element(21, 'TRIANGLE_3N', [10, 24, 3]),
            22: Element(22, 'TRIANGLE_3N', [7, 24, 12]),
            23: Element(23, 'TRIANGLE_3N', [12, 24, 10])
        }

        # Create a dummy mesh and assign nodes and elements
        mesh = Mesh(ndim=2)
        mesh.nodes = nodes
        mesh.elements = elements
        model.process_model_parts[0].mesh = mesh
        model.gmsh_io.mesh_data["elements"]["TRIANGLE_3N"] = {
            21: [10, 24, 3],
            22: [7, 24, 12],
            23: [12, 24, 10],
            20: [3, 24, 7]
        }
        # also in the physical groups
        model.gmsh_io.mesh_data["physical_groups"][model.process_model_parts[0].name] = {
            "node_ids": list(nodes.keys()),
            "element_ids": list(elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }

        # Prepare a mapping to update node id 2 to 10.
        map_new_node_ids = {2: 8}
        # define the dictionary of connections
        connections = {"load": [True, True]}
        # Call the private update method using name mangling.
        model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, model.body_model_parts[1],
                                                                connections)

        # Retrieve the updated process model part.
        updated_mp = model.process_model_parts[0]
        updated_nodes = updated_mp.mesh.nodes

        # Check that the node with id 2 is updated to 10
        assert list(updated_nodes.keys()) == [3, 7, 10, 12, 24]
        assert updated_mp.mesh.elements[21].node_ids == [10, 24, 3]
        assert updated_mp.mesh.elements[22].node_ids == [7, 24, 12]
        assert updated_mp.mesh.elements[23].node_ids == [12, 24, 10]
        assert updated_mp.mesh.elements[20].node_ids == [3, 24, 7]

    @pytest.mark.parametrize(
        "connections",
        [
            ({
                "load": [True, True]
            }),
            ({
                "load": [False, True]
            }),
        ],
    )
    def test_update_process_model_3d_parts_applied_both_part_1(self, connections: Dict[str, List[bool]],
                                                               model_setup_large_3d_custom: Model):
        """
        Test updating process model parts with new node IDs. This test checks that the
        method correctly updates the process model parts. The process model part is applied to part 1 only,
        so only part 2 should be updated with the new node IDs. We use a parameterized test to check
        different connection scenarios. These are:

        - [True, True]: load is applied to both parts, however, the process model part is only applied to part 1 because
        there are no elements in part 1 that are connected to the process model part.
        - [False, True]: load is applied to part 2 only, so the process model part is only applied to part 1.

        Args:
            - connections (Dict[str, List[bool]]): Dictionary containing the connections for the loads.
            - model_setup_large_3d_custom (:class:`stem.model.Model`): Model instance set up for testing.
        """
        model = model_setup_large_3d_custom
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 27, 3: 28, 6: 29, 7: 30, 14: 31, 17: 32}

        # let's add a surface load in the process model part
        surface_load = SurfaceLoad(value=[1, 0], active=[True, True, True])
        load_coordinates = [[2.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0,
                                                                                1.0]]  # Coordinates for the load
        # Add the load to the model
        model.add_load_by_coordinates(load_coordinates, surface_load, "load")
        # add the mesh manually to the process model part
        nodes = {
            3: Node(3, [1.0, 1.0, 1.0]),
            7: Node(7, [2.0, 1.0, 1.0]),
            10: Node(10, [1.0, 0.0, 1.0]),
            12: Node(12, [2.0, 0.0, 1.0]),
            24: Node(24, [1.5, 0.5, 1.0])
        }
        elements = {
            21: Element(21, "TRIANGLE_3N", [10, 24, 3]),
            22: Element(22, "TRIANGLE_3N", [7, 24, 12]),
            23: Element(23, "TRIANGLE_3N", [12, 24, 10])
        }
        # Create a dummy mesh and assign nodes and elements
        mesh = Mesh(ndim=2)
        mesh.nodes = nodes
        mesh.elements = elements
        model.process_model_parts[0].mesh = mesh
        model.gmsh_io.mesh_data["elements"]["TRIANGLE_3N"] = {21: [10, 24, 3], 22: [7, 24, 12], 23: [12, 24, 10]}
        # also in the physical groups
        model.gmsh_io.mesh_data["physical_groups"][model.process_model_parts[0].name] = {
            "node_ids": list(nodes.keys()),
            "element_ids": list(elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }

        # Prepare a mapping to update node id 2 to 10.
        map_new_node_ids = {2: 8}
        # define the dictionary of connections
        connections = {"load": [True, True]}
        # Call the private update method using name mangling.
        model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, model.body_model_parts[1],
                                                                connections)

        # Retrieve the updated process model part.
        updated_mp = model.process_model_parts[0]
        updated_nodes = updated_mp.mesh.nodes
        updated_element = updated_mp.mesh.elements

        # Check that the node with id 2 is updated to 10
        assert list(updated_nodes.keys()) == [3, 7, 10, 12, 24]
        assert updated_mp.mesh.elements[21].node_ids == [10, 24, 3]
        assert updated_mp.mesh.elements[22].node_ids == [7, 24, 12]
        assert updated_mp.mesh.elements[23].node_ids == [12, 24, 10]

    @pytest.mark.parametrize(
        "connections",
        [
            ({
                "load": [True, True]
            }),
            ({
                "load": [False, True]
            }),
        ],
    )
    def test_update_process_model_3d_parts_applied_both_part_2(self, connections: Dict[str, List[bool]],
                                                               model_setup_large_3d_custom: Model):
        """
        Test updating process model parts with new node IDs. This test checks that the
        method correctly updates the process model parts. The process model part is applied to part 2 only,
        so only part 2 should be updated with the new node IDs. We use a parameterized test to check
        different connection scenarios. These are:

        - [True, True]: load is applied to both parts, however, the process model part is only applied to part 2 because
        there are no elements in part 1 that are connected to the process model part.
        - [False, True]: load is applied to part 2 only, so the process model part is only applied to part 2.

        Args:
            - connections (Dict[str, List[bool]]): Dictionary containing the connections for the loads.
            - model_setup_large_3d_custom (:class:`stem.model.Model`): Model instance set up for testing.
        """
        model = model_setup_large_3d_custom
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 27, 3: 28, 6: 29, 7: 30, 14: 31, 17: 32}

        # let's add a surface load in the process model part
        surface_load = SurfaceLoad(value=[1, 0], active=[True, True, True])
        load_coordinates = [[2.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0,
                                                                                1.0]]  # Coordinates for the load
        # Add the load to the model
        model.add_load_by_coordinates(load_coordinates, surface_load, "load")
        # add the mesh manually to the process model part
        nodes = {
            3: Node(3, [1.0, 1.0, 1.0]),
            4: Node(4, [1.0, 2.0, 1.0]),
            7: Node(7, [2.0, 1.0, 1.0]),
            8: Node(8, [2.0, 2.0, 1.0]),
            18: Node(18, [1.5, 1.5, 1.0])
        }
        elements = {
            3: Element(3, "TRIANGLE_3N", [3, 18, 4]),
            4: Element(4, "TRIANGLE_3N", [7, 18, 3]),
            5: Element(5, "TRIANGLE_3N", [4, 18, 8]),
            6: Element(6, "TRIANGLE_3N", [8, 18, 7])
        }
        # Create a dummy mesh and assign nodes and elements
        mesh = Mesh(ndim=2)
        mesh.nodes = nodes
        mesh.elements = elements
        model.process_model_parts[0].mesh = mesh
        model.gmsh_io.mesh_data["elements"]["TRIANGLE_3N"] = {
            3: [3, 18, 4],
            4: [7, 18, 3],
            5: [4, 18, 8],
            6: [8, 18, 7]
        }
        # also in the physical groups
        model.gmsh_io.mesh_data["physical_groups"][model.process_model_parts[0].name] = {
            "node_ids": list(nodes.keys()),
            "element_ids": list(elements.keys()),
            "ndim": 2,
            "element_type": "TRIANGLE_3N"
        }

        # Prepare a mapping to update node id 2 to 10.
        map_new_node_ids = {2: 8}
        # define the dictionary of connections
        connections = {"load": [True, True]}
        # Call the private update method using name mangling.
        model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, model.body_model_parts[1],
                                                                connections)

        # Retrieve the updated process model part.
        updated_mp = model.process_model_parts[0]
        updated_nodes = updated_mp.mesh.nodes
        updated_element = updated_mp.mesh.elements

        # Check that the node with id 2 is updated to 10
        assert list(updated_nodes.keys()) == [3, 4, 7, 8, 18]
        assert updated_mp.mesh.elements[3].node_ids == [3, 18, 4]
        assert updated_mp.mesh.elements[4].node_ids == [7, 18, 3]
        assert updated_mp.mesh.elements[5].node_ids == [4, 18, 8]

    @pytest.mark.parametrize(
        "connections",
        [
            ({
                "load": [True, True]
            }),
            ({
                "load": [True, False]
            }),
        ],
    )
    def test_update_process_model_parts_applied_part_1(self, connections: Dict[str, List[bool]],
                                                       model_setup_large_2d: Model):
        """
        Test updating process model parts with new node IDs. This test checks that the
        method correctly updates the process model parts. The process model part is applied to part 1 only,
        so only part 1 should be updated with the new node IDs. We use a parameterized test to check
        different connection scenarios. These are:

        - [True, True]: load is applied to both parts, however, the process model part is only applied to part 1 because
        there are no elements in part 2 that are connected to the process model part.
        - [True, False]: load is applied to part 1 only, so the process model part is only applied to part 1.

        Args:
            - connections (Dict[str, List[bool]]): Dictionary containing the connections for the loads.
            - model_setup_large_2d (:class:`stem.model.Model`): Model instance set up for testing.
        """
        model = model_setup_large_2d
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 8, 3: 10, 5: 9}

        # let's add a point load in the process model part
        line_load = LineLoad(value=[1, 0], active=[True, True])
        load_coordinates = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]  # Coordinates for the load
        # Add the load to the model
        model.add_load_by_coordinates(load_coordinates, line_load, "load")
        # add the mesh manually to the process model part
        nodes = {
            1: Node(1, [0.0, 0.0, 0.0]),
            2: Node(2, [0.0, 1.0, 0.0])  # This will be updated to 8
        }
        elements = {1: Element(1, "LINE_2N", [2, 1])}
        # Create a dummy mesh and assign nodes and elements
        mesh = Mesh(ndim=2)
        mesh.nodes = nodes
        mesh.elements = elements
        model.process_model_parts[0].mesh = mesh
        model.gmsh_io.mesh_data["elements"]["LINE_2N"] = {1: [2, 1]}
        # also in the physical groups
        model.gmsh_io.mesh_data["physical_groups"][model.process_model_parts[0].name] = {
            "node_ids": list(nodes.keys()),
            "element_ids": list(elements.keys()),
            "ndim": 2,
            "element_type": "LINE_2N"
        }

        # Prepare a mapping to update node id 2 to 10.
        map_new_node_ids = {2: 8}
        # Call the private update method using name mangling.
        model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, model.body_model_parts[1],
                                                                connections)

        # Retrieve the updated process model part.
        updated_mp = model.process_model_parts[0]
        updated_nodes = updated_mp.mesh.nodes
        updated_element = updated_mp.mesh.elements

        # Check that the node with id 2 is updated to 10
        assert list(updated_nodes.keys()) == [1, 2]
        assert updated_mp.mesh.elements[1].node_ids == [2, 1]

    @pytest.mark.parametrize(
        "connections",
        [
            ({
                "load": [True, True]
            }),
            ({
                "load": [False, True]
            }),
        ],
    )
    def test_update_process_model_parts_applied_part_2(self, connections: Dict[str, List[bool]],
                                                       model_setup_large_2d: Model):
        """
        Test updating process model parts with new node IDs. This test checks that the
        method correctly updates the process model parts. The process model part is applied to part 2 only,
        so only part 2 should be updated with the new node IDs. We use a parameterized test to check
        different connection scenarios. These are:

        - [True, True]: load is applied to both parts, however, the process model part is only applied to part 2 because
        there are no elements in part 1 that are connected to the process model part.
        - [False, True]: load is applied to part 2 only, so the process model part is only applied to part 2.

        Args:
            - connections (Dict[str, List[bool]]): Dictionary containing the connections for the loads.
            - model_setup_large_2d (:class:`stem.model.Model`): Model instance set up for testing.
        """
        model = model_setup_large_2d
        # Create a mapping for node IDs 2 and 3 (common nodes)
        map_new_node_ids = {2: 8, 3: 10, 5: 9}

        # let's add a point load in the process model part
        line_load = LineLoad(value=[1, 0], active=[True, True])
        load_coordinates = [(0.0, 2.0, 0.0), (0.0, 1.0, 0.0)]  # Coordinates for the load
        # Add the load to the model
        model.add_load_by_coordinates(load_coordinates, line_load, "load")
        # add the mesh manually to the process model part
        nodes = {
            7: Node(1, [0.0, 2.0, 0.0]),
            2: Node(2, [0.0, 1.0, 0.0])  # This will be updated to 8
        }
        elements = {1: Element(1, "LINE_2N", [2, 7])}
        # Create a dummy mesh and assign nodes and elements
        mesh = Mesh(ndim=2)
        mesh.nodes = nodes
        mesh.elements = elements
        model.process_model_parts[0].mesh = mesh
        model.gmsh_io.mesh_data["elements"]["LINE_2N"] = {1: [2, 7]}
        # also in the physical groups
        model.gmsh_io.mesh_data["physical_groups"][model.process_model_parts[0].name] = {
            "node_ids": list(nodes.keys()),
            "element_ids": list(elements.keys()),
            "ndim": 2,
            "element_type": "LINE_2N"
        }

        # Prepare a mapping to update node id 2 to 10.
        map_new_node_ids = {2: 8}
        # Call the private update method using name mangling.
        model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, model.body_model_parts[1],
                                                                connections)

        # Retrieve the updated process model part.
        updated_mp = model.process_model_parts[0]
        updated_nodes = updated_mp.mesh.nodes
        updated_element = updated_mp.mesh.elements

        # Check that the node with id 2 is updated to 10
        assert list(updated_nodes.keys()) == [7, 8]
        assert updated_mp.mesh.elements[1].node_ids == [8, 7]

    def test_update_process_model_parts_raises_error_if_no_mesh_process_part(self):
        """
        Test that updating process model parts raises ValueError if the mesh is None.
        """
        # Create a model instance
        model = Model(ndim=2)
        # Create a process model part that does not have a mesh (mesh is None)
        mp = ModelPart("no_mesh")
        mp.mesh = None
        model.process_model_parts.append(mp)

        # Prepare a mapping (can be arbitrary)
        map_new_node_ids = {1: 100}
        with pytest.raises(ValueError,
                           match="Process model part `no_mesh` has no mesh. Please generate the mesh first."):
            model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, None, {})

    def test_update_process_model_parts_raises_error_if_no_mesh_body_model_part(self,
                                                                                model_2d_with_interface: Dict[str,
                                                                                                              Any]):
        """
        Test that updating process model parts raises ValueError if the mesh is None.

        Args:
            - model_2d_with_interface (Dict[str, Any]): Dictionary containing the model and other test data.
        """
        # Create a model instance
        model = model_2d_with_interface["model"]

        # Prepare a mapping (can be arbitrary)
        map_new_node_ids = {1: 100}

        updating_body_model_part = model.body_model_parts[1]
        # remove mesh from the body model part
        updating_body_model_part.mesh = None

        # create a process model part that has a mesh
        process_model_part = ModelPart("process_model_part_with_mesh")
        process_model_part.mesh = model.body_model_parts[0].mesh

        # add the process model part to the model
        model.process_model_parts.append(process_model_part)

        with pytest.raises(
                ValueError,
                match="Updating body model part `changing_part` has no mesh. Please generate the mesh first."):
            model._Model__update_process_model_parts_for_interfaces(map_new_node_ids, updating_body_model_part, {})
