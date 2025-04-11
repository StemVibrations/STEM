import json
import os
from pathlib import Path
from typing import Sequence, Tuple, get_args, Set, Optional, List, Dict, Any, Union

from gmsh_utils import gmsh_IO
import numpy as np

from stem.additional_processes import ParameterFieldParameters, HingeParameters
from stem.field_generator import RandomFieldGenerator
from stem.globals import ELEMENT_DATA, OUT_OF_PLANE_AXIS_2D, VERTICAL_AXIS, GRAVITY_VALUE
from stem.load import *
from stem.boundary import *
from stem.geometry import Geometry, Point
from stem.mesh import Mesh, MeshSettings, Node, Element
from stem.model_part import ModelPart, BodyModelPart, Material, ProcessParameters
from stem.output import Output, OutputParametersABC, JsonOutputParameters
from stem.plot_utils import PlotUtils
from stem.soil_material import *
from stem.solver import Problem, StressInitialisationType
from stem.structural_material import *
from stem.utils import Utils
from stem.water_processes import WaterProcessParametersABC, UniformWaterPressure


class Model:
    """
    A class to represent the main model.

    Attributes:
        - ndim (int): Number of dimensions of the model
        - project_parameters (:class:`stem.solver.Problem): Object containing the problem data and solver settings.
        - geometry (Optional[:class:`stem.geometry.Geometry`]) The geometry of the whole model.
        - body_model_parts (List[:class:`stem.model_part.BodyModelPart`]): A list containing the body model parts.
        - process_model_parts (List[:class:`stem.model_part.ModelPart`]): A list containing the process model parts.
        - output_settings (List[:class:`stem.output.Output`]): A list containing the output settings.
        - extrusion_length (Optional[float]): The extrusion length in the out of plane direction.
        - groups (Dict[str, Any]): A dictionary containing shared information among sets of model parts.

    """

    def __init__(self, ndim: int):
        """
        Constructor of the Model class.

        Args:
            - ndim (int): Number of dimensions of the model
        """
        self.ndim: int = ndim
        self.project_parameters: Optional[Problem] = None
        self.geometry: Optional[Geometry] = None
        self.mesh_settings: MeshSettings = MeshSettings()
        self.gmsh_io = gmsh_IO.GmshIO()
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []
        self.output_settings: List[Output] = []
        self.extrusion_length: Optional[float] = None
        self.groups: Dict[str, Any] = {}

    def __del__(self):
        """
        Destructor of the Model class. Finalizes the gmsh_io instance.

        """
        self.gmsh_io.finalize_gmsh()

    @property
    def all_model_parts(self) -> List[ModelPart]:
        """
        Get all model parts.

        Returns:
            - List[:class:`stem.model_part.ModelPart`]: A list containing all model parts.
        """
        return self.body_model_parts + self.process_model_parts

    def generate_straight_track(self, sleeper_distance: float, n_sleepers: int, rail_parameters: EulerBeam,
                                sleeper_parameters: NodalConcentrated, rail_pad_parameters: ElasticSpringDamper,
                                rail_pad_thickness: float, origin_point: Sequence[float],
                                direction_vector: Sequence[float], name: str):
        """
        Generates a track geometry. With rail, rail-pads and sleepers as mass elements. Sleepers are placed at the
        bottom of the track with a distance of sleeper_distance between them. The sleepers are connected to the rail
        with rail-pads with a thickness of rail_pad_thickness. The track is generated in the direction of the
        direction_vector starting from the origin_point. The track can only move in the vertical direction.

        Args:
            - sleeper_distance (float): distance between sleepers
            - n_sleepers (int): number of sleepers
            - rail_parameters (:class:`stem.structural_material.EulerBeam`): rail parameters
            - sleeper_parameters (:class:`stem.structural_material.NodalConcentrated`): sleeper parameters
            - rail_pad_parameters (:class:`stem.structural_material.ElasticSpringDamper`): rail pad parameters
            - rail_pad_thickness (float): thickness of the rail pad
            - origin_point (Sequence[float]): origin point of the track
            - direction_vector (Sequence[float]): direction vector of the track
            - name (str): name of the track
        """

        rail_name = f"{name}"

        sleeper_name = f"sleeper_{name}"
        rail_pads_name = f"rail_pads_{name}"

        normalized_direction_vector = np.array(direction_vector) / np.linalg.norm(direction_vector)

        # set local rail geometry
        rail_local_distance = np.linspace(0, sleeper_distance * (n_sleepers - 1), n_sleepers)
        sleeper_local_coords = np.copy(rail_local_distance)

        # set global rail geometry
        rail_global_coords = rail_local_distance[:, None].dot(normalized_direction_vector[None, :]) + origin_point
        rail_global_coords[:, VERTICAL_AXIS] += rail_pad_thickness
        rail_geo_settings = {rail_name: {"coordinates": rail_global_coords, "ndim": 1}}

        # set sleepers geometry
        sleeper_global_coords = sleeper_local_coords[:, None].dot(normalized_direction_vector[None, :]) + origin_point
        connection_geo_settings = {"": {"coordinates": sleeper_global_coords, "ndim": 1}}

        sleeper_geo_settings = {sleeper_name: {"coordinates": sleeper_global_coords, "ndim": 0}}

        # firstly create lines for the connection between the track and the foundation

        self.gmsh_io.generate_geometry(connection_geo_settings, "")

        # add the sleepers to the track
        self.gmsh_io.generate_geometry(sleeper_geo_settings, "")

        # add the rail geometry
        self.gmsh_io.generate_geometry(rail_geo_settings, "")

        rail_model_part = BodyModelPart(rail_name)
        rail_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rail_name)
        rail_model_part.material = StructuralMaterial(name=rail_name, material_parameters=rail_parameters)

        sleeper_model_part = BodyModelPart(sleeper_name)
        sleeper_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, sleeper_name)

        # create rail pad geometries
        rail_pad_line_ids_aux = [
            self.gmsh_io.make_geometry_1d((top_coordinates, bot_coordinates))
            for top_coordinates, bot_coordinates in zip(rail_global_coords, sleeper_global_coords)
        ]

        rail_pad_line_ids = [ids[0] for ids in rail_pad_line_ids_aux]

        self.gmsh_io.add_physical_group(rail_pads_name, 1, rail_pad_line_ids)

        # create rail, sleeper, and rail_pad body model parts
        sleeper_model_part.material = StructuralMaterial(name=sleeper_name, material_parameters=sleeper_parameters)

        rail_pads_model_part = BodyModelPart(rail_pads_name)
        rail_pads_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rail_pads_name)
        rail_pads_model_part.material = StructuralMaterial(name=rail_pads_name, material_parameters=rail_pad_parameters)

        # add physical group to gmsh
        rail_constraint_name = f"constraint_{rail_name}"
        rail_constraint_geometry_ids = self.gmsh_io.geo_data["physical_groups"][rail_name]["geometry_ids"]
        self.gmsh_io.add_physical_group(f"constraint_{rail_name}", 1, rail_constraint_geometry_ids)

        # create model part
        constraint_model_part = ModelPart(rail_constraint_name)

        # retrieve geometry from gmsh and add to model part
        constraint_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rail_constraint_name)

        # add displacement_constraint in the non-vertical directions
        constraint_model_part.parameters = DisplacementConstraint(active=[True, True, True],
                                                                  is_fixed=[True, True, True],
                                                                  value=[0, 0, 0])
        constraint_model_part.parameters.is_fixed[VERTICAL_AXIS] = False

        self.body_model_parts.append(rail_model_part)
        self.body_model_parts.append(sleeper_model_part)
        self.body_model_parts.append(rail_pads_model_part)

        self.process_model_parts.append(constraint_model_part)

        # add no rotation constraint at the rail ends for a more realistic boundary in 2D and 3D and to prevent torsion
        # in 3D
        rotation_constraint_name = f"rotation_constraint_{rail_name}"

        no_rotation_model_part = ModelPart(rotation_constraint_name)
        no_rotation_constraint = RotationConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True],
                                                    value=[0, 0, 0])
        no_rotation_model_part.parameters = no_rotation_constraint

        # add constraint geometries to both edges of the rail
        no_rotation_geo_settings = {
            rotation_constraint_name: {
                "coordinates": [rail_global_coords[0], rail_global_coords[-1]],
                "ndim": 0
            }
        }
        self.gmsh_io.generate_geometry(no_rotation_geo_settings, "")

        no_rotation_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rotation_constraint_name)

        self.process_model_parts.append(no_rotation_model_part)

    def generate_extended_straight_track(self, sleeper_distance: float, n_sleepers: int, rail_parameters: EulerBeam,
                                         sleeper_parameters: NodalConcentrated,
                                         rail_pad_parameters: ElasticSpringDamper, rail_pad_thickness: float,
                                         origin_point: Sequence[float], soil_equivalent_parameters: ElasticSpringDamper,
                                         length_soil_equivalent_element: float, direction_vector: Sequence[float],
                                         name: str):
        """
        Generates a track geometry. With rail, rail-pads and sleepers as mass elements. Sleepers are placed at the
        bottom of the track with a distance of sleeper_distance between them. The sleepers are connected to the rail
        with rail-pads with a thickness of rail_pad_thickness. The track is generated in the direction of the
        direction_vector starting from the origin_point. The track can only move in the vertical direction.
        When part of the track is located outside the 2D or 3D soil domain, 1D elements are placed below the sleepers
        which simulate the behaviour of the soil in vertical direction. The bottom of the 1D elements are fixed in all
        directions.

        Args:
            - sleeper_distance (float): distance between sleepers
            - n_sleepers (int): number of sleepers
            - rail_parameters (:class:`stem.structural_material.EulerBeam`): rail parameters
            - sleeper_parameters (:class:`stem.structural_material.NodalConcentrated`): sleeper parameters
            - rail_pad_parameters (:class:`stem.structural_material.ElasticSpringDamper`): rail pad parameters
            - rail_pad_thickness (float): thickness of the rail pad
            - origin_point (Sequence[float]): origin point of the track
            - soil_equivalent_parameters: (:class:`stem.structural_material.ElasticSpringDamper`): soil equivalent
            parameters
            - length_soil_equivalent_element (float): length of the 1D soil equivalent
            - direction_vector (Sequence[float]): direction vector of the track
            - name (str): name of the track
        """
        self.generate_straight_track(sleeper_distance, n_sleepers, rail_parameters, sleeper_parameters,
                                     rail_pad_parameters, rail_pad_thickness, origin_point, direction_vector, name)
        self.__generate_extended_rail_part(soil_equivalent_parameters, name, length_soil_equivalent_element)

    def __generate_extended_rail_part(self, soil_equivalent_parameters: ElasticSpringDamper, name: str,
                                      length_soil_equivalent_element: float):
        """
        Generates the soil equivalent elements outside the 2D or 3D soil domain. The soil equivalent elements are
        spring-damper elements that represents the soil below the rail in vertical direction. The soil equivalent
        elements are connected to the rail with rail-pads. The bottom of the soil equivalent elements are fixed in
        all directions. While the soil equivalent elements can only move in the vertical direction.

        Args:
            - soil_equivalent_parameters: (:class:`stem.structural_material.ElasticSpringDamper`): soil equivalent
            parameters
            - name (str): name of the track
            - length_soil_equivalent_element (float): length of the 1D soil equivalent elements
        """

        soil_equivalent_name = f"soil_equivalent_{name}"
        sleeper_name = f"sleeper_{name}"

        # check which sleepers are outside the soil domain
        points_outside_soil_domain = self.get_points_outside_soil(sleeper_name)
        points_outside_ids = [point.id for point in points_outside_soil_domain]
        points_outside_coords = [point.coordinates for point in points_outside_soil_domain]
        # create bottom points for the soil equivalent
        # set global rail geometry
        soil_equivalent_bottom = np.copy(points_outside_coords)
        soil_equivalent_bottom[:, VERTICAL_AXIS] -= length_soil_equivalent_element

        # create geometries of the soil equivalent lines
        soil_equivalent_lines = [
            self.gmsh_io.make_geometry_1d((top_coordinates, bot_coordinates))
            for top_coordinates, bot_coordinates in zip(points_outside_coords, soil_equivalent_bottom)
        ]

        soil_equivalent_line_ids = [ids[0] for ids in soil_equivalent_lines]

        self.gmsh_io.add_physical_group(soil_equivalent_name, 1, soil_equivalent_line_ids)

        soil_equivalent_part = BodyModelPart(soil_equivalent_name)
        soil_equivalent_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, soil_equivalent_name)
        soil_equivalent_part.material = StructuralMaterial(name=soil_equivalent_name,
                                                           material_parameters=soil_equivalent_parameters)
        self.body_model_parts.append(soil_equivalent_part)
        # add constraint to the soil equivalent as a new model part
        constraint_horizontal_soil_equivalent_name = f"constraint_horizontal_{soil_equivalent_name}"
        # can only move in the vertical direction
        constraint_list = [True, True, True]
        constraint_list[VERTICAL_AXIS] = False
        constraint_parameters = DisplacementConstraint(active=constraint_list,
                                                       is_fixed=constraint_list,
                                                       value=[0, 0, 0])
        self.add_boundary_condition_by_geometry_ids(0, points_outside_ids, constraint_parameters,
                                                    constraint_horizontal_soil_equivalent_name)

        # add bottom points fixed
        constraint_model_soil_equivalent_name = f"constraint_{soil_equivalent_name}"
        constraint_model_soil_equivalent_part = ModelPart(f"constraint_{soil_equivalent_name}")
        constraint_model_soil_equivalent = DisplacementConstraint(active=[True, True, True],
                                                                  is_fixed=[True, True, True],
                                                                  value=[0, 0, 0])
        constraint_model_soil_equivalent_part.parameters = constraint_model_soil_equivalent
        constraint_model_soil_equivalent_part_settings = {
            constraint_model_soil_equivalent_name: {
                "coordinates": soil_equivalent_bottom,
                "ndim": 0
            }
        }
        self.gmsh_io.generate_geometry(constraint_model_soil_equivalent_part_settings, "")

        constraint_model_soil_equivalent_part.get_geometry_from_geo_data(self.gmsh_io.geo_data,
                                                                         constraint_model_soil_equivalent_name)

        self.process_model_parts.append(constraint_model_soil_equivalent_part)

    def get_points_outside_soil(self, model_part_name: str) -> List[Point]:
        """
        Get the points of the model part that are outside the soil model parts.

        Args:
            - model_part_name (str): The name of the model part to check the points

        Raises:
            - ValueError: if the model part is not found.
            - ValueError: if the model part has no geometry.

        Returns:
            - List[int]: The ids of the points that are outside the volume of the model part.
            - List[List[float]]: The coordinates of the points that are outside the volume of the model part.

        """
        # get bbox of the soil model parts
        min_coords, max_coords = self.get_bounding_box_soil()

        model_part = self.get_model_part_by_name(model_part_name)

        if model_part is None:
            raise ValueError(f"Model part {model_part_name} not found.")
        else:
            points_outside_geometry = []
            if model_part.geometry is None:
                raise ValueError(f"Model part {model_part_name} has no geometry.")
            for point_id, point in model_part.geometry.points.items():

                # check if point is within the bounding box of the soil model parts
                x_is_in = min_coords[0] <= point.coordinates[0] <= max_coords[0]
                y_is_in = min_coords[1] <= point.coordinates[1] <= max_coords[1]
                is_inside = x_is_in and y_is_in
                if self.ndim == 3:
                    z_is_in = min_coords[2] <= point.coordinates[2] <= max_coords[2]
                    is_inside = (is_inside and z_is_in)
                if not is_inside:
                    points_outside_geometry.append(point)
            return points_outside_geometry

    def get_bounding_box_soil(self) -> Tuple[List[float], List[float]]:
        """
        Get the bounding box of the soil model parts.

        Raises:
            - ValueError: if the model part has no geometry

        Returns:
            - Tuple[List[float], List[float]]: The minimum and maximum coordinates of the bounding box.
        """
        min_coords = [np.inf, np.inf, np.inf]
        max_coords = [-np.inf, -np.inf, -np.inf]

        for model_part in self.body_model_parts:
            if isinstance(model_part.material, SoilMaterial):
                if model_part.geometry is None:
                    raise ValueError("Model part has no geometry.")
                # Extract all points' coordinates and convert them into a NumPy array
                coordinates = np.array([point.coordinates for point in model_part.geometry.points.values()])
                # Find the minimum and maximum for each axis (x, y, z) across all points
                min_coords = np.min(np.vstack((coordinates, min_coords)), axis=0)
                max_coords = np.max(np.vstack((coordinates, max_coords)), axis=0)

        return min_coords, max_coords

    def add_all_layers_from_geo_file(self, geo_file_name: str, body_names: Sequence[str]):
        """
        Add all physical groups from a geo file to the model. The physical groups with the names in body_names are
        added as body model parts, the other physical groups are added as process model parts.

        Args:
            - geo_file_name (str): name of the geo file
            - body_names (Sequence[str]): names of the physical groups which should be added as body model parts

        """

        # read the geo file and generate the geo_data dictionary
        self.gmsh_io.read_gmsh_geo(geo_file_name)

        # Reset the gmsh instance with the geo data, as read from the geo file
        self.gmsh_io.generate_geo_from_geo_data()

        # Create geometry and model part for each physical group in the gmsh geo_data
        model_part: Union[ModelPart, BodyModelPart]

        for group_name in self.gmsh_io.geo_data["physical_groups"].keys():
            # create model part, if the group name is in the body names, create a body model part, otherwise a process
            # model part
            if group_name in body_names:
                model_part = BodyModelPart(group_name)
            else:
                model_part = ModelPart(group_name)

            # set the name and geometry of the model part
            model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, group_name)

            # add model part to either body model parts or process model part
            if isinstance(model_part, BodyModelPart):
                self.body_model_parts.append(model_part)
            else:
                self.process_model_parts.append(model_part)

    def add_group_for_extrusion(self, group_name: str, reference_depth: float, extrusion_length: float):
        """
        Adds a group for extrusion which consists of a starting coordinate in the out of plane direction a name and the
        the length for the extrusion. The group must be always unique while extrusion length can also be negative.

        Args:
            - group_name (str): The name of the group. Must be unique.
            - reference_depth (float): The reference (starting) depth for the extrusion in the out of plane direction.
            - extrusion_length (float): The length of the group used for the extrusion. It can also be negative

        Raises:
            - ValueError: if the section_name matches an already an existing 3D section.
        """
        if group_name in self.groups.keys():
            raise ValueError(f"The group `{group_name}` already exists, but group names must be unique.")

        direction_vector: List[float] = [0, 0, 0]
        direction_vector[OUT_OF_PLANE_AXIS_2D] = 1

        reference_coordinate: List[float] = [0, 0, 0]
        reference_coordinate[OUT_OF_PLANE_AXIS_2D] = reference_depth

        self.groups[group_name] = {
            "model_part_names": [],
            "extrusion_parameters": {
                "reference_coordinate": reference_coordinate,
                "length": extrusion_length,
                "direction_vector": direction_vector
            }
        }

    def add_model_part_to_group(self, group_name: str, part_name: str):
        """
        Adds a model part name to a pre-existing group for extrusion.

        Args:
            - group_name (str): The name of the group.
            - part_name (str): The name of the model part to be added to the group.

        Raises:
            - ValueError: if the group doesn't exist.
            - ValueError: if the model part doesn't exist.
        """
        if group_name not in self.groups.keys():
            raise ValueError(f"The group specified `{group_name}` does not exist.")

        if self.get_model_part_by_name(part_name) is None:
            raise ValueError(f"The model part specified `{part_name}` does not exist.")

        self.groups[group_name]["model_part_names"].append(part_name)

    def add_soil_layer_by_coordinates(self,
                                      coordinates: Sequence[Sequence[float]],
                                      material_parameters: Union[SoilMaterial, StructuralMaterial],
                                      name: str,
                                      group_name: Optional[str] = None):
        """
        Adds a soil layer to the model by giving a sequence of 3D coordinates.
        The coordinates have to belong to the same plane.
        In a 3D model, the 2D geometry is extruded in the direction of the extrusion group.
        If no extrusion group is provided, the geometry is extruded in the out of plane direction.

        Args:
            - coordinates (Sequence[Sequence[float]]): The plane coordinates of the soil layer.
            - material_parameters (Union[:class:`stem.soil_material.SoilMaterial`, \
                :class:`stem.structural_material.StructuralMaterial`]): The material parameters of the soil layer.
            - name (str): The name of the soil layer.
            - group_name (Optional[str]): The name of the 3D group name for extruding the layer.

        Raises:
            - ValueError: if the polygon of the soil layer is not planar.
            - ValueError: if the model is 3D and the specified group_name doesn't exist.
            - ValueError: if the model is 3D but no group_name nor model.extrusion_length are specified.
            - ValueError: if the model is 3D, a valid group is specified, but the reference point of the group \
                is not in the same plane of the polygon of the soil layer.
        """

        # sort coordinates in anti-clockwise order, such that elements in mesh are also in anti-clockwise order
        if Utils.are_2d_coordinates_clockwise(coordinates):
            coordinates = coordinates[::-1]

        if not Utils.is_polygon_planar(coordinates):
            raise ValueError("Polygon for the soil layer are not on the same plane.")

        # validation of group_name
        if group_name is not None and group_name not in self.groups.keys():
            raise ValueError(f"Non-existent group specified `{group_name}`.")

        gmsh_input = {name: {"coordinates": coordinates, "ndim": self.ndim}}

        # check if extrusion length is specified in 3D
        if self.ndim == 3:

            if self.extrusion_length is None and group_name is None:
                raise ValueError("For 3D models either the extrusion length or the group name for the extrusion must be"
                                 " specified.")

            elif group_name is not None:

                # retrieve information about group
                extrusion_parameters = self.groups[group_name]["extrusion_parameters"]
                # normalise the direction vector and scale it by the extrusion length
                direction_vector = extrusion_parameters["direction_vector"]
                norm = np.linalg.norm(direction_vector)
                extrusion_vector: List[float] = [dv * extrusion_parameters["length"] / norm for dv in direction_vector]
                gmsh_input[name]["extrusion_length"] = extrusion_vector

                reference_point_group = extrusion_parameters["reference_coordinate"]

                if not Utils.is_point_coplanar_to_polygon(reference_point_group, coordinates):
                    raise ValueError(f"The reference coordinate of group: {group_name}, "
                                     f"does not lay on the same plane as soil layer: {name}")

            elif self.extrusion_length is not None:

                extrusion_vector = [0, 0, 0]
                extrusion_vector[OUT_OF_PLANE_AXIS_2D] = self.extrusion_length
                gmsh_input[name]["extrusion_length"] = extrusion_vector

        # todo check if this function in gmsh io can be improved
        self.gmsh_io.generate_geometry(gmsh_input, "")

        # create body model part
        body_model_part = BodyModelPart(name)
        body_model_part.material = material_parameters

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        self.body_model_parts.append(body_model_part)

        # add the model part to the group
        if group_name is not None:
            self.add_model_part_to_group(group_name, part_name=name)

    def add_load_by_geometry_ids(self, geometry_ids: Sequence[int], load_parameters: LoadParametersABC, name: str):
        """
        Add a load to the model by giving the geometry ids of the geometry where the load has to be applied.
        The geometry dimension of the entity where the load needs to be applied is determined based on the
        load_parameters (0=point load, 1=line load, 2=surface load, 3=volume).

        Args:
            - geometry_ids (Sequence[int]): geometry ids of the entities where the load needs to be applied.
            - load_parameters (:class:`stem.load.LoadParametersABC`): load parameters to define the load object.
            - name (str): name of the load.

        Raises:
            - NotImplementedError: when the load parameter provided is not one of point, line, moving, UVEC
            or surface loads.
        """

        # point load can only be assigned to 0d geometry
        if isinstance(load_parameters, PointLoad):
            ndim_load = 0
        # line and moving load can only be assigned to 1d geometry
        elif isinstance(load_parameters, (LineLoad, MovingLoad, UvecLoad)):
            ndim_load = 1
        # surface load can only be assigned to 2d geometry
        elif isinstance(load_parameters, SurfaceLoad):
            ndim_load = 2
        else:
            raise NotImplementedError(
                f"Load parameter provided is not supported: `{load_parameters.__class__.__name__}`.")
        # add physical group to gmsh
        self.gmsh_io.add_physical_group(name, ndim_load, geometry_ids)

        # create model part
        model_part = ModelPart(name)

        # retrieve geometry from gmsh and add to model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        # add load parameters to model part
        model_part.parameters = load_parameters

        model_part.validate_input()

        self.process_model_parts.append(model_part)

    def add_load_by_coordinates(self, coordinates: Sequence[Sequence[float]], load_parameters: LoadParametersABC,
                                name: str):
        """
        Adds a load to the model by giving a sequence of 3D coordinates. For a 2D model, the third coordinate is
        ignored.

        Args:
            - coordinates (Sequence[Sequence[float]]): The coordinates of the load.
            - load_parameters (:class:`stem.load.LoadParametersABC`): The parameters of the load.
            - name (str): The name of the load part.

        Raises:
            - ValueError: if load_parameters is not of one of the classes PointLoad, MovingLoad, LineLoad
                          or SurfaceLoad.

        """

        # validation of inputs
        Utils.validate_coordinates(coordinates)

        # create input for gmsh
        if isinstance(load_parameters, PointLoad):
            gmsh_input = {name: {"coordinates": coordinates, "ndim": 0}}
        elif isinstance(load_parameters, (LineLoad, MovingLoad, UvecLoad)):
            gmsh_input = {name: {"coordinates": coordinates, "ndim": 1}}
        elif isinstance(load_parameters, SurfaceLoad):
            gmsh_input = {name: {"coordinates": coordinates, "ndim": 2}}
        else:
            raise ValueError(f'Invalid load_parameters ({load_parameters.__class__.__name__}) object'
                             f' provided for the load {name}. Expected one of PointLoad, MovingLoad,'
                             f' LineLoad or SurfaceLoad.')

        self.gmsh_io.generate_geometry(gmsh_input, "")

        # create model part
        model_part = ModelPart(name)
        model_part.parameters = load_parameters

        # set the geometry of the model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        # validate the input
        model_part.validate_input()

        self.process_model_parts.append(model_part)

    def add_load_on_line_model_part(self, model_part_name: str, load_parameters: LoadParametersABC, load_name: str):
        """
        Adds a load to the model by giving the name of the line model part where the load has to be applied.
        It only works with LineLoad, MovingLoad and UvecLoad.

        Args:
            - model_part_name (str): name of the line model part where the load needs to be applied.
            - load_parameters (:class:`stem.load.LoadParametersABC`): load parameters to define the load object.
            - load_name (str): name of the load.

        Raises:
            - ValueError: if the model part name is not found.
            - ValueError: if the model part is not a line model part.
            - ValueError: if the load parameters are not of type LineLoad or MovingLoad or UvecLoad.
        """

        # line and moving load can only be assigned to 1d geometry
        if isinstance(load_parameters, (LineLoad, MovingLoad, UvecLoad)):
            ndim_load = 1
        else:
            raise ValueError(f"Load parameter provided is not supported: `{load_parameters.__class__.__name__}`.")

        # Get the geometry of the matching model part
        geometry = next((bmp.geometry for bmp in self.body_model_parts if bmp.name == model_part_name), None)
        if geometry is None:
            raise ValueError(f"Geometry in model part with name `{model_part_name}` not found.")

        geometry_ids = list(geometry.lines.keys())

        # add physical group to gmsh
        self.gmsh_io.add_physical_group(load_name, ndim_load, geometry_ids)

        # create model part
        model_part = ModelPart(load_name)

        # retrieve geometry from gmsh and add to model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, load_name)
        # add load parameters to model part
        model_part.parameters = load_parameters
        # validate the input
        model_part.validate_input()

        self.process_model_parts.append(model_part)

    def add_boundary_condition_by_geometry_ids(self, ndim_boundary: int, geometry_ids: Sequence[int],
                                               boundary_parameters: BoundaryParametersABC, name: str):
        """
        Add a boundary condition to the model by giving the geometry ids of the boundary condition.

        Args:
            - ndim_boundary (int): dimension of the boundary condition
            - geometry_ids (Sequence[int]): geometry ids of the boundary condition
            - boundary_condition (:class:`stem.boundary_condition.BoundaryCondition`): boundary condition object
            - name (str): name of the boundary condition

        """

        # add physical group to gmsh
        self.gmsh_io.add_physical_group(name, ndim_boundary, geometry_ids)

        # create model part
        model_part = ModelPart(name)

        # retrieve geometry from gmsh and add to model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        # add boundary parameters to model part
        model_part.parameters = boundary_parameters

        model_part.validate_input()

        self.process_model_parts.append(model_part)

    def add_boundary_condition_on_plane(self, plane_vertices: Sequence[Sequence[float]],
                                        boundary_parameters: BoundaryParametersABC, name: str):
        """
        Adds a boundary condition to the model by giving a sequence of 3D coordinates. The boundary condition is added
        to all the surfaces which fall within the plane.

        Args:
            - plane_vertices (Sequence[Sequence[float]]): Minimum 3 vertices of the plane.
            - boundary_parameters (:class:`stem.boundary.BoundaryParametersABC`): The parameters of the boundary
                condition.
            - name (str): The name of the boundary condition.

        Raises:
            - ValueError: if the plane has less than 3 vertices.

        """

        if len(plane_vertices) < 3:
            raise ValueError("At least 3 vertices are required to define a plane.")

        # get surface ids on the plane
        surface_ids = self.gmsh_io.get_surface_ids_at_plane(plane_vertices)

        # add physical group to gmsh
        self.gmsh_io.add_physical_group(name, 2, surface_ids)

        # create model part
        model_part = ModelPart(name)

        # retrieve geometry from gmsh and add to model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        # add boundary parameters to model part
        model_part.parameters = boundary_parameters

        model_part.validate_input()

        model_part_exists = False
        for existing_part in self.process_model_parts:
            if existing_part.name == name and existing_part.parameters == model_part.parameters:
                # extra geometry ids are added to the geometry of an existing model part
                model_part_exists = True
                existing_part.geometry = model_part.geometry

        if not model_part_exists:
            self.process_model_parts.append(model_part)

    def add_boundary_condition_on_polygon(self, polygon_coordinates: Sequence[Sequence[float]],
                                          boundary_parameters: BoundaryParametersABC, name: str):
        """
        Adds a boundary condition to the model by giving a sequence of 3D coordinates. The boundary condition is added
        to all the surfaces which fall within the polygon. A surface is considered to be within the polygon if all its
        points are within the polygon.

        Args:
            - polygon_coordinates (Sequence[Sequence[float]]): The coordinates of the polygon.
            - boundary_parameters (:class:`stem.boundary.BoundaryParametersABC`): The parameters of the boundary
                condition.
            - name (str): The name of the boundary condition.

        """

        # get surface ids within the polygon
        surface_ids = self.gmsh_io.get_surface_ids_at_polygon(polygon_coordinates)

        # add physical group to gmsh
        self.gmsh_io.add_physical_group(name, 2, surface_ids)

        # create model part
        model_part = ModelPart(name)

        # retrieve geometry from gmsh and add to model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        # add boundary parameters to model part
        model_part.parameters = boundary_parameters

        model_part.validate_input()

        self.process_model_parts.append(model_part)

    def add_hinge_on_beam(self, beam_model_part_name: str, hinge_coordinates: Sequence[Sequence[float]],
                          hinge_parameters: HingeParameters, hinge_model_part_name: str):
        """
        Adds a hinge to the model by giving the name of the beam model part where the hinge has to be applied.

        Args:
            - beam_model_part_name (str): name of the beam model part where the hinge needs to be applied.
            - hinge_coordinates (Sequence[Sequence[float]]): coordinates of the hinge.
            - hinge_parameters (:class:`stem.hinge.HingeParametersABC`): hinge parameters to define the hinge object.
            - hinge_model_part_name (str): name of the hinge.

        Raises:
            - ValueError: if the hinge model part does not have a geometry.
            - ValueError: if the beam model part is not found.
            - ValueError: if the beam model part does not have a geometry.
            - ValueError: if the beam model part does not have a beam material.
            - NotImplementedError: if the hinge is applied in a 2D model.
            - ValueError: if the hinge points are not part of the beam model part.
            """

        gmsh_input = {hinge_model_part_name: {"coordinates": hinge_coordinates, "ndim": 0}}
        self.gmsh_io.generate_geometry(gmsh_input, "")
        self.synchronise_geometry()

        # create model part
        model_part = ModelPart(hinge_model_part_name)
        model_part.parameters = hinge_parameters

        # set the geometry of the model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, hinge_model_part_name)

        if model_part.geometry is None:
            raise ValueError(f"Model part `{hinge_model_part_name}` has no geometry.")

        beam_model_part = self.get_model_part_by_name(beam_model_part_name)
        if beam_model_part is None:
            raise ValueError(f"Model part `{beam_model_part_name}` not found.")
        if beam_model_part.geometry is None:
            raise ValueError(f"Model part `{beam_model_part_name}` has no geometry.")

        # validate if the hinge is applied on a 3D beam model part
        if not isinstance(beam_model_part, BodyModelPart) or not isinstance(
                beam_model_part.material, StructuralMaterial) or not isinstance(
                    beam_model_part.material.material_parameters, EulerBeam):
            raise ValueError("Hinges can only be applied to beam model parts")

        if self.ndim != 3:
            raise NotImplementedError("Hinges can only be applied in 3D models.")

        beam_points = beam_model_part.geometry.points
        if not all(point_id in beam_points for point_id in model_part.geometry.points.keys()):
            raise ValueError(f"The hinge points are not part of the beam model part `{beam_model_part_name}`.")

        self.process_model_parts.append(model_part)

    def add_output_settings(self,
                            output_parameters: OutputParametersABC,
                            part_name: Optional[str] = None,
                            output_dir: str = "./",
                            output_name: Optional[str] = None):
        """
        Adds an output to the model, including the output folder, the name of the output file (if applicable) and the
        part of interest to output.

        If no part is specified, the whole model is considered as output.

        Args:
            - output_parameters (:class:`OutputParametersABC`): class containing the output parameters
            - part_name (Optional[str]): name of the submodelpart to be given in output. If None, all the model is
                provided in  output.
            - output_dir (str): output directory for the relative or absolute path to the output file. The \
                path will be created if it does not exist yet. \n

                example1='test1' results in the test1 output folder relative to current folder as './test1'\
                example2='path1/path2/test2' saves the outputs in './path1/path2/test2' \
                example3='C:/Documents/yourproject/test3' saves the outputs in 'C:/Documents/yourproject/test3'.

                if output_dir is None, then the current directory is assumed.

                [NOTE]: for VTK file type, the content of the target directory will be deleted. Therefore a subfolder is
                always appended to the specified output directory to avoid erasing important memory content.
                The appended folder is defined based on the submodelpart name specified.

            - output_name (Optional[str]): Name for the output file. This parameter is \
                  used by GiD and JSON outputs while is ignored in VTK. If the name is not \
                  given, the part_name is used instead.

        Raises:
            - ValueError: if the model part for which output needs to be requested doesn't exist.

        """

        # check if the model part exists (if None, all model is output)
        if (part_name is not None and part_name != "porous_computational_model_part"
                and self.get_model_part_by_name(part_name=part_name) is None):
            raise ValueError("Model part for which output needs to be requested doesn't exist.")

        self.output_settings.append(
            Output(output_parameters=output_parameters,
                   part_name=part_name,
                   output_dir=output_dir,
                   output_name=output_name))

    def add_output_settings_by_coordinates(self,
                                           coordinates: Sequence[Sequence[float]],
                                           output_parameters: OutputParametersABC,
                                           part_name: str,
                                           output_dir: str = "./",
                                           output_name: Optional[str] = None):
        """
        Sets coordinates where the output is to be defined.
        The coordinates have to be laying on an existing geometry surface.
        Both the first- and end-point has to lie on one of the edges of the surface. A new process model part is
        created, to specify the list of nodes of interest.

        Current limitations:
            - The nodes have to be laying on an existing geometry surface.
            - The first and endpoint have to lie on one of the edges of the surface.
            - A single point cannot be provided, but is always a sequence of lines.

        Args:
            - coordinates (Optional[Sequence[Sequence[float]]]): A list of nodes that are of interest for the
                outputs.
            - output_parameters (:class:`OutputParametersABC`): class containing the output parameters
            - part_name (str): name of the submodelpart name for the output. Must be different from
                existing parts.
            - output_dir (Optional[str]): output directory for the relative or absolute path to the output file. The \
                path will be created if it does not exist yet. \n

                example1='test1' results in the test1 output folder relative to current folder as './test1'\
                example2='path1/path2/test2' saves the outputs in './path1/path2/test2' \
                example3='C:/Documents/yourproject/test3' saves the outputs in 'C:/Documents/yourproject/test3'.

                if output_dir is None, then the current directory is assumed.

                [NOTE]: for VTK file type, the content of the target directory will be deleted. Therefore, a subfolder
                is always appended to the specified output directory to avoid erasing important memory content.
                The appended folder is defined based on the submodelpart name specified.

            - output_name (Optional[str]): Name for the output file. This parameter is \
                  used by GiD and JSON outputs while is ignored in VTK. If the name is not \

        """

        # TODO: add validation for sequential pair of points to lie on the an existing geometry surface.
        # TODO: add validation for start and end-point to lie on the edges

        # validation of input coordinates
        Utils.validate_coordinates(coordinates)

        gmsh_input = {part_name: {"coordinates": coordinates, "ndim": 1}}

        self.gmsh_io.generate_geometry(gmsh_input, "")

        # create model part
        model_part = ModelPart(part_name)
        model_part.parameters = output_parameters

        # set the geometry of the model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, part_name)

        self.process_model_parts.append(model_part)

        # add output to the output list
        self.add_output_settings(output_parameters=output_parameters,
                                 part_name=part_name,
                                 output_dir=output_dir,
                                 output_name=output_name)

    def __exclude_non_output_nodes(self, eps: float = 1e-06):
        """
        Exclude the nodes that are further than `eps` to the requested output nodes for the output model part.

        Args:
            - eps (float): the radius distance to search for nodes. In practice is a tolerance for the search
                algorithm to look for close nodes.

        Raises:
            - ValueError: if the parameters of the model part are None.
            - ValueError: if the model part has no geometry.
            - ValueError: if the model part is not yet meshed.

        """

        for model_part in self.process_model_parts:

            # adjust the mesh of output model parts. Exclude element, and keep only the nodes of corresponding to the
            # output locations.
            if isinstance(model_part.parameters, OutputParametersABC):

                if model_part.parameters is None:
                    raise ValueError("The model part doesn't have parameters.")

                if model_part.geometry is None:
                    raise ValueError("The model part has no geometry.")

                if model_part.mesh is None:
                    raise ValueError("process model part has not been meshed yet!")

                # retrieve the node ids close to the geometry points (smaller than eps meters)
                filtered_node_ids = Utils.find_node_ids_close_to_geometry_nodes(mesh=model_part.mesh,
                                                                                geometry=model_part.geometry,
                                                                                eps=eps)

                new_mesh = Mesh(ndim=model_part.mesh.ndim)
                new_mesh.nodes = {node_id: model_part.mesh.nodes[node_id] for node_id in filtered_node_ids}
                new_mesh.elements = {}
                model_part.mesh = new_mesh

                self.gmsh_io.mesh_data["physical_groups"][model_part.name]["node_ids"] = (list(new_mesh.nodes.keys()))

    def add_field(self, part_name: str, field_parameters: ParameterFieldParameters):
        """
        Add a parameter field to a given model part (specified by the part_name input). if the `mean_value` attribute
        of the field generator is None, the corresponding material property is used as mean.

        Args:
            - part_name (str): model of the part name where to apply the random field generation.
            - field_parameters (:class:`stem.additional_processes.ParameterFieldParameters`): the objects containing \
                the parameters necessary for the definition of the field.

        Raises:
            - ValueError: if the part name is not a body model part.
            - ValueError: if the body model part has no material.
            - ValueError: if the mean value of the material property is a boolean.

        """

        # Check if the model part exists and retrieve the part
        target_part = self.get_model_part_by_name(part_name=part_name)

        # Check if the model part is a body model part
        if not isinstance(target_part, BodyModelPart):
            raise ValueError(f"The target part, `{part_name}`, is not a body model part.")

        # Check that the body model part has a material
        if target_part.material is None:
            raise ValueError("No material assigned to the body model part!")

        # Check if the field file names are provided, if not, set them to empty strings
        if field_parameters.field_file_names is None:
            field_parameters.field_file_names = [""] * len(field_parameters.property_names)

        for i, property_name in enumerate(field_parameters.property_names):
            # define the name of the new model part to generate the random field
            new_part_name = part_name + "_" + property_name.lower() + "_field"

            # validation for json input files
            if field_parameters.function_type == "json_file":
                if isinstance(field_parameters.field_generator, RandomFieldGenerator):
                    if field_parameters.field_generator.mean_value is None:

                        # Get the property of the material, this is the mean value of the random field.
                        # Checks also if the material of the body model part contains the desired parameter
                        mean_value_material = target_part.material.get_property_in_material(property_name=property_name)

                        if isinstance(mean_value_material, bool) or not isinstance(mean_value_material, (float, int)):
                            raise ValueError("The property for which a random field needs to be generated, "
                                             f"`{property_name}` is not a numeric value.")

                        field_parameters.field_generator.mean_value = mean_value_material

                if field_parameters.field_file_names[i] == "":
                    field_parameters.field_file_names[i] = new_part_name + ".json"

            model_part_geometry_ids = self.gmsh_io.geo_data["physical_groups"][part_name]["geometry_ids"]
            model_part_ndim = self.gmsh_io.geo_data["physical_groups"][part_name]["ndim"]
            # create the field_parameter physical group and model part
            self.gmsh_io.add_physical_group(new_part_name, model_part_ndim, model_part_geometry_ids)
            model_part = ModelPart(new_part_name)

            model_part.parameters = field_parameters

            model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, new_part_name)

            # add the field_parameter part to process model parts
            self.process_model_parts.append(model_part)

    def synchronise_geometry(self):
        """
        Synchronise the geometry of all model parts and synchronise the geometry of the whole model. This function
        recalculates all ids and connectivities of all geometrical entities.

        """

        # synchronize gmsh and extract geo data
        self.gmsh_io.synchronize_gmsh()
        self.gmsh_io.extract_geo_data()

        # Get the geometry from the geo_data for each model part
        for model_part in self.all_model_parts:
            model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, model_part.name)

        # get the complete geometry
        self.geometry = Geometry.create_geometry_from_geo_data(self.gmsh_io.geo_data)

    def set_mesh_size(self, element_size: float):
        """
        Set the element size to dimension [m].

        Args:
            - element_size (float): the desired element size [m].

        """
        self.mesh_settings.element_size = element_size

    def generate_mesh(self,
                      save_file: bool = False,
                      mesh_output_dir: str = "./",
                      mesh_name: str = "mesh_file",
                      open_gmsh_gui: bool = False):
        """
        Generate the mesh for the whole model.

        Args:
            - save_file (bool): If True, saves mesh data to gmsh msh file. (default is False)
            - mesh_name (str): Name of gmsh model and mesh output file.  (default is working directory)
            - mesh_output_dir (bool): Output directory of mesh file. (default is `mesh_file`)
            - open_gmsh_gui (bool): User indicates whether to open gmsh interface (default is False)

        """

        # generate mesh
        self.gmsh_io.generate_mesh(self.ndim,
                                   element_size=self.mesh_settings.element_size,
                                   order=self.mesh_settings.element_order,
                                   save_file=save_file,
                                   mesh_output_dir=mesh_output_dir,
                                   mesh_name=mesh_name,
                                   open_gmsh_gui=open_gmsh_gui)

        # add the mesh to each model part
        for model_part in self.all_model_parts:
            model_part.mesh = Mesh.create_mesh_from_gmsh_group(self.gmsh_io.mesh_data, model_part.name)

        # per process model part, check if the condition elements are applied to a body model part and set the
        # node ordering of the condition elements to match the body elements
        for process_model_part in self.process_model_parts:

            # only check if the process model part is a condition element
            if isinstance(process_model_part.parameters,
                          (LineLoad, MovingLoad, UvecLoad, SurfaceLoad, AbsorbingBoundary)):
                # match the condition elements with the body elements on which the conditions are applied
                matched_elements = self.__find_matching_body_elements_for_process_model_part(process_model_part)

                # check the ordering of the nodes of the conditions. If it does not match flip the order.
                self.__check_ordering_process_model_part(matched_elements, process_model_part)

        # perform post mesh operations
        self.__post_mesh()

    def __post_mesh(self):
        """
        Function to be called after the mesh is generated and finalised. The following steps are performed:
            - initialise field parameters (e.g., random fields).
            - exclude the nodes that are not output nodes.
            - adjust the elements for the spring damper parts.

        """
        self.__initialise_fields()

        self.__exclude_non_output_nodes()
        self.__adjust_mesh_spring_dampers()

    def __initialise_fields(self):
        """
        Initialise the field parameters for the field generator objects.

        Raises:
            - ValueError: if the field generator is not provided for a parameter field.

        """

        for model_part in self.process_model_parts:

            if isinstance(model_part.parameters, ParameterFieldParameters):

                # initialise the fields for the json output files. Tiny expressions don't require it.
                if model_part.parameters.function_type == "json_file":
                    if model_part.parameters.field_generator is None:
                        raise ValueError("Field generator is not provided for parameter field.")

                    if model_part.mesh is None:
                        raise ValueError(
                            "Mesh not yet initialised. Please generate the mesh using Model.generate_mesh().")

                    centroids = model_part.mesh.calculate_centroids()
                    model_part.parameters.field_generator.generate(centroids)

    def __adjust_mesh_spring_dampers(self):
        """
        Adjusts the mesh of the spring dampers which are normally added on an existing line.
        If the line is broken in multiple elements, mesh requires to be adjusted so that there is only one element
        per spring-damper.

        Raises:
            - ValueError: if the mesh is not initialised.

        """

        # get the new element id which is the maximum current element id + 1
        new_element_id = self.__get_maximum_element_id() + 1

        # retrieve connectivities and cluster into individual spring-damper elements
        for mp in self.body_model_parts:

            if (isinstance(mp.material, StructuralMaterial)
                    and isinstance(mp.material.material_parameters, ElasticSpringDamper)):

                # assert mesh is initialised
                if mp.mesh is None:
                    raise ValueError("Mesh not yet initialised. Please generate the mesh using Model.generate_mesh().")

                # get the sequences of springs in the body model part
                spring_node_ids = self.__get_line_string_end_nodes(model_part=mp)

                new_mesh = Mesh(ndim=1)

                # loop over each spring-damper sequence
                for (start_node_id, end_node_id) in spring_node_ids:
                    # add the existing nodes to the new mesh
                    new_mesh.nodes[start_node_id] = mp.mesh.nodes[start_node_id]
                    new_mesh.nodes[end_node_id] = mp.mesh.nodes[end_node_id]

                    # create new 2n line element
                    new_mesh.elements[new_element_id] = Element(id=new_element_id,
                                                                element_type="LINE_2N",
                                                                node_ids=[start_node_id, end_node_id])

                    # increment the element id
                    new_element_id += 1

                # add the new mesh to the mesh data
                self.gmsh_io.mesh_data["physical_groups"][mp.name]["node_ids"] = sorted(list(new_mesh.nodes.keys()))
                self.gmsh_io.mesh_data["physical_groups"][mp.name]["element_ids"] = \
                    sorted(list(new_mesh.elements.keys()))

                for element_id, element in new_mesh.elements.items():
                    self.gmsh_io.mesh_data["elements"]["LINE_2N"][element_id] = element.node_ids

                mp.mesh = new_mesh

    def __get_maximum_element_id(self) -> int:
        """
        Returns the maximum element id within the mesh from the mesh data

        Returns:
            - int: the maximum element id

        """
        max_element_id = 0
        for mesh_element_info in self.gmsh_io.mesh_data["elements"].values():
            max_element_id = max(max_element_id, max(mesh_element_info.keys()))

        return int(max_element_id)

    def __get_line_string_end_nodes(self, model_part: ModelPart) -> List[List[int]]:
        """
        Finds the nodes at both of the line string ends. A line string end is defined as the node which coincides with
        a geometry point.

        Args:
            - model_part (:class:`stem.model_part.ModelPart`): model part from which the spring elements need to be
                extracted.

        Raises:
            - ValueError: if the geometry is not initialised.
            - ValueError: if the mesh is not initialised.

        Returns:
            - List[List[int]]: a list of lists which contains both end nodes for separate each line string.

        """

        # assert mesh and geometry are initialised
        if model_part.geometry is None:
            raise ValueError(f"Geometry of model part `{model_part.name}` not yet initialised.")

        if model_part.mesh is None:
            raise ValueError(f"Mesh of model part `{model_part.name}` not yet initialised.")

        # find end nodes
        end_nodes = self.__find_end_nodes_of_line_strings(model_part.mesh)
        # find the node ids corresponding to the geometry points
        node_ids_at_geometry_points = set(
            int(node_id) for node_id in Utils.find_node_ids_close_to_geometry_nodes(
                mesh=model_part.mesh, geometry=model_part.geometry, eps=1e-06))
        element_ids_search_space = set(model_part.mesh.elements.keys())
        node_ids_search_space = set(model_part.mesh.nodes.keys())

        # retrieve the connectivity
        # node -> elements
        node_to_elements = model_part.mesh.find_elements_connected_to_nodes()

        # initialise output list
        line_node_ids = []
        # initialise a set for end-point we have already encountered in the clustering algorithm
        completed_points = set()
        for end_node in end_nodes:

            # only consider the end nodes that are not already in the completed_points list
            if end_node not in completed_points:

                # remove the end node from the list containing the node ids
                node_ids_search_space.remove(end_node)
                first_node_id = None

                # if the point is not the end of the cluster, continue until you find the end of the cluster and include
                # all the line strings
                while first_node_id not in end_nodes and len(element_ids_search_space) > 0:

                    # first point is the end node
                    if first_node_id is None:
                        first_node_id = end_node

                    second_node_id = self.__find_next_node_along_line_elements(first_node_id, element_ids_search_space,
                                                                               node_ids_search_space, node_to_elements,
                                                                               model_part.mesh.elements,
                                                                               node_ids_at_geometry_points)

                    # add the end nodes to the list (start node, end node)
                    line_node_ids.append([first_node_id, second_node_id])
                    # update the next point for the search
                    first_node_id = second_node_id

                # add the end point to the completed_points in order to reduce the search space
                completed_points.add(first_node_id)
        return line_node_ids

    @staticmethod
    def __find_end_nodes_of_line_strings(mesh: Mesh) -> Set[int]:
        """
        Finds the nodes at the end of linestrings.

        Args:
            - mesh (:class:`stem.mesh.Mesh`): mesh from which end nodes needs to be extracted.

        Returns:
            - end_nodes (Set[int]): End node ids of linestring clusters.

        """
        nodes_to_elements = mesh.find_elements_connected_to_nodes()
        end_nodes = set(node_id for node_id, elements in nodes_to_elements.items() if len(elements) == 1)
        return end_nodes

    @staticmethod
    def __find_next_node_along_line_elements(start_node_id: int, remaining_element_ids: Set[int],
                                             remaining_node_ids: Set[int], node_to_elements: Dict[int, List[int]],
                                             line_elements: Dict[int, Element], target_node_ids: Set[int]) -> int:
        """
        Finds the next node along line element. The remaining_element_ids and remaining_node_ids keeps track of
        the direction of the previous searches and orients the search on a unique direction.

        Args:
            - start_node_id (int): the node id to start searching the next node along the elements.
            - remaining_element_ids (Set[int]): the element ids that have not been followed yet.
            - remaining_node_ids (Set[int]): the node ids that have not been crossed yet.
            - node_to_elements (Dict[int, List[int]]): mapping of node_ids to the element_ids which is connected to.
            - line_elements (Dict[int, :class:`stem.mesh.Element`]): dictionary of line elements.
            - target_node_ids (Set[int]): set of nodes to be searched for.

        Raises:
            - ValueError: if not all elements are line elements.
            - ValueError: if there is a fork in the mesh, the algorithm cannot find the next node.
            - ValueError: if number of interation in the while loop are exceeded and something went wrong in the
                algorithm.

        Returns:
            - int: the node id which is connected to the start_node_id within the search space.

        """

        # check if all elements are line elements
        if any(element.element_type not in {"LINE_2N", "LINE_3N"} for element in line_elements.values()):
            raise ValueError("Not all elements are line elements.")

        # initialise variables before loop
        next_node = start_node_id

        # start the search for the connected node
        max_iterations = len(remaining_element_ids)
        for _ in range(max_iterations):

            # find the element(s) connected to the node that have not yet been searched for.
            elements_connected = set(node_to_elements[next_node]) & remaining_element_ids

            # check if there is a fork in the mesh, which is not allowed
            if len(elements_connected) > 1:
                raise ValueError(f"There is a fork in the mesh at elements: {elements_connected}, the next node along "
                                 f"the line cannot be found.")

            next_element_id = elements_connected.pop()

            # reduce search space for next iteration
            remaining_element_ids.remove(next_element_id)

            # find the node(s) connected to the element that have not yet been found yet.
            next_node = next(node_id for node_id in line_elements[next_element_id].node_ids
                             if node_id in remaining_node_ids)

            # reduce search space for next iteration
            remaining_node_ids.remove(next_node)

            # if the node is one of the nodes of interest, return them, otherwise continue.
            if next_node in target_node_ids:
                return next_node

        raise ValueError("Next node along the line cannot be found. As it is not included in the search space")

    def __find_matching_body_elements_for_process_model_part(self, process_model_part: ModelPart) \
            -> List[Tuple[Element, Element]]:
        """
        For a process model part, tries finds the matching body elements on which the condition elements are applied.

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): model part from which element nodes needs to be \
                extracted.
        Raises:
            - ValueError: if mesh is not initialised yet.
            - ValueError: if condition elements don't have a corresponding body element.

        Returns:
            - List[Tuple[:class:`stem.mesh.Element`, :class:`stem.mesh.Element`]]: List containing
                the matched condition and body element parts.

        """
        # validation step for process model part
        if process_model_part.mesh is None:
            raise ValueError(f"Mesh of process model part: {process_model_part.name} is not yet initialised.")

        # initialise body element dictionaries
        nodes_to_elements_body: Dict[int, List[int]] = {}
        all_body_elements = {}

        # loop over the body model parts (bmp) to match the elements of the process model part
        for body_model_part in self.body_model_parts:

            # validation step for body model part
            if body_model_part.mesh is None:
                raise ValueError(f"Mesh of body model part: {body_model_part.name} is not yet initialised.")

            # find which nodes within the body model part are connected to which elements
            for node_id, element_ids in body_model_part.mesh.find_elements_connected_to_nodes().items():
                nodes_to_elements_body.setdefault(node_id, element_ids).extend(element_ids)

            all_body_elements.update(body_model_part.mesh.elements)

        # for each process element, check if there is a match with the current body part elements
        process_elements = process_model_part.mesh.elements
        matched_elements = []
        for process_element_id in process_elements:

            # check if all nodes of the process element are present in the body elements
            if not all(node_id in nodes_to_elements_body for node_id in process_elements[process_element_id].node_ids):
                break

            # get the connected body elements for each node of the process element
            connected_elements = [
                set(nodes_to_elements_body[node_id]) for node_id in process_elements[process_element_id].node_ids
            ]

            # find which body elements are connected to all nodes of the process element
            common_elements = list(set.intersection(*connected_elements))

            # if there are common elements, add the process element and the first connected body element to the
            # matched_elements list
            if len(common_elements) > 0:
                matched_elements.append(
                    (process_model_part.mesh.elements[process_element_id], all_body_elements[common_elements[0]]))

        # if not all process elements are matched, raise an error
        if len(matched_elements) < len(process_elements):
            # find which process elements are not matched
            matched_process_elements = set(pe.id for pe, _ in matched_elements)
            unmatched_process_elements = set(process_model_part.mesh.elements.keys()) - matched_process_elements

            raise ValueError(f"Condition elements: {list(unmatched_process_elements)} do not have a corresponding "
                             f"body element.")

        return matched_elements

    def __check_ordering_process_model_part(self, matched_elements: List[Tuple[Element, Element]],
                                            process_model_part: ModelPart):
        """
        Check if the node ordering of the process element matches the node ordering of the neighbouring body element.
        If not, flip the node ordering of the process element.

        Args:
            - matched_elements (List[Tuple[:class:`stem.mesh.Element`, :class:`stem.mesh.Element`]]): Dictionary \
                containing the matched condition and body element parts.
            - process_model_part (:class:`stem.model_part.ModelPart`): model part from which element nodes needs to be \
                extracted.

        Raises:
            - ValueError: if mesh is not initialised yet.

        """

        if process_model_part.mesh is None:
            raise ValueError(f"Mesh of process model part: {process_model_part.name} is not yet initialised.")

        # loop over the matched elements
        elements_to_flip = []

        for (process_element, body_element) in matched_elements:

            # element info such as order, number of edges, element types etc.
            process_el_info = ELEMENT_DATA[process_element.element_type]
            body_el_info = ELEMENT_DATA[body_element.element_type]

            if process_el_info["ndim"] == 1:

                # get all line edges of the body element and check if the process element is defined on one of them
                # if the nodes are equal, but the node order isn't, flip the node order of the process element
                body_line_edges = Utils.get_element_edges(body_element)
                for edge in body_line_edges:
                    if set(edge) == set(process_element.node_ids) and list(edge) != process_element.node_ids:
                        elements_to_flip.append(process_element)

            elif body_el_info["ndim"] == 3 and process_el_info["ndim"] == 2:

                # check if the normal of the condition element is not defined outwards of the body element
                if not Utils.is_volume_edge_defined_outwards(process_element, body_element,
                                                             self.gmsh_io.mesh_data["nodes"]):
                    elements_to_flip.append(process_element)

        # flip condition elements if required
        if len(elements_to_flip) > 0:

            # flip elements, it is required that all elements in the array are of the same type
            Utils.flip_node_order(elements_to_flip)

    def __validate_model_part_names(self):
        """
        Checks if all model parts have a unique name.

        Raises:
            - ValueError: If not all model parts have a name.
            - ValueError: If not all model part names are unique.

        """

        unique_names = set()

        for model_part in self.all_model_parts:
            # Check if all model parts have a name
            if model_part.name is None:
                raise ValueError("All model parts must have a name")

            if model_part.name in unique_names:
                raise ValueError("All model parts must have a unique name")
            unique_names.add(model_part.name)

    def __add_gravity_model_part(self, gravity_load: GravityLoad, ndim: int, geometry_ids: Sequence[int]):
        """
        Add a gravity model part to the complete model.

        Args:
            - gravity_load (:class:`stem.load.GravityLoad`): The gravity load object.
            - ndim (int): The number of dimensions of the on which the gravity load should be applied.
            - geometry_ids (Sequence[int]): The geometry on which the gravity load should be applied.

        """

        # set new model part name
        model_part_name = f"gravity_load_{ndim}d"

        # create new gravity physical group and model part
        self.gmsh_io.add_physical_group(model_part_name, ndim, geometry_ids)
        model_part = ModelPart(model_part_name)

        model_part.parameters = gravity_load

        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, model_part_name)

        # add gravity load to process model parts
        self.process_model_parts.append(model_part)

    def get_model_part_by_name(self, part_name: str) -> Optional[ModelPart]:
        """
        Find the model part matching the given part_name

        Args:
            - part_name (str): the name of the part to retrieve.

        Returns:
            - Optional[:class:`stem.model_part.ModelPart`]: matched model part or None if no match.
        """

        for model_part in self.all_model_parts:
            if model_part.name == part_name:
                return model_part
        print(f"Model part `{part_name}` not found!")
        return None

    def __add_gravity_load(self):
        """
        Add a gravity load to the complete model.

        """

        # set gravity load at vertical axis
        gravity_load_values: List[float] = [0, 0, 0]
        gravity_load_values[VERTICAL_AXIS] = GRAVITY_VALUE
        gravity_load = GravityLoad(value=gravity_load_values, active=[True, True, True])

        # get all body model part names
        body_model_part_names = [body_model_part.name for body_model_part in self.body_model_parts]

        # get geometry ids and ndim for each body model part
        model_parts_geometry_ids = np.array(
            [self.gmsh_io.geo_data["physical_groups"][name]["geometry_ids"] for name in body_model_part_names])

        model_parts_ndim = np.array(
            [self.gmsh_io.geo_data["physical_groups"][name]["ndim"] for name in body_model_part_names]).ravel()

        # add gravity load as physical group per dimension
        body_geometries_1d = model_parts_geometry_ids[model_parts_ndim == 1].ravel()
        if len(body_geometries_1d) > 0:
            self.__add_gravity_model_part(gravity_load, 1, body_geometries_1d)

        body_geometries_2d = model_parts_geometry_ids[model_parts_ndim == 2].ravel()
        if len(body_geometries_2d) > 0:
            self.__add_gravity_model_part(gravity_load, 2, body_geometries_2d)

        body_geometries_3d = model_parts_geometry_ids[model_parts_ndim == 3].ravel()
        if len(body_geometries_3d) > 0:
            self.__add_gravity_model_part(gravity_load, 3, body_geometries_3d)

        self.synchronise_geometry()
        self.gmsh_io.finalize_gmsh()

    def get_all_nodes(self):
        """
        Retrieve all the unique nodes in the model mesh.

        Raises:
            - ValueError: If the geometry has not been meshed yet.

        Returns:
            - node_dict (Dict[int, :class:`stem.mesh.Node`]): dictionary containing nodes id and nodes objects.

        """

        node_dict: Dict[int, Node] = {}
        for mp in self.all_model_parts:
            if mp.mesh is None:
                raise ValueError("Geometry has not been meshed yet! Please first run the Model.generate_mesh method.")
            node_dict.update(mp.mesh.nodes)

        return node_dict

    def validate(self):
        """
        Validate the model. \
            - Checks if all model parts have a unique name.

        """

        self.__validate_model_part_names()

    def show_geometry(self,
                      show_volume_ids: bool = False,
                      show_surface_ids: bool = False,
                      show_line_ids: bool = False,
                      show_point_ids: bool = False,
                      file_name: str = "tmp_geometry_file.html",
                      auto_open: bool = True):
        """
        Show the 2D or 3D geometry in a plot.

        Args:
            - show_volume_ids (bool): Show the volume ids in the plot. (default False)
            - show_surface_ids (bool): Show the surface ids in the plot. (default False)
            - show_line_ids (bool): Show the line ids in the plot. (default False)
            - show_point_ids (bool): Show the point ids in the plot. (default False)
            - file_name (str): The name of the html file in which the plot is saved. (default "tmp_geometry_file.html")
            - auto_open (bool): Open the html file automatically. (default True)

        Raises:
            - ValueError: If the geometry is not set.

        """
        self.synchronise_geometry()

        if self.geometry is None:
            raise ValueError("Geometry must be set before showing the geometry")

        fig = PlotUtils.create_geometry_figure(self.ndim, self.geometry, show_volume_ids, show_surface_ids,
                                               show_line_ids, show_point_ids)

        fig.write_html(file_name, auto_open=auto_open)

    def __setup_stress_initialisation(self):
        """
        Set up the stress initialisation. For K0 procedure and gravity loading, a gravity load is added to the model.

        Raises:
            - ValueError: If the project parameters are not set.

        """

        if self.project_parameters is None:
            raise ValueError("Project parameters must be set before setting up the stress initialisation")

        # add gravity load if K0 procedure or gravity loading is used
        if (self.project_parameters.settings.stress_initialisation_type == StressInitialisationType.K0_PROCEDURE
                or self.project_parameters.settings.stress_initialisation_type
                == StressInitialisationType.GRAVITY_LOADING):
            self.__add_gravity_load()

    def __add_water_condition_if_not_provided(self):
        """
        Add a water condition if not provided by the user.

        """
        for process_model_part in self.process_model_parts:
            # if one of the model parts already contains water, do not add zero water pressure
            if isinstance(process_model_part.parameters, WaterProcessParametersABC):
                return

        # if all model parts are structural, do not add water pressure
        materials = [body_model_part.material for body_model_part in self.body_model_parts]
        if all(isinstance(material, StructuralMaterial) for material in materials):
            return

        water_model_part = ModelPart("zero_water_pressure")
        water_model_part.parameters = UniformWaterPressure(water_pressure=0.0)

        geometry_ids = []

        for body_model_part in self.body_model_parts:

            # if body model part has geometry, add the geometry ids to the list
            if body_model_part.geometry is not None:
                if self.ndim == 2:
                    geometry_ids.extend(list(body_model_part.geometry.surfaces.keys()))
                elif self.ndim == 3:
                    geometry_ids.extend(list(body_model_part.geometry.volumes.keys()))

        # add physical group to gmsh
        self.gmsh_io.add_physical_group(water_model_part.name, self.ndim, geometry_ids)

        # retrieve geometry from gmsh and add to model part
        water_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, water_model_part.name)

        self.process_model_parts.append(water_model_part)

        # re-synchronise the geometry as the water model part has been added
        self.synchronise_geometry()

    def post_setup(self):
        """
        Post setup of the model:
            - Synchronise the geometry.
            - Generate the mesh.
            - Validate the model.
            - Set up the stress initialisation.

        """

        self.synchronise_geometry()
        self.validate()

        self.__add_water_condition_if_not_provided()
        self.__setup_stress_initialisation()

        # finalize gmsh
        self.gmsh_io.finalize_gmsh()

    def set_element_size_of_group(self, element_size: float, group_name: str):
        """
        Set the element size of a group of elements. In multiple groups share the same mesh, the lowest element size is
        used.

        Args:
            - element_size (float): The element size.
            - group_name (str): The name of the group.

        Raises:
            - ValueError: If the group name is not found.

        """
        if group_name not in self.gmsh_io.geo_data["physical_groups"]:
            raise ValueError(f"Group name `{group_name}` not found.")

        self.gmsh_io.geo_data["physical_groups"][group_name]["element_size"] = element_size

        self.gmsh_io.generate_geo_from_geo_data()

    def split_model_part(self, from_model_part_name: str, to_model_part_name: str, geometry_ids: List[int],
                         new_parameters: Union[Material, ProcessParameters]):
        """
        Move the geometry from one model part to another.

        Args:
            - from_model_part_name (str): The name of the model part from which the geometry needs to be moved.
            - to_model_part_name (str): The name of the model part to which the geometry needs to be moved.
            - geometry_ids (List[int]): The geometry ids to be moved.
            - new_parameters (Union[:class:`stem.model_part.Material`, :class:`stem.model_part.ProcessParameters`]): \
                The new material or process parameters for the model part.

        Raises:
            - ValueError: If the model part name is not found.
            - ValueError: If the geometry is not defined in the model part.
            - ValueError: If the new parameters are not of the same type as the existing material or process parameters.
            - ValueError: If the geometry is empty in the model part.
        """

        from_model_part = self.get_model_part_by_name(from_model_part_name)
        if from_model_part is None:
            raise ValueError(f"Model part: {from_model_part_name} not found.")
        if from_model_part.geometry is None:
            raise ValueError(f"Geometry is not defined in the model part: {from_model_part_name}.")

        # define type of new model part
        new_model_part: Union[BodyModelPart, ModelPart]

        # create new body model part if from_model_part is a body model part
        if isinstance(from_model_part, BodyModelPart) and isinstance(new_parameters, get_args(Material)):

            # check if the new parameters are of the same type as the existing material
            if not isinstance(new_parameters, from_model_part.material.__class__):
                raise ValueError("New parameters must have the same material type as in the original "
                                 "body model part.")

            # create a new body model part
            new_model_part = BodyModelPart(name=to_model_part_name)
            new_model_part.material = new_parameters  # type: ignore

            self.body_model_parts.append(new_model_part)

        # create new process model part if from_model_part is a process model part
        elif isinstance(from_model_part, ModelPart) and isinstance(new_parameters, get_args(ProcessParameters)):

            # check if the new parameters are of the same type as the existing process parameters
            if not isinstance(new_parameters, from_model_part.parameters.__class__):
                raise ValueError("New parameters must have the same process parameter type as in the original "
                                 "process model part.")

            new_model_part = ModelPart(name=to_model_part_name)
            new_model_part.parameters = new_parameters  # type: ignore
            self.process_model_parts.append(new_model_part)
        else:
            raise ValueError("Model part type and new parameters type must match.")

        # get the geometry from the from-model part
        ndim = self.gmsh_io.geo_data["physical_groups"][from_model_part_name]["ndim"]
        existing_geometry_ids = self.gmsh_io.geo_data["physical_groups"][from_model_part_name]["geometry_ids"]

        # remove the geometry from gmsh physical groups
        self.gmsh_io.geo_data["physical_groups"][from_model_part_name]["geometry_ids"] = \
            [id for id in existing_geometry_ids if id not in geometry_ids]

        # update the geometry in the from-model part
        updated_from_geometry = Geometry.create_geometry_from_gmsh_group(self.gmsh_io.geo_data, from_model_part_name)
        from_model_part.geometry = updated_from_geometry

        # get current max physical group id in gmsh
        max_existing_group_id = max(self.gmsh_io.geo_data["physical_groups"][name]["id"]
                                    for name in self.gmsh_io.geo_data["physical_groups"])

        # add the geometry ids to the new gmsh physical group
        self.gmsh_io.geo_data["physical_groups"][to_model_part_name] = {
            "geometry_ids": geometry_ids,
            "ndim": ndim,
            "id": max_existing_group_id + 1
        }

        # create new geometry and add to new model part
        new_geometry = Geometry.create_geometry_from_gmsh_group(self.gmsh_io.geo_data, to_model_part_name)
        new_model_part.geometry = new_geometry

        # generate the geometry within gmsh
        self.gmsh_io.generate_geo_from_geo_data()
        self.synchronise_geometry()

    def __finalise_json_output(self, input_folder: str):
        """
        Adjust json output for nodal outputs:
          * order of the output nodes should match the order of the given order.
          * include nodal coordinates in the node output to ease the interpretation.

        Args:
            - input_folder (str): input folder for the written files.

        Raises:
            - ValueError: if the parameters of the output settings are None.
            - ValueError: if the output settings has no output name specified.
            - ValueError: if the model part has no geometry.
            - ValueError: if the model part is not yet meshed.
            - IOError: if no JSON output file is found in the specified input folder.
        """

        # reorder json file nodes based on the order of the desired output
        for output_settings in self.output_settings:

            # output settings contain info on the output directory
            if isinstance(output_settings.output_parameters, JsonOutputParameters) and output_settings is not None:

                if output_settings.part_name is None:
                    raise ValueError("The output model part has no part name specified.")

                if output_settings.output_name is None:
                    raise ValueError("No name is specified for the json file.")

                part_name = output_settings.part_name
                # get corresponding model part (info on the geometry and mesh)
                output_model_part = self.get_model_part_by_name(part_name)

                if output_model_part is None:
                    raise ValueError("No model part matches the part name specified in the output settings.")

                if output_model_part.mesh is None:
                    raise ValueError("process model part has not been meshed yet!")

                # get absolute or relative directory of the json file
                if os.path.isabs(output_settings.output_dir):
                    json_file_dir = Path(output_settings.output_dir)
                else:
                    json_file_dir = Path(input_folder) / output_settings.output_dir

                # retrieve the filepath of the json file
                json_file_path = json_file_dir / output_settings.output_name
                json_file_path = json_file_path.with_suffix(".json")

                if not os.path.exists(json_file_path):
                    raise IOError(f"No JSON file is found in the output directory for path: {json_file_path}. "
                                  f"Either the working folder is incorrectly specified or no simulation has been"
                                  f" performed yet.")

                with open(json_file_path, "r") as infile:
                    json_data_tmp = json.load(infile)

                # remove old file
                os.remove(json_file_path)

                # copy the dictionary except for nodal outputs
                new_json = {key: value for key, value in json_data_tmp.items() if "NODE" not in key}

                # adjust the nodal outputs in the right order
                for node_id, node in output_model_part.mesh.nodes.items():
                    node_key = f"NODE_{node_id}"
                    # reassign the corresponding nodal outputs including the nodal coordinates at the top
                    new_json[node_key] = {'COORDINATES': node.coordinates} | json_data_tmp[node_key]

                # write back the json file
                with open(json_file_path, "w") as outfile:
                    json.dump(new_json, outfile, indent=2)

    def finalise(self, input_folder: str):
        """
        Finalise the model run:
        * adjust json output for nodal output coordinates so the order matches the desired one.

        Args:
            - input_folder (str): input folder for the written files.
        """

        # Adjust the order of the json output so it matches the cordinates as the order of the
        # required coordinates
        self.__finalise_json_output(input_folder=input_folder)
