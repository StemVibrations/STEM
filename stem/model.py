from typing import List, Sequence, Dict, Any, Optional, Union

import numpy.typing as npty
import numpy as np

from gmsh_utils import gmsh_IO

from stem.model_part import ModelPart, BodyModelPart
from stem.soil_material import *
from stem.structural_material import *
from stem.boundary import *
from stem.geometry import Geometry
from stem.mesh import Mesh, MeshSettings, Node, Element
from stem.load import *
from stem.solver import Problem, StressInitialisationType
from stem.output import Output
from stem.utils import Utils
from stem.plot_utils import PlotUtils
from stem.globals import ELEMENT_DATA, VERTICAL_AXIS, GRAVITY_VALUE,  OUT_OF_PLANE_AXIS_2D


NUMBER_TYPES = (int, float, np.int64, np.float64)


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

    def generate_straight_track(self, sleeper_distance: float, n_sleepers: int, rail_parameters: EulerBeam,
                       sleeper_parameters: NodalConcentrated, rail_pad_parameters: ElasticSpringDamper,
                                origin_point: Sequence[float], direction_vector: Sequence[float], name):
        """
        Generates a track geometry. With rail, rail-pads and sleepers as mass elements.

        Args:
            - sleeper_distance (float): distance between sleepers
            - n_sleepers (int): number of sleepers
            - rail_parameters (:class:`stem.structural_material.EulerBeam`): rail parameters
            - sleeper_parameters (:class:`stem.structural_material.NodalConcentrated`): sleeper parameters
            - rail_pad_parameters (:class:`stem.structural_material.ElasticSpringDamper`): rail pad parameters
            - origin_point (Sequence[float]): origin point of the track
            - direction_vector (Sequence[float]): direction vector of the track

        Returns:
            - np.ndarray: coordinates of the sleepers, i.e. the bottom coordinates of the track

        """

        rail_name = f"rail_{name}"
        sleeper_name = f"sleeper_{name}"
        rail_pads_name = f"rail_pads_{name}"

        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

        rail_local_distance = np.linspace(0, sleeper_distance * (n_sleepers - 1), n_sleepers)
        sleeper_local_coords = np.copy(rail_local_distance)

        # # todo kratos allows for a 0 thickness rail pad height, however gmsh needs to deal with fragmentation,
        # # so we add a small height to prevent wrong fragmentation. Investigate the possibility to reset the thickness to
        # # zero after the mesh is generated
        # rail_pad_height = TEMP_ZERO_THICKNESS

        # set rail geometry
        rail_global_coords = rail_local_distance[:, None].dot(normalized_direction_vector[None, :]) + origin_point
        rail_global_coords[:, VERTICAL_AXIS] += TEMP_ZERO_THICKNESS

        rail_geo_settings = {rail_name: {"coordinates": rail_global_coords, "ndim": 1}}
        self.gmsh_io.generate_geometry(rail_geo_settings, "")

        # set sleepers geometry
        sleeper_global_coords = sleeper_local_coords[:, None].dot(normalized_direction_vector[None, :]) + origin_point

        sleeper_geo_settings = {sleeper_name: {"coordinates": sleeper_global_coords, "ndim": 0}}
        self.gmsh_io.generate_geometry(sleeper_geo_settings, "")

        # create rail pad geometries
        top_point_ids = self.gmsh_io.make_points(rail_global_coords)
        bot_point_ids = self.gmsh_io.make_points(sleeper_global_coords)

        rail_pad_line_ids = [self.gmsh_io.create_line([top_point_id, bot_point_id])
                             for top_point_id, bot_point_id in zip(top_point_ids, bot_point_ids)]

        self.gmsh_io.add_physical_group(rail_pads_name, 1, rail_pad_line_ids)

        # create rail, sleeper, and rail_pad body model parts
        rail_model_part = BodyModelPart(rail_name)
        rail_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rail_name)
        rail_model_part.material = rail_parameters
        rail_model_part._is_shifted = True

        sleeper_model_part = BodyModelPart(sleeper_name)
        sleeper_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, sleeper_name)
        sleeper_model_part.material = sleeper_parameters

        rail_pads_model_part = BodyModelPart(rail_pads_name)
        rail_pads_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rail_pads_name)
        rail_pads_model_part.material = rail_pad_parameters

        # add physical group to gmsh
        rail_constraint_name = f"constraint_{rail_name}"
        rail_constraint_geometry_ids = self.gmsh_io.geo_data["physical_groups"][rail_name]["geometry_ids"]
        self.gmsh_io.add_physical_group(f"constraint_{rail_name}", 1, rail_constraint_geometry_ids)

        # create model part
        model_part = ModelPart(rail_constraint_name)

        # retrieve geometry from gmsh and add to model part
        model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, rail_constraint_name)

        # add displacement_constraint in x and z direction
        model_part.parameters = DisplacementConstraint(active=[True, True, True],  is_fixed=[True, False, True],
                                                       value=[0, 0, 0])

        self.body_model_parts.append(rail_model_part)
        self.body_model_parts.append(sleeper_model_part)
        self.body_model_parts.append(rail_pads_model_part)

        self.process_model_parts.append(model_part)



        return sleeper_global_coords

    def __del__(self):
        """
        Destructor of the Model class. Finalizes the gmsh_io instance.

        """
        self.gmsh_io.finalize_gmsh()

    def __get_geometry_from_geo_data(self, geo_data: Dict[str, Any]):
        """
        Get the geometry from the geo_data as generated by gmsh_io.

        Args:
            - geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io

        """

        self.geometry = Geometry.create_geometry_from_geo_data(geo_data)

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

    def add_soil_layer_by_coordinates(self, coordinates: Sequence[Sequence[float]],
                                      material_parameters: Union[SoilMaterial, StructuralMaterial], name: str):
        """
        Adds a soil layer to the model by giving a sequence of 2D coordinates. In 3D the 2D geometry is extruded in
        the out of plane direction.

        Args:
            - coordinates (Sequence[Sequence[float]]): The plane coordinates of the soil layer.
            - material_parameters (Union[:class:`stem.soil_material.SoilMaterial`, \
                :class:`stem.structural_material.StructuralMaterial`]): The material parameters of the soil layer.
            - name (str): The name of the soil layer.

        Raises:
            - ValueError: if extrusion_length is not specified in 3D.
        """

        # sort coordinates in anti-clockwise order, such that elements in mesh are also in anti-clockwise order
        if Utils.are_2d_coordinates_clockwise(coordinates):
            coordinates = coordinates[::-1]

        gmsh_input = {name: {"coordinates": coordinates, "ndim": self.ndim}}
        # check if extrusion length is specified in 3D
        if self.ndim == 3:
            if self.extrusion_length is None:
                raise ValueError("Extrusion length must be specified for 3D models")

            extrusion_length: List[float] = [0, 0, 0]
            extrusion_length[OUT_OF_PLANE_AXIS_2D] = self.extrusion_length
            gmsh_input[name]["extrusion_length"] = extrusion_length

        # todo check if this function in gmsh io can be improved
        self.gmsh_io.generate_geometry(gmsh_input, "")

        # create body model part
        body_model_part = BodyModelPart(name)
        body_model_part.material = material_parameters

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        self.body_model_parts.append(body_model_part)

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

        # todo add validation that load is applied on a body model part

        # validation of inputs
        self.validate_coordinates(coordinates)
        if isinstance(load_parameters, MovingLoad):
            self.__validate_moving_load_parameters(coordinates, load_parameters)

        # create input for gmsh
        if isinstance(load_parameters, PointLoad):
            gmsh_input = {name: {"coordinates": coordinates, "ndim": 0}}
        elif isinstance(load_parameters, LineLoad) or isinstance(load_parameters, MovingLoad):
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

        self.process_model_parts.append(model_part)

    @staticmethod
    def validate_coordinates(coordinates: Union[Sequence[Sequence[float]], npty.NDArray[np.float64]]):
        """
        Validates the coordinates in input.

        Args:
            - coordinates (Sequence[Sequence[float]]): The coordinates of the load.

        Raises:
            - ValueError: if coordinates is not a sequence real numbers.
            - ValueError: if coordinates is not convertible to a 2D array (i.e. a sequence of sequences)
            - ValueError: if the number of elements (number of coordinates) is not 3.
        """

        # if is not an array, make it array!
        if not isinstance(coordinates, np.ndarray):
            coordinates = np.array(coordinates, dtype=np.float64)

        if len(coordinates.shape) != 2:
            raise ValueError(f"Coordinates are not a sequence of a sequence or a 2D array.")

        if coordinates.shape[1] != 3:
            raise ValueError(f"Coordinates should be 3D but {coordinates.shape[1]} coordinates were given.")

        # check if coordinates are real numbers
        for coordinate in coordinates:
            for i in coordinate:
                if not isinstance(i, NUMBER_TYPES) or np.isnan(i) or np.isinf(i):
                    raise ValueError(f"Coordinates should be a sequence of sequence of real numbers, "
                                     f"but {i} was given.")

    @staticmethod
    def __validate_moving_load_parameters(coordinates: Sequence[Sequence[float]], load_parameters: MovingLoad) -> None:
        """
        Validates the coordinates in input for the moving load and the trajectory (collinearity of the
        points and if the origin is between the point).

        Args:
            - coordinates (Sequence[Sequence[float]]): The start-end coordinate of the moving load.
            - parameters (:class:`stem.load.LoadParametersABC`): The parameters of the load.

        Raises:
            - ValueError: if moving load origin is not on trajectory

        Returns:
            - None
        """

        # iterate over each line constituting the trajectory
        for ix in range(len(coordinates) - 1):
            # check origin is collinear to the edges of the line
            collinear_check = Utils.is_collinear(
                point=load_parameters.origin, start_point=coordinates[ix], end_point=coordinates[ix + 1]
            )
            # check origin is between the edges of the line (edges included)
            is_between_check = Utils.is_point_between_points(
                point=load_parameters.origin, start_point=coordinates[ix], end_point=coordinates[ix + 1]
            )
            # check if point complies
            is_on_line = collinear_check and is_between_check
            # exit at the first success of the test (point in the line)
            if is_on_line:
                return

        # none of the lines contain the origin, then raise an error
        raise ValueError(f"Origin is not in the trajectory of the moving load.")

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

        self.process_model_parts.append(model_part)

    def synchronise_geometry(self):
        """
        Synchronise the geometry of all model parts and synchronise the geometry of the whole model. This function
        recalculates all ids and connectivities of all geometrical entities.

        """

        # synchronize gmsh and extract geo data
        self.gmsh_io.synchronize_gmsh()
        self.gmsh_io.extract_geo_data()

        # collect all model parts
        all_model_parts: List[Union[BodyModelPart, ModelPart]] = []
        all_model_parts.extend(self.body_model_parts)
        all_model_parts.extend(self.process_model_parts)

        # Get the geometry from the geo_data for each model part
        for model_part in all_model_parts:
            model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, model_part.name)

        # get the complete geometry
        self.__get_geometry_from_geo_data(self.gmsh_io.geo_data)

    def set_mesh_size(self, element_size: float):
        """
        Set the element size to dimension [m].

        Args:
            - element_size (float): the desired element size [m].
        """
        self.mesh_settings.element_size = element_size

    def generate_mesh(self):
        """
        Generate the mesh for the whole model.

        """

        # generate mesh
        self.gmsh_io.generate_mesh(self.ndim, element_size=self.mesh_settings.element_size,
                                   order=self.mesh_settings.element_order)

        # collect all model parts
        all_model_parts: List[Union[BodyModelPart, ModelPart]] = []
        all_model_parts.extend(self.body_model_parts)
        all_model_parts.extend(self.process_model_parts)

        # add the mesh to each model part
        for model_part in all_model_parts:
            model_part.mesh = Mesh.create_mesh_from_gmsh_group(self.gmsh_io.mesh_data, model_part.name)

        # per process model part, check if the condition elements are applied to a body model part and set the
        # node ordering of the condition elements to match the body elements
        for process_model_part in self.process_model_parts:

            # only check if the process model part is a condition element
            if isinstance(process_model_part.parameters, (LineLoad, MovingLoad, SurfaceLoad, AbsorbingBoundary)):
                # match the condition elements with the body elements on which the conditions are applied
                matched_elements = self.__find_matching_body_elements_for_process_model_part(process_model_part)

                # check the ordering of the nodes of the conditions. If it does not match flip the order.
                self.__check_ordering_process_model_part(matched_elements, process_model_part)

    @staticmethod
    def __get_model_part_element_connectivities(model_part: ModelPart) -> npty.NDArray[np.int64]:
        """
        Extract the node ids of each of the elements in a model part.

        Args:
            - model_part (:class:`stem.model_part.ModelPart`): model part from which element nodes needs to be
                extracted.

        Returns:
            - npty.NDArray[np.int64]: array containing the node ids of the elements in the model_part
        """
        if model_part.mesh is not None:
            return np.array([el.node_ids for el in model_part.mesh.elements.values()])
        else:
            return np.array([])

    def __find_matching_body_elements_for_process_model_part(self, process_model_part: ModelPart) \
            -> Dict[Element, Element]:
        """
        For a process model part, tries finds the matching body elements on which the condition elements are applied.

        Args:
            - process_model_part (:class:`stem.model_part.ModelPart`): model part from which element nodes needs to be \
                extracted.
        Raises:
            - ValueError: if mesh is not initialised yet.
            - ValueError: if condition elements don't have a corresponding body element.

        Returns:
            - matched_elements (Dict[:class:`stem.mesh.Element`, :class:`stem.mesh.Element`]): Dictionary containing
                the matched condition and body element parts.
        """
        # validation step for process model part
        if process_model_part.mesh is None:
            raise ValueError(f"Mesh of process model part: {process_model_part.name} is not yet initialised.")

        # get all the node ids for all the elements in the process model (pmp) part and the indices of each element in
        # the array
        unmatched_connectivities_pmp = self.__get_model_part_element_connectivities(process_model_part)
        pmp_element_ids = np.array(list(process_model_part.mesh.elements.keys()))

        # initialise matching dictionary: process_element --> body_element
        matched_elements: Dict[Element, Element] = {}

        # loop over the body model parts (bmp) to match the elements of the process model part
        for body_model_part in self.body_model_parts:

            # validation step for body model part
            if body_model_part.mesh is None:
                raise ValueError(f"Mesh of body model part: {body_model_part.name} is not yet initialised.")

            # if there is nothing to match, break the loop
            if len(unmatched_connectivities_pmp) == 0:
                # finished matching elements
                break

            # get the node ids for the elements in the current body model part and their ids
            bmp_connectivities = self.__get_model_part_element_connectivities(body_model_part)
            bmp_element_ids = np.array(list(body_model_part.mesh.elements.keys()))

            # initialised matched ids and indices for the element of the process model part
            matched_element_id_process_to_body = {}
            matched_indices_process_element = []
            # for each process element, check if there is a match with the current body part elements
            for ix, (process_element_id, process_element_connectivities) in (
                    enumerate(zip(pmp_element_ids, unmatched_connectivities_pmp))):
                # find the indices of the element in the body model parts that contains the node ids of the current
                # process model part. An element is considered a match if all the nodes of the process element are also
                # in the body element
                found_indices = np.where(np.sum(np.isin(bmp_connectivities, process_element_connectivities), axis=1) ==
                                         len(process_element_connectivities))[0]

                # from the first match, retrieve the element id of the body model part and the element id of the process
                # model part
                if len(found_indices) > 0:
                    matched_element_id_process_to_body[process_element_id] = bmp_element_ids[found_indices.tolist()[0]]
                    matched_indices_process_element.append(ix)

            # if there is match, couple the element objects together in the matched_elements dictionary
            # then remove the matched process model part elements from the unmatched_connectivities_pmp array
            # and the pmp_element_ids array in order to avoid matching the same elements twice
            if len(matched_element_id_process_to_body) > 0:

                for process_element_id, body_element_id in matched_element_id_process_to_body.items():
                    matched_elements[process_model_part.mesh.elements[process_element_id]] = (
                        body_model_part.mesh.elements)[body_element_id]

                # remove the matched elements from the unmatched_elements_pmp and pmp_element_ids arrays, in order
                # to avoid matching the same elements twice
                process_elements_idxs = np.array(list(matched_indices_process_element))
                unmatched_connectivities_pmp = np.delete(unmatched_connectivities_pmp, process_elements_idxs, axis=0)
                pmp_element_ids = np.delete(pmp_element_ids, process_elements_idxs)

        # if there are still process elements which do not share the nodes of body elements, raise an error
        if len(unmatched_connectivities_pmp) != 0:
            raise ValueError(f"In process model part: {process_model_part.name}, the node ids: "
                             f"{list(unmatched_connectivities_pmp)}, are not present in a body model part.")

        return matched_elements

    def __check_ordering_process_model_part(self, matched_elements: Dict[Element, Element],
                                            process_model_part: ModelPart):
        """
        Check if the node ordering of the process element matches the node ordering of the neighbouring body element.
        If not, flip the node ordering of the process element.

        Args:
            - matched_elements (Dict[:class:`stem.mesh.Element`, :class:`stem.mesh.Element`]): Dictionary containing \
                the matched condition and body element parts.
            - process_model_part (:class:`stem.model_part.ModelPart`): model part from which element nodes needs to be \
                extracted.

        Raises:
            - ValueError: if mesh is not initialised yet.
            - ValueError: if the integration order of the process element is different from the body element.
        """

        if process_model_part.mesh is None:
            raise ValueError(f"Mesh of process model part: {process_model_part.name} is not yet initialised.")

        # loop over the matched elements
        flip_node_order = np.zeros(len(matched_elements), dtype=bool)

        for i, (process_element, body_element) in enumerate(matched_elements.items()):

            # element info such as order, number of edges, element types etc.
            process_el_info = ELEMENT_DATA[process_element.element_type]
            body_el_info = ELEMENT_DATA[body_element.element_type]

            if process_el_info["ndim"] == 1:

                # get all line edges of the body element and check if the process element is defined on one of them
                # if the nodes are equal, but the node order isn't, flip the node order of the process element
                body_line_edges = Utils.get_element_edges(body_element)
                for edge in body_line_edges:
                    if set(edge) == set(process_element.node_ids):
                        if list(edge) != process_element.node_ids:
                            flip_node_order[i] = True

            elif body_el_info["ndim"] == 3 and process_el_info["ndim"] == 2:

                # check if the normal of the condition element is defined outwards of the body element
                flip_node_order[i] = Utils.is_volume_edge_defined_outwards(process_element, body_element,
                                                                           self.gmsh_io.mesh_data["nodes"])

        # flip condition elements if required
        if any(flip_node_order):
            # elements to be flipped
            elements = np.array(list(process_model_part.mesh.elements.values()))[flip_node_order]

            # flip elements, it is required that all elements in the array are of the same type
            Utils.flip_node_order(elements)

    def __validate_model_part_names(self):
        """
        Checks if all model parts have a unique name.

        Raises:
            - ValueError: If not all model parts have a name.
            - ValueError: If not all model part names are unique .
        """

        # collect all model parts
        all_model_parts: List[Union[BodyModelPart, ModelPart]] = []
        all_model_parts.extend(self.body_model_parts)
        all_model_parts.extend(self.process_model_parts)

        unique_names = []
        for model_part in all_model_parts:
            # Check if all model parts have a name
            if model_part.name is None:
                raise ValueError("All model parts must have a name")
            else:
                if model_part.name in unique_names:
                    raise ValueError("All model parts must have a unique name")
                unique_names.append(model_part.name)

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

        # add gravity load to process model parts
        self.process_model_parts.append(model_part)

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
        model_parts_geometry_ids = np.array([self.gmsh_io.geo_data["physical_groups"][name]["geometry_ids"] for name in
                                    body_model_part_names])

        model_parts_ndim = np.array([self.gmsh_io.geo_data["physical_groups"][name]["ndim"]
                                     for name in body_model_part_names]).ravel()

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

    def get_all_model_parts(self) -> List[Union[BodyModelPart, ModelPart]]:
        """
        Returns both body and process model parts in the model.

        Returns:
            - all_model_parts (List[:class:`stem.model_part.ModelPart`]): list of all the model parts.
        """
        all_model_parts = []
        all_model_parts.extend(self.process_model_parts)
        all_model_parts.extend(self.body_model_parts)
        return all_model_parts

    def get_all_nodes(self):
        """
        Retrieve all the unique nodes in the model mesh.

        Raises:
            - ValueError: If the geometry has not been meshed yet.

        Returns:
            - node_dict (Dict[int, :class:`stem.mesh.Node`]): dictionary containing nodes id and nodes objects.
        """

        node_dict: Dict[int, Node] = {}
        for mp in self.get_all_model_parts():
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

    def show_geometry(self, show_volume_ids: bool = False, show_surface_ids: bool = False, show_line_ids: bool = False,
                      show_point_ids: bool = False, file_name: str = "tmp_geometry_file.html", auto_open: bool = True):
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
        if self.geometry is None:
            raise ValueError("Geometry must be set before showing the geometry")

        fig = PlotUtils.create_geometry_figure(self.ndim, self.geometry, show_volume_ids, show_surface_ids, show_line_ids,
                                               show_point_ids)

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
        if (self.project_parameters.settings.stress_initialisation_type == StressInitialisationType.K0_PROCEDURE) or (
            self.project_parameters.settings.stress_initialisation_type == StressInitialisationType.GRAVITY_LOADING):
            self.__add_gravity_load()

    def post_setup(self):
        """
        Post setup of the model.
            - Synchronise the geometry.
            - Generate the mesh.
            - Validate the model.
            - Set up the stress initialisation.

        """

        self.synchronise_geometry()
        self.validate()

        self.__setup_stress_initialisation()

        # finalize gmsh
        self.gmsh_io.finalize_gmsh()
