from typing import List, Sequence, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

import numpy as np
from gmsh_utils import gmsh_IO

from stem.model_part import ModelPart, BodyModelPart
from stem.soil_material import *
from stem.structural_material import *
from stem.boundary import *
from stem.geometry import Geometry
from stem.mesh import Mesh, MeshSettings
from stem.load import *
from stem.solver import Problem, StressInitialisationType


class Model:
    """
    A class to represent the main model.

    Attributes:
        project_parameters (dict): A dictionary containing the project parameters.
        solver (Solver): The solver used to solve the problem.
        body_model_parts (list): A list containing the body model parts.
        process_model_parts (list): A list containing the process model parts.

    """
    def __init__(self, ndim: int):
        self.ndim: int = ndim
        self.project_parameters: Optional[Problem] = None
        self.solver = None
        self.geometry: Optional[Geometry] = None
        self.mesh_settings: MeshSettings = MeshSettings()
        self.gmsh_io = gmsh_IO.GmshIO()
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []


    def generate_track(self, sleeper_distance: float, n_sleepers: int, rail_parameters: EulerBeam,
                       sleeper_parameters: NodalConcentrated, rail_pad_parameters: ElasticSpringDamper):
        """
        Generates a track geometry. With rail, rail-pads and sleepers as mass elements.

        Args:
            sleeper_distance (float): distance between sleepers
            n_sleepers (int): number of sleepers

        Returns:

        """

        origin_point = np.array([1, 1, 1])
        direction_vector = np.array([1, 2, 0])

        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

        rotation_matrix = np.diag(normalized_direction_vector)

        rail_length = sleeper_distance * n_sleepers
        rail_end_coords = np.array([origin_point,
                                    origin_point + normalized_direction_vector * rail_length])

        # # rail_end_coords = np.array([rail_end_local_distance, y_local_coords, z_local_coords]).T
        # rail_end_coords = np.array([origin_point, end_global_coordinates]).T

        rail_local_distance = np.linspace(0, sleeper_distance * n_sleepers, n_sleepers + 1)
        sleeper_local_coords = np.copy(rail_local_distance)

        # todo kratos allows for a 0 thickness rail pad height, however gmsh needs to deal with fragmentation,
        # so we add a small height to prevent wrong fragmentation. Investigate the possibility to reset the thickness to
        # zero after the mesh is generated

        rail_pad_height = 0.1

        # todo transfer from local to global coordinates, currently local coordinates are used
        # global rail coordinates

        rail_global_coords = rail_local_distance[:, None].dot(normalized_direction_vector[None, :]) + origin_point

        rail_geo_settings = {"rail": {"coordinates": rail_global_coords, "ndim": 1}}
        self.gmsh_io.generate_geometry(rail_geo_settings, "")


        # global sleeper coordinates
        sleeper_global_coords = sleeper_local_coords[:, None].dot(normalized_direction_vector[None, :]) + origin_point
        # y coord is vertical direction
        vertical_direction = 1
        sleeper_global_coords[:, vertical_direction] -= rail_pad_height

        sleeper_geo_settings = {"sleeper": {"coordinates": sleeper_global_coords, "ndim": 0}}
        self.gmsh_io.generate_geometry(sleeper_geo_settings, "")

        # create rail pad lines
        top_point_ids = self.gmsh_io.make_points(rail_global_coords)
        bot_point_ids = self.gmsh_io.make_points(sleeper_global_coords)

        rail_pad_line_ids = [self.gmsh_io.create_line([top_point_id, bot_point_id])
                             for top_point_id, bot_point_id in zip(top_point_ids, bot_point_ids)]

        self.gmsh_io.add_physical_group("rail_pads", 1, rail_pad_line_ids)


        rail_model_part = BodyModelPart("rail")
        rail_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, "rail")

        sleeper_model_part = BodyModelPart("sleeper")
        sleeper_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, "sleeper")

        rail_pads_model_part = BodyModelPart("rail_pads")
        rail_pads_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, "rail_pads")

        self.body_model_parts.append(rail_model_part)
        self.body_model_parts.append(sleeper_model_part)
        self.body_model_parts.append(rail_pads_model_part)

        a=1+1



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

        geo_data = self.gmsh_io.geo_data

        # Create geometry and model part for each physical group in the gmsh geo_data
        model_part: Union[ModelPart, BodyModelPart]
        for group_name in geo_data["physical_groups"].keys():

            # create model part, if the group name is in the body names, create a body model part, otherwise a process
            # model part
            if group_name in body_names:
                model_part = BodyModelPart(group_name)
            else:
                model_part = ModelPart(group_name)

            # set the name and geometry of the model part
            model_part.get_geometry_from_geo_data(geo_data, group_name)

            # add model part to either body model parts or process model part
            if isinstance(model_part, BodyModelPart):
                self.body_model_parts.append(model_part)
            else:
                self.process_model_parts.append(model_part)

    def add_soil_layer_by_coordinates(self, coordinates: Sequence[Sequence[float]],
                       material_parameters: Union[SoilMaterial, StructuralMaterial], name: str,
                       ):
        """
        Adds a soil layer to the model by giving a sequence of 2D coordinates. In 3D the 2D geometry is extruded in
        the direction of the extrusion_length

        Args:
            - coordinates (Sequence[Sequence[float]]): The plane coordinates of the soil layer.
            - material_parameters (Union[:class:`stem.soil_material.SoilMaterial`, \
                :class:`stem.structural_material.StructuralMaterial`]): The material parameters of the soil layer.
            - name (str): The name of the soil layer.

        """

        gmsh_input = {name: {"coordinates": coordinates, "ndim": self.ndim}}
        # check if extrusion length is specified in 3D
        if self.ndim == 3:
            if self.extrusion_length is None:
                raise ValueError("Extrusion length must be specified for 3D models")

            gmsh_input[name]["extrusion_length"] = self.extrusion_length

        # todo check if this function in gmsh io can be improved
        self.gmsh_io.generate_geometry(gmsh_input, "")

        # create body model part
        body_model_part = BodyModelPart(name)
        body_model_part.material = material_parameters

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        self.body_model_parts.append(body_model_part)

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
            - gravity_load (GravityLoad): The gravity load object.
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

    def __add_gravity_load(self, gravity_value: float = -9.81, vertical_axis: int = 1):
        """
        Add a gravity load to the complete model.

        Args:
            - gravity_value (float): The gravity value [m/s^2]. (default -9.81)
            - vertical_axis (int): The vertical axis of the model. x=>0, y=>1, z=>2. (default y, 1)

        """

        # set gravity load at vertical axis
        gravity_load_values: List[float] = [0, 0, 0]
        gravity_load_values[vertical_axis] = gravity_value
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

    def validate(self):
        """
        Validate the model. \
            - Checks if all model parts have a unique name.

        """

        self.__validate_model_part_names()

    def __setup_stress_initialisation(self):
        """
        Set up the stress initialisation. For K0 procedure and gravity loading, a gravity load is added to the model.

        Raises:
            - ValueError: If the project parameters are not set.

        """

        if self.project_parameters is None:
            raise ValueError("Project parameters must be set before setting up the stress initialisation")

        # add gravity load if K0 procedure or gravity loading is used
        if (self.project_parameters.settings.stress_initialisation_type ==
            StressInitialisationType.K0_PROCEDURE) or \
                (self.project_parameters.settings.stress_initialisation_type ==
                 StressInitialisationType.GRAVITY_LOADING):

            self.__add_gravity_load()

    def post_setup(self):
        """
        Post setup of the model. \
            - Synchronise the geometry. \
            - Generate the mesh. \
            - Validate the model. \
            - Set up the stress initialisation.

        """

        self.synchronise_geometry()
        self.generate_mesh()
        self.validate()

        self.__setup_stress_initialisation()


if __name__ == '__main__':

    # from collections.abc import Sequence
    import numpy as np
    from typing import get_args
    import numpy.typing as npt


    a= np.array([1,2,3,4])
    b=np.array([1,2,6,4])

    np.testing.assert_array_equal(a,b)
    c = a==b

    d=1+1


    def a_test(a: Union[Sequence[Sequence[int]], npt.NDArray[np.int64]]):

        print(a)

        b=1+1
        pass


    c = a_test([[1.1,2,3],[4,5,6]])

    # npt.NDArray[np.float64, np.int64]
    #
    # npt.NDArray[np.int64]

    class A:
        def __init__(self):
            self.a = 1

    class B:
        def __init__(self):
            self.b = 2

    # c: Union[A, B]


    a1 = A()
    b1 = B()


    # tmp = isinstance(a1, Union[A, B])
    tmp2 = isinstance(a1, (A, B))
    # tmp3 = (isinstance(a1, A) or isinstance(a1, B))

    # a = np.array([1,2,3])
    a= ([1,2,3],[4,5,6])

    # test = isinstance(a, (float))


    b=1+1
    # coords = np.array([(0,0,"abs"), (1,1,1)], dtype=float)

    # assert coords.shape[1] == 3

    #
    # model = Model(2)
    #
    # rail_parameters = EulerBeam(2, 1, 1, 1, 1 ,1)
    # rail_pad_parameters = ElasticSpringDamper([1,1,1], [1,1,1], [1,1,1], [1,1,1])
    # sleeper_parameters = NodalConcentrated([1,1,1], 1, [1,1,1])
    #
    # rail_nodes = model.generate_track(0.6, 10)
    #
    #
    #
    # print(rail_nodes)

