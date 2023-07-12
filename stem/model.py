from typing import List, Sequence, Dict, Any, Optional, Union

from gmsh_utils import gmsh_IO

from stem.model_part import ModelPart, BodyModelPart
from stem.soil_material import *
from stem.structural_material import *
from stem.geometry import Geometry


class Model:
    """
    A class to represent the main model.

    Attributes:
        - ndim (int): Number of dimensions of the model
        - project_parameters (dict): A dictionary containing the project parameters.
        - solver (Solver): The solver used to solve the problem.
        - geometry (Optional[:class:`stem.geometry.Geometry`]) The geometry of the whole model.
        - body_model_parts (List[BodyModelPart]): A list containing the body model parts.
        - process_model_parts (List[ModelPart]): A list containing the process model parts.
        - extrusion_length(Optional[Sequence[float]]): The extrusion length in x,y and z direction

    """
    def __init__(self, ndim: int):
        self.ndim: int = ndim
        self.project_parameters = None
        self.solver = None
        self.geometry: Optional[Geometry] = None
        self.mesh = None
        self.gmsh_io = gmsh_IO.GmshIO()
        self.body_model_parts: List[BodyModelPart] = []
        self.process_model_parts: List[ModelPart] = []

        self.extrusion_length: Optional[Sequence[float]] = None

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
                model_part = BodyModelPart()
            else:
                model_part = ModelPart()

            # set the name and geometry of the model part
            model_part.name = group_name
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
            - coordinates (Sequence[Sequence[float]]): The coordinates of the soil layer.
            - material_parameters (Union[:class:`stem.soil_material.SoilMaterial`, \
                :class:`stem.structural_material.StructuralMaterial`]): The material parameters of the soil layer.
            - name (str): The name of the soil layer.

        """

        # check if extrusion length is specified in 3D
        if self.ndim == 3:
            if self.extrusion_length is None:
                raise ValueError("Extrusion length must be specified for 3D models")
            else:
                extrusion_length = self.extrusion_length
        else:
            # in 2D extrusion length is not needed
            extrusion_length = [0, 0, 0]

        # todo check if this function in gmsh io can be improved
        self.gmsh_io.generate_geometry([coordinates], extrusion_length, self.ndim, "", [name])

        # create body model part
        body_model_part = BodyModelPart()
        body_model_part.name = name
        body_model_part.material = material_parameters

        # set the geometry of the body model part
        body_model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, name)

        self.body_model_parts.append(body_model_part)

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
            # Check if all model parts have a name
            if model_part.name is None:
                raise ValueError("All model parts must have a name")
            else:
                model_part.get_geometry_from_geo_data(self.gmsh_io.geo_data, model_part.name)

        # get the complete geometry
        self.__get_geometry_from_geo_data(self.gmsh_io.geo_data)






