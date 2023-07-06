from typing import Optional, Union

from stem.soil_material import SoilMaterial
from stem.structural_material import StructuralMaterial

from stem.geometry import Geometry, Volume, Surface, Line, Point

class ModelPart:
    """
    One part of the complete model, this can be a boundary condition, a loading or another special process
    like excavation.

    Attributes:
        - name (str): name of the model part
        - nodes (np.array or None): node id followed by node coordinates in an array
        - elements (np.array or None): element id followed by connectivities in an array
        - conditions (np.array or None): condition id followed by connectivities in an array
        - parameters (dict): dictionary containing the model part parameters
    """
    def __init__(self):
        self.name = None
        self.nodes = None
        self.elements = None
        self.conditions = None

        self.geometry: Geometry = Geometry()
        self.parameters = {}

    def get_geometry_from_geo_data(self, geo_data, name):
        """
        Get the geometry from the geo_data and set the nodes and elements attributes.

        Args:
            - geo_data (dict): dictionary containing the geometry data

        """

        group_data = geo_data["physical_groups"][name]
        ndim_group = group_data["ndim"]

        if ndim_group == 3:

            for id in group_data["geometry_id"]:
                volume = Volume()
                volume.id = id
                volume.surface_ids = geo_data["volumes"][volume.id]

            self.geometry.volumes = geo_data["volumes"][group_data["geometry_id"]]
            # self.geometry.surfaces =

        geo_data["points"] = {k: v for k, v in geo_data["points"].items() if k in group_data["points"]}





class BodyModelPart(ModelPart):
    """
    This class contains model parts which are part of the body, e.g. a soil layer or track components.

        Inheritance:
        - :class:`ModelPart`

    Attributes:
        - name (str): name of the model part
        - nodes (np.array or None): node id followed by node coordinates in an array
        - elements (np.array or None): element id followed by connectivities in an array
        - conditions (np.array or None): condition id followed by connectivities in an array
        - parameters (dict): dictionary containing the model part parameters
        - material (Union[:class:`stem.soil_material.SoilMaterial`, \
            :class:`stem.structural_material.StructuralMaterial`]): material of the model part
    """

    def __init__(self):
        super().__init__()

        self.material: Optional[Union[SoilMaterial, StructuralMaterial]] = None
