from stem.IO.kratos_io import KratosIO

from stem.boundary import Boundary, DisplacementConstraint
from stem.load import LineLoad, Load
from stem.model_part import *
from stem.output import Output, GiDOutputParameters, NodalOutput, GaussPointOutput
from stem.soil_material import (
    SoilMaterial,
    LinearElasticSoil,
    TwoPhaseSoil,
    SaturatedBelowPhreaticLevelLaw,
    FluidProperties,
)


def geometry_set_list_from_dictionary(
    geometry_set_dic: List[dict],
    nodes_collection: Dict[int, Node],
    geometry_class,
):
    _geometry_sets = []
    _geometries = {}
    for _set in geometry_set_dic:
        _geometries_in_set = []

        for _id, (property_id, node_ids) in _set["items"].items():
            nodes_in_geo = [nodes_collection[_id] for _id in node_ids]
            _geometries_in_set.append(
                geometry_class(_id, property_id, nodes_in_geo)
            )
            _geometries[_id] = geometry_class(_id, property_id, nodes_in_geo)

        set_type = "Condition"
        if isinstance(geometry_class, Element):
            set_type = "Element"
        _geometry_sets.append(
            GeometrySet(set_type=set_type, name=_set["key"], items=_geometries_in_set)
        )
    return _geometries, _geometry_sets


# define properties
properties = {1: None}

# -------------------------------------------------------------------------------------
# define nodes, elements and conditions (or get them from gmesh!)

nodes_dict = {
    1: (0.0, 1.00, 0.0),
    2: (1.0, 1.00, 0.0),
    3: (0.0, 0.00, 0.0),
    4: (1.0, 0.00, 0.0),
}

element_sets = [
    {
        "key": "UPwSmallStrainElement2D3N",
        "items": {
            1: [1, (3, 4, 2)],
            2: [1, (2, 1, 3)],
        },
    }
]

condition_sets = [
    {
        "key": "UPwFaceLoadCondition2D2N",
        "items": {1: [1, (2, 1)]},
    }
]


nodes = {_id: Node(id=_id, coordinates=coords) for _id, coords in nodes_dict.items()}

# assemble the element sets and condition set
elements, geometry_element_sets = geometry_set_list_from_dictionary(
    geometry_set_dic=element_sets, nodes_collection=nodes, geometry_class=Element
)

conditions, geometry_condition_sets = geometry_set_list_from_dictionary(
    geometry_set_dic=condition_sets, nodes_collection=nodes, geometry_class=Condition
)

# -------------------------------------------------------------------------------------
# PART DEFINITIONS AND OUTPUTS
# -------------------------------------------------------------------------------------
# Loads
modelpart_lineload = ModelPart(
    name="Line_Load-auto-1", nodes=[nodes[1], nodes[2]], conditions=[conditions[1]]
)
line_load = Load(
    part_name=modelpart_lineload.name,
    load_parameters=LineLoad(active=[True, True, True], value=[0, -10, 0]),
)

# -------------------------------------------------------------------------------------
# Constraints
modelpart_supports = ModelPart(
    name="Solid_Displacement-auto-1", nodes=[nodes[3], nodes[4]]
)
supports = Boundary(
    part_name=modelpart_supports.name,
    boundary_parameters=DisplacementConstraint(
        active=[True, True, True], is_fixed=[True, True, True], value=[0, 0, 0]
    ),
)
# -------------------------------------------------------------------------------------
# Soil volume
modelpart_soil = ModelPart(
    name="Soil_drained-auto-1",
    nodes=[nodes[1], nodes[2], nodes[3], nodes[4]],
    elements=[elements[1], elements[2]],
)

# Materials

# Create materials
elastic_soil_material = SoilMaterial(
    name=f"PorousDomain.{modelpart_soil.name}",
    soil_formulation=TwoPhaseSoil(
        ndim=2,
        DENSITY_SOLID=2.650,
        POROSITY=0.3,
        BULK_MODULUS_SOLID=1e9,
        PERMEABILITY_XX=4.5e-30,
        PERMEABILITY_YY=4.5e-30,
        PERMEABILITY_XY=0.0,
        BIOT_COEFFICIENT=1.0,
    ),
    constitutive_law=LinearElasticSoil(YOUNG_MODULUS=1e4, POISSON_RATIO=0.2),
    retention_parameters=SaturatedBelowPhreaticLevelLaw(),
    fluid_properties=FluidProperties(
        DENSITY_FLUID=1.000,
        DYNAMIC_VISCOSITY=8.9e-7,
        BULK_MODULUS_FLUID=2.0e-30,
    ),
)


# -------------------------------------------------------------------------------------
# Outputs
# Nodal results
nodal_results = [
    NodalOutput.DISPLACEMENT,
    NodalOutput.TOTAL_DISPLACEMENT,
    NodalOutput.WATER_PRESSURE,
    NodalOutput.VOLUME_ACCELERATION,
]
# Gauss point results
gauss_point_results = [
    GaussPointOutput.VON_MISES_STRESS,
    GaussPointOutput.FLUID_FLUX_VECTOR,
    GaussPointOutput.HYDRAULIC_HEAD,
    GaussPointOutput.GREEN_LAGRANGE_STRAIN_TENSOR,
    GaussPointOutput.ENGINEERING_STRAIN_TENSOR,
    GaussPointOutput.CAUCHY_STRESS_TENSOR,
    GaussPointOutput.TOTAL_STRESS_TENSOR,
]
# define output parameters
gid_output = Output(
    part_name=modelpart_soil.name,
    output_dir="dir_test",
    output_name="test_gid_output",
    output_parameters=GiDOutputParameters(
        file_format="binary",
        output_interval=1,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
    ),
)

# -------------------------------------------------------------------------------------
# Assemble parts together and write input files
# mdpa file
part_collection = PartCollection(
    nodes=list(nodes.values()),
    element_sets=geometry_element_sets,
    condition_sets=geometry_condition_sets,
    model_parts=[modelpart_lineload, modelpart_supports, modelpart_soil],
    properties=properties,
)

mdpa_text = part_collection.write_mdpa(
    ind=2, output_name="test_mdpa_file.mdpa", output_dir="run_stem"
)
# -------------------------------------------------------------------------------------
# MaterialsParameters.json

kratos_io = KratosIO(ndim=2)
kratos_io.write_material_parameters_json(
    [elastic_soil_material], "test_MaterialParameters_file.json"
)

# -------------------------------------------------------------------------------------
# ProjectParameters.json
kratos_io.write_project_parameters_json(
    boundaries=[supports],
    loads=[line_load],
    outputs=[gid_output],
    filename="test_ProjectParameters_file.json",
)
