import json
import os
import sys

from stem.IO.kratos_io import KratosIO
from stem.default_materials import DefaultMaterial
from stem.mesh import Mesh
from stem.model import Model
from stem.model_part import *
from stem.output import Output, GiDOutputParameters, NodalOutput, GaussPointOutput
from stem.utils import add_solver_settings_to_project_parameters


def create_soil_layer(points, material) -> BodyModelPart:
    pass


def gmsh_to_kratos(gmsh_key: str):
    mapper = {
        "tri6": "UPwSmallStrainElement2D3N",
        "lineload2D": "UPwFaceLoadCondition2D2N",
        "lineload3D": "UPwFaceLoadCondition3D2N",
    }
    try:
        return mapper[gmsh_key]
    except KeyError:
        raise (KeyError, f"Key `{gmsh_key}`not implented in the gmsh->KRATOS mapper!")


# IMPORTANT !!!! SPECIFY LOCAL PATH TO KRATOS...
pth_kratos = r"C:\Users\morettid\OneDrive - TNO\Desktop\projects\STEM"

# --------------------------------------------------------------------------------------------------------------------
sys.path.append(os.path.join(pth_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(pth_kratos, r"KratosGeoMechanics\libs"))

import KratosMultiphysics.GeoMechanicsApplication
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (
    GeoMechanicsAnalysis,
)
# --------------------------------------------------------------------------------------------------------------------

# define properties
properties = {1: None}

# -------------------------------------------------------------------------------------
# define nodes, elements and conditions (or get them from gmesh!)
# where is the info on the material?
gmsh_info = dict(
    nodes={
        1: (0.0, 1.00, 0.0),
        2: (1.0, 1.00, 0.0),
        3: (0.0, 0.00, 0.0),
        4: (1.0, 0.00, 0.0),
    },
    # elements can also have connectivity in here
    elements={
        1: dict(type="tri6", connectivity=[3, 4, 2]),
        2: dict(type="tri6", connectivity=[2, 1, 3]),
        3: dict(type="lineload2D", connectivity=[2, 1]),
    },
    physical_groups={
        "Solid_Displacement-auto-1": dict(node_ids=[1, 2]),
        "Line_Load-auto-1": dict(node_ids=[1, 2], element_ids=[3]),
        "Soil_drained-auto-1": dict(node_ids=[1, 2, 3, 4], element_ids=[1, 2]),
    },
)

# re-adjust the gmsh element labels to Kratos! Later in a more elegant way....
for _el_id, _el_properties in gmsh_info["elements"].items():
    if _el_properties["type"] is not None:
        # thus not a condition...
        gmsh_info["elements"][_el_id]["type"] = gmsh_to_kratos(_el_properties["type"])

mesh = Mesh.read_mesh_from_dictionary(gmsh_info)


# -------------------------------------------------------------------------------------
# PART DEFINITIONS AND OUTPUTS
# -------------------------------------------------------------------------------------
# Loads
mp_line_load = ModelPart.from_physical_group_and_parameters(
    physical_group=mesh.physical_groups["Line_Load-auto-1"],
    parameters=LineLoad(active=[True, True, True], value=[0, -10, 0]),
)

# -------------------------------------------------------------------------------------
# Constraints
mp_supports = ModelPart.from_physical_group_and_parameters(
    physical_group=mesh.physical_groups["Solid_Displacement-auto-1"],
    parameters=DisplacementConstraint(
        active=[True, True, True], is_fixed=[True, True, True], value=[0, 0, 0]
    ),
)

# -------------------------------------------------------------------------------------
# Soil volume

mp_soil = BodyModelPart.from_physical_group_and_parameters(
    physical_group=mesh.physical_groups["Soil_drained-auto-1"],
    parameters=DefaultMaterial.Peat.value,
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
    part_name=mp_soil.name,
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
# Model and IO
# assemble the model and create IO writer
model = Model(
    ndim=2,
    model_parts=[mp_line_load, mp_supports],
    body_model_parts=[mp_soil],
    mesh=mesh,
)
kratos_io = KratosIO(ndim=2, model=model, outputs=[gid_output])

# -------------------------------------------------------------------------------------
# MaterialsParameters.json
# NB this goes first, because it provides ID to the materials
kratos_io.write_material_parameters_json("MaterialParameters_file.json")

# -------------------------------------------------------------------------------------
# ProjectParameters.json
kratos_io.write_project_parameters_json(filename="ProjectParameters.json")

# read parameters in and write block for solver!
project_parameters = json.load(open("ProjectParameters.json", "r"))
project_parameters = add_solver_settings_to_project_parameters(
    project_parameters=project_parameters,
    fname="test_mesh_file",
    materials_fname="MaterialParameters_file.json",
)
json.dump(project_parameters, open("ProjectParameters.json", "w"), indent=4)
# -------------------------------------------------------------------------------------
# Assemble parts together and write input files
# mdpa file
kratos_io.write_mesh_to_mdpa(filename="test_mesh_file.mdpa")

# -------------------------------------------------------------------------------------
# run Kratos!!


with open("ProjectParameters.json", "r") as parameter_file:
    parameters = KratosMultiphysics.Parameters(parameter_file.read())

# TODO: now we get an error because the condition are specified as the whole list a node comprising element 1
#  however, should only take a list of descending nodes.

model = KratosMultiphysics.Model()
simulation = GeoMechanicsAnalysis(model, parameters)
simulation.Run()
