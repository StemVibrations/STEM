input_files_dir = "kakalawa_hs"
results_dir = "output"
import math
import copy
import os
import sys
import json
import orjson
import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw, SmallStrainUmatLaw, MohrCoulombLaw, SmallStrainUdsmLaw, InterfaceMaterial, \
    HardeningSoilLaw
from stem.structural_material import ElasticSpringDamper, NodalConcentrated, EulerBeam, Anchor, StructuralMaterial, \
    EmbeddedBeamInterface
from stem.additional_processes import Excavation, ApplyFinalStressesOfPreviousStageToInitialState, \
    FindNeighboursOfInterfacesProcess, MasterSlaveConstraint, ExtrapolateIntegrationPointToNodesParameters, K0Procedure
# from stem.default_materials import DefaultMaterial
from stem.load import LineLoad, WaterLineLoad, PointLoad, PointMoment
from stem.water_processes import PhreaticLineWaterPressure
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg, Amgcl, Lu, LineSearchStrategy, ResidualConvergenceCriteria, \
    GlobalForceErrorConvergenceCriteria, LocalErrorCriteria
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters, GaussPointOutput, GiDOutputParameters
from stem.table import Table
from stem.stem import Stem
from stem.utils import Utils
from stem.prob_utils import ProbUtils

from stem.sheetpile_utils import SheetPileUtils

# For all materials: p_ref = 100 kPa, Rf = 0.9, nu_ur = 0.2.

# check if windows or linux
if sys.platform.startswith("win"):
    HARDENING_SOIL_DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardening_soil.dll")
else:
    HARDENING_SOIL_DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardening_soil.so")

HS_P_REF = 100e3
HS_FAILURE_RATIO = 0.9


def _p(mean, rel_std=0.05):
    return {"mean": mean, "std": abs(mean) * rel_std, "distribution": "normal"}


def _hs_soil(gamma_unsat, gamma_sat, e50_ref, eoed_ref, eur_ref, m, c, phi, psi, k0_nc, r_inter=0.85):
    """ all input in PLAXIS units (kN/m3, kN/m2, degrees), converted to SI """
    return {"sat_unit_weight": _p(gamma_sat * 1e3),
            "unsat_unit_weight": _p(gamma_unsat * 1e3),
            "e50_ref": _p(e50_ref * 1e3),
            "eoed_ref": _p(eoed_ref * 1e3),
            "eur_ref": _p(eur_ref * 1e3),
            "power_m": _p(m),
            # small cohesion for the sands, as in kakalawa.py (1 Pa)
            "cohesion": _p(max(c * 1e3, 1.0)),
            "friction_angle": _p(phi),
            "dilatancy_angle": _p(psi),
            "k0_nc": _p(k0_nc),
            "r_inter": _p(r_inter)}


prob_parameters = {
    #                            g_unsat g_sat  E50_ref  Eoed_ref Eur_ref   m     c    phi    psi   K0nc
    "sand_clean_loose": _hs_soil(17.0, 19.0, 20.5e3, 20.5e3, 20.5e3*3, 0.55, 0.0, 32.6, 2.6, 0.4612, 0.57),  # 01
    "clay_sandy":       _hs_soil(15.0, 15.0, 3000.0,  2000.0,  9000, 0.8, 5.0, 22.50, 0.00, 0.6173, 0.88),  # 06
    "sand_clean_mid":   _hs_soil(18.0, 20.0, 41.1e3, 41.1e3, 41.1e3*3, 0.55, 0.0, 38.3, 8.3, 0.38, 0.46),  # 08
    "sand_clayey":      _hs_soil(17.4, 19.4, 45e3, 33.8e3, 135e3, 0.50, 0.0, 37.7, 15, 0.4,0.47 ),  # 12
    "clay_clean":       _hs_soil(17.0, 17.0, 4500,  3000.0,  14e3, 0.80, 28.2, 25.20, 0.00, 0.67, 0.8),  # 10
    "sand_clean_dense": _hs_soil(19.0, 21.0, 75.5e3, 75.5e3, 226.5e3, 0.550, 0.0, 37.8, 7.8, 0.428, 0.47),  # 07
                   "sheetpile_E15": {"EI": {"mean": 640e6, "std": 1e6, "distribution": "normal"},
                                    "EA": {"mean": 12e9, "std": 1e6, "distribution": "normal"},
                                    "w": {"mean": 12e3, "std": 1e3, "distribution": "normal"}},
                   "sheetpile_E357": {"EI": {"mean": 1.523e9, "std": 1e6, "distribution": "normal"},
                                    "EA": {"mean": 28.56e9, "std": 1e6, "distribution": "normal"},
                                    "w": {"mean": 12e3, "std": 1e3, "distribution": "normal"}},
                   "anchor":        {"EA": {"mean": 472.5e6, "std": 1e6, "distribution": "normal"},
                                    "prestress": {"mean": 1000e3, "std": 1e3, "distribution": "normal"}},


}

# set all disitrubutions to none
for material_name, material_parameters in prob_parameters.items():
    for parameter_name, parameter_info in material_parameters.items():
        parameter_info["distribution"] = "none"


def connect_anchor_to_embedded_beam(model, anchor_name, beam_name, connection_coordinates, tolerance=1e-6):
    """
    Replaces the node of the anchor at the connection coordinates (a soil node) by the node of the embedded beam at the
    same location, such that the anchor is connected to the embedded beam instead of to the soil. Must be called after
    the mesh is generated.
    """
    anchor_mesh = model.get_model_part_by_name(anchor_name).mesh
    beam_mesh = model.embedded_beams[beam_name]["beam_model_part"].mesh
    connection_coordinates = np.array(connection_coordinates, dtype=float)

    def find_node_id(mesh):
        for node_id, node in mesh.nodes.items():
            if np.linalg.norm(np.array(node.coordinates) - connection_coordinates) < tolerance:
                return node_id
        raise ValueError(f"No node found at {connection_coordinates.tolist()}.")

    old_node_id = find_node_id(anchor_mesh)
    new_node_id = find_node_id(beam_mesh)

    anchor_mesh.nodes[new_node_id] = beam_mesh.nodes[new_node_id]
    del anchor_mesh.nodes[old_node_id]

    line_elements = model.gmsh_io.mesh_data["elements"]["LINE_2N"]
    for element_id, element in anchor_mesh.elements.items():
        element.node_ids = [new_node_id if node_id == old_node_id else node_id for node_id in element.node_ids]
        line_elements[element_id] = list(element.node_ids)

    model.gmsh_io.mesh_data["physical_groups"][anchor_name]["node_ids"] = sorted(anchor_mesh.nodes.keys())


def collect_sampled_parameters(parameters_dict):
    sampled_parameters = {}
    for material_name, material_parameters in parameters_dict.items():
        for parameter_name, parameter_info in material_parameters.items():
            sampled_value = parameter_info.get("sample")
            sampled_parameters[f"{material_name}.{parameter_name}"] = float(sampled_value)
    return sampled_parameters


def correlated_dilatancy_angle(parameters_dict, material_name, base_dilatancy_angle):
    if base_dilatancy_angle == 0.0:
        return 0.0
    mean_friction_angle = parameters_dict[material_name]["friction_angle"]["mean"]
    sampled_friction_angle = parameters_dict[material_name]["friction_angle"]["sample"]
    percentual_difference_sample_mean = (sampled_friction_angle - mean_friction_angle) / mean_friction_angle
    return base_dilatancy_angle + percentual_difference_sample_mean * base_dilatancy_angle


def append_sample_to_json(results_file, sample_record):
    """Append a single sample record to the monte_carlo_results.json file."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Read existing results if file exists, otherwise start fresh
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_dict = orjson.loads(f.read())
    else:
        results_dict = {"samples": []}
    
    # Append new sample
    results_dict["samples"].append(sample_record)
    
    # Write back
    with open(results_file, "w") as f:
        json_string = orjson.dumps(results_dict, option=orjson.OPT_INDENT_2)
        f.write(json_string.decode("utf-8"))


def load_sample_parameters_from_results(results_file, target_sample_number):
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "rb") as f:
        data = orjson.loads(f.read())

    for sample in data.get("samples", []):
        if sample.get("sample_number") == target_sample_number:
            if "parameters" not in sample:
                raise KeyError(f"Sample {target_sample_number} has no 'parameters' field")
            return sample["parameters"]

    raise ValueError(f"Sample number {target_sample_number} not found in {results_file}")


def apply_sampled_parameters(parameters_dict, sampled_parameters):
    for full_parameter_name, sample_value in sampled_parameters.items():
        material_name, parameter_name = full_parameter_name.split(".", 1)
        if material_name not in parameters_dict or parameter_name not in parameters_dict[material_name]:
            raise KeyError(f"Unknown sampled parameter: {full_parameter_name}")
        parameters_dict[material_name][parameter_name]["sample"] = float(sample_value)


n_samples = 1
results_dict = {"samples": []}

cwd = os.getcwd()
res_dir = os.path.join(cwd, input_files_dir, results_dir)
results_file = os.path.join(cwd, input_files_dir, results_dir, "monte_carlo_results.json")

# Optional replay mode:
# python kakalawa.py <sample_number> [results_file_path]
# replay_sample_number = int(sys.argv[1]) if len(sys.argv) > 1 else None

replay_sample_number = None
# replay_results_file = sys.argv[2] if len(sys.argv) > 2 else results_file
replay_parameters = None
if replay_sample_number is not None:
    replay_parameters = load_sample_parameters_from_results(replay_results_file, replay_sample_number)
    n_samples = 1


for sample_number in range(n_samples):

    if replay_parameters is None:
        ProbUtils.sample_all_parameters(prob_parameters)
        active_sample_number = sample_number + 1
    else:
        apply_sampled_parameters(prob_parameters, replay_parameters)
        active_sample_number = replay_sample_number

    ndim = 2
    model = Model(ndim)
    model.mesh_settings.element_order = 2
    gravity_constant = 9.81

    # Poisson ratio (unloading-reloading) of the soil
    soil_poisson_ratio = 0.2


    def hs_cap_parameters(soil_name):
        """
        Cap parameters of the hardening soil model, estimated from K0nc and Eoed_ref (PLAXIS input):
         - M_CAP: with associated flow on the elliptic cap, an oedometric (K0nc) stress path requires
           d eps_q / d eps_v = 2/3, which gives M^2 = 3/2 * eta, with eta = q/p = 3 (1 - K0nc) / (1 + 2 K0nc).
         - K_RATIO = Ks / Kc, with Ks = Eur / (3 (1 - 2 nu_ur)) and Kc = dp/deps_v in the oedometer test
           = Eoed (1 + 2 K0nc) / 3.
        """
        prm = prob_parameters[soil_name]
        k0 = prm["k0_nc"]["sample"]
        eta = 3.0 * (1.0 - k0) / (1.0 + 2.0 * k0)
        # m_cap = math.sqrt(1.5 * eta)
        # m_cap = 3 - 2.8* k0
        m_cap = 1.5
        k_ratio = prm["eur_ref"]["sample"] / ((1.0 - 2.0 * soil_poisson_ratio) * prm["eoed_ref"]["sample"]
                                              * (1.0 + 2.0 * k0))
        return m_cap, max(k_ratio, 1.05)

    def create_hs_soil(soil_name):
        """
        Creates the formulation, the hardening soil law, a Mohr-Coulomb law for the interfaces and the soil material.
        """
        prm = prob_parameters[soil_name]
        gamma_unsat = prm["unsat_unit_weight"]["sample"]
        gamma_sat = prm["sat_unit_weight"]["sample"]
        gamma_w = 9.81e3
        porosity = max((gamma_sat - gamma_unsat) / gamma_w, 0.0)
        # DENSITY_SOLID is the density of the grains: Kratos weighs the soil as (1 - n) rho_s + n S rho_w, so
        # the dry soil weighs gamma_unsat and the saturated soil gamma_unsat + n gamma_w = gamma_sat
        formulation = OnePhaseSoil(ndim, IS_DRAINED=True,
                                   DENSITY_SOLID=gamma_unsat / ((1.0 - porosity) * gravity_constant),
                                   POROSITY=porosity)

        dilatancy_angle = correlated_dilatancy_angle(prob_parameters, soil_name, prm["dilatancy_angle"]["mean"])
        m_cap, k_ratio = hs_cap_parameters(soil_name)

        law = HardeningSoilLaw(UMAT_NAME=HARDENING_SOIL_DLL,
                               E50_REF=prm["e50_ref"]["sample"],
                               EUR_REF=prm["eur_ref"]["sample"],
                               POWER_M=prm["power_m"]["sample"],
                               COHESION=prm["cohesion"]["sample"],
                               FRICTION_ANGLE=prm["friction_angle"]["sample"],
                               DILATANCY_ANGLE=dilatancy_angle,
                               P_REF=HS_P_REF,
                               FAILURE_RATIO=HS_FAILURE_RATIO,
                               POISSON_RATIO=soil_poisson_ratio,
                               M_CAP=m_cap,
                               K_RATIO=k_ratio)

        # the interfaces stay Mohr-Coulomb, with the strength of the soil and E50_ref as stiffness. The strength is
        # reduced by R_inter when the interfaces are activated (see set_interface_strength). As in PLAXIS (tension
        # cut-off with a tensile strength of 0), the interfaces cannot transfer tension: a gap opens instead, which
        # must close before compression can develop again (consider gap closure, set on the interface materials)
        interface_law = MohrCoulombLaw(YOUNG_MODULUS=prm["e50_ref"]["sample"],
                                       POISSON_RATIO=soil_poisson_ratio,
                                       GEO_FRICTION_ANGLE=prm["friction_angle"]["sample"],
                                       GEO_DILATANCY_ANGLE=dilatancy_angle,
                                       GEO_COHESION=prm["cohesion"]["sample"],
                                       GEO_TENSILE_STRENGTH=0.0)

        # stage 1 is the K0 procedure (as in PLAXIS): linear elastic, of which only the diagonal of the stiffness
        # matrix is used, such that the vertical effective stress equals the weight of the soil column above minus
        # the water pressure. The horizontal effective stresses are then set to K0nc times the vertical effective
        # stress and the shear stresses are zero. The hardening soil law is assigned in stage 2 (nil step), which
        # restores equilibrium where the surface or the layers are not horizontal.
        k0_procedure_law = LinearElasticSoil(YOUNG_MODULUS=prm["e50_ref"]["sample"],
                                             POISSON_RATIO=soil_poisson_ratio,
                                             K0_MAIN_DIRECTION=1,
                                             K0_NC=prm["k0_nc"]["sample"])
        hs_laws[f"material_{soil_name}"] = law

        material = SoilMaterial(f"material_{soil_name}", formulation, k0_procedure_law,
                                SaturatedBelowPhreaticLevelLaw())
        return formulation, law, interface_law, material

    # hardening soil laws per soil material name, assigned in stage 2
    hs_laws = {}

    formulation_clay_clean, constitutive_law_clay_clean, interface_law_clay_clean, material_clay_clean = \
        create_hs_soil("clay_clean")
    formulation_clay_sandy, constitutive_law_clay_sandy, interface_law_clay_sandy, material_clay_sandy = \
        create_hs_soil("clay_sandy")
    formulation_sand_clayey, constitutive_law_sand_clayey, interface_law_sand_clayey, material_sand_clayey = \
        create_hs_soil("sand_clayey")
    formulation_sand_clean_loose, constitutive_law_sand_clean_loose, interface_law_sand_clean_loose, \
        material_sand_clean_loose = create_hs_soil("sand_clean_loose")
    formulation_sand_clean_mid, constitutive_law_sand_clean_mid, interface_law_sand_clean_mid, \
        material_sand_clean_mid = create_hs_soil("sand_clean_mid")
    formulation_sand_clean_dense, constitutive_law_sand_clean_dense, interface_law_sand_clean_dense, \
        material_sand_clean_dense = create_hs_soil("sand_clean_dense")


    formulation_concrete = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1000/ gravity_constant, POROSITY=0.0)
    constitutive_law_concrete = MohrCoulombLaw(YOUNG_MODULUS=500e6, POISSON_RATIO=0.0, GEO_FRICTION_ANGLE=0.0,
                                                        GEO_DILATANCY_ANGLE=0.0, GEO_COHESION=10e6, GEO_TENSILE_STRENGTH=10e3)

    material_concrete = SoilMaterial("material_concrete", formulation_concrete, constitutive_law_concrete,
                                       SaturatedBelowPhreaticLevelLaw())


    w = prob_parameters["sheetpile_E15"]["w"]["sample"]
    EA = prob_parameters["sheetpile_E15"]["EA"]["sample"]
    EI = prob_parameters["sheetpile_E15"]["EI"]["sample"]
    d = 0.8

    E = EA / d
    I = EI / E
    cross_area = d * 1
    density = w/ (d * 1 * gravity_constant)
    sheetpile_E15 = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.15, CROSS_AREA=cross_area, I33=I, DENSITY=density)

    w = prob_parameters["sheetpile_E357"]["w"]["sample"]
    EA = prob_parameters["sheetpile_E357"]["EA"]["sample"]
    EI = prob_parameters["sheetpile_E357"]["EI"]["sample"]
    d = 0.8

    E = EA / d
    I = EI / E
    cross_area = d * 1
    density = w/ (d * 1 * gravity_constant)
    sheetpile_E357 = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.15, CROSS_AREA=cross_area, I33=I, DENSITY=density)

    w = 0
    EA = 4e9
    EI = 208.3e9
    d = 25

    E = EA / d
    I = EI / E
    cross_area = d * 1
    density = w/ (d * 1 * gravity_constant)
    bottom_plate = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.0, CROSS_AREA=cross_area, I33=I, DENSITY=density)


    EA = prob_parameters["anchor"]["EA"]["sample"]
    cross_area = cross_area
    spacing = 3.6
    E = EA / spacing / cross_area
    anchor = Anchor(YOUNG_MODULUS=E, CROSS_AREA=cross_area, PRESTRESS=prob_parameters["anchor"]["prestress"]["sample"] / cross_area / spacing)
    # anchor = Anchor(YOUNG_MODULUS=E, CROSS_AREA=cross_area)

    # grout body as a Plaxis embedded beam row: solid circular beam, properties per anchor, smeared over the spacing
    gamma_grout = 26e3                                  # unit weight gamma = 26 kN/m3
    spacing = 3.6                                       # L_spacing [m]
    grout_diameter = 0.185                              # D [m]
    cross_area = math.pi * grout_diameter ** 2 / 4      # A = 0.02688 m2
    I33 = math.pi * grout_diameter ** 4 / 64            # I = 0.05750e-3 m4
    E = 17.58e9 / spacing                               # E = 17.58e6 kN/m2

    # weight per metre width is gamma * A / spacing, so the (smeared) density is gamma / (g * spacing)
    grout = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.15, CROSS_AREA=cross_area, I33=I33,
                      DENSITY=gamma_grout / (gravity_constant * spacing))
    grout_material = StructuralMaterial("grout", grout)



    no_interface_constitutive_law = LinearElasticSoil(YOUNG_MODULUS=1e15, POISSON_RATIO=0.2)

    # interfaces
    # interface_sand_clayey_material = InterfaceMaterial("interface_sand_clayey_material", constitutive_law=constitutive_law_sand_clayey, soil_formulation=formulation_sand_clayey, retention_parameters=SaturatedBelowPhreaticLevelLaw())
    # interface_clay_clean_material = InterfaceMaterial("interface_clay_clean_material", constitutive_law=constitutive_law_clay_clean, soil_formulation=formulation_clay_clean, retention_parameters=SaturatedBelowPhreaticLevelLaw())
    # interface_clay_sandy_material = InterfaceMaterial("interface_clay_sandy_material", constitutive_law=constitutive_law_clay_sandy, soil_formulation=formulation_clay_sandy, retention_parameters=SaturatedBelowPhreaticLevelLaw())
    # interface_sand_clean_loose_material = InterfaceMaterial("interface_sand_clean_loose_material", constitutive_law=constitutive_law_sand_clean_loose, soil_formulation=formulation_sand_clean_loose, retention_parameters=SaturatedBelowPhreaticLevelLaw())
    # interface_sand_clean_mid_material = InterfaceMaterial("interface_sand_clean_mid_material", constitutive_law=constitutive_law_sand_clean_mid, soil_formulation=formulation_sand_clean_mid, retention_parameters=SaturatedBelowPhreaticLevelLaw())
    # interface_sand_clean_dense_material = InterfaceMaterial("interface_sand_clean_dense_material", constitutive_law=constitutive_law_sand_clean_dense, soil_formulation=formulation_sand_clean_dense, retention_parameters=SaturatedBelowPhreaticLevelLaw())

    # all interfaces consider gap closure, as in PLAXIS (only effective once the Mohr-Coulomb law is assigned)
    interface_sand_clayey_material = InterfaceMaterial("interface_sand_clayey_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clayey, retention_parameters=SaturatedBelowPhreaticLevelLaw(), consider_gap_closure=True)
    interface_clay_clean_material = InterfaceMaterial("interface_clay_clean_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_clay_clean, retention_parameters=SaturatedBelowPhreaticLevelLaw(), consider_gap_closure=True)
    interface_clay_sandy_material = InterfaceMaterial("interface_clay_sandy_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_clay_sandy, retention_parameters=SaturatedBelowPhreaticLevelLaw(), consider_gap_closure=True)
    interface_sand_clean_loose_material = InterfaceMaterial("interface_sand_clean_loose_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clean_loose, retention_parameters=SaturatedBelowPhreaticLevelLaw(), consider_gap_closure=True)
    interface_sand_clean_mid_material = InterfaceMaterial("interface_sand_clean_mid_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clean_mid, retention_parameters=SaturatedBelowPhreaticLevelLaw(), consider_gap_closure=True)
    interface_sand_clean_dense_material = InterfaceMaterial("interface_sand_clean_dense_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clean_dense, retention_parameters=SaturatedBelowPhreaticLevelLaw(), consider_gap_closure=True)
    # #
    #


    mid_left = -0.4
    mid_right = 0.4

    bottom_coordinate = -40
    top_coordinate = 5.5
    left_coord = -100
    right_coord = 100


    bot_sand = 3.2
    sand_clean_loose_left_coordinates =[(-21,bot_sand,0), (mid_left,bot_sand,0), (mid_left,top_coordinate,0), (-13.18,top_coordinate,0)]
    sand_clean_loose_right_coordinates = [(mid_right,bot_sand,0), (right_coord,bot_sand,0), (right_coord,top_coordinate,0), (mid_right,top_coordinate,0)]
    sand_clean_loose_mid = [(mid_right,bot_sand,0), (mid_left,bot_sand,0), (mid_left,top_coordinate,0), (mid_right,top_coordinate,0)]

    top_clay_sandy = bot_sand
    bot_clay_sandy = 2.9
    clay_sandy_left_coordinates = [(-22,bot_clay_sandy,0), (mid_left,bot_clay_sandy,0), (mid_left,top_clay_sandy,0), (-21,top_clay_sandy,0)]
    clay_sandy_right_coordinates = [(mid_right,bot_clay_sandy,0), (right_coord,bot_clay_sandy,0), (right_coord,top_clay_sandy,0), (mid_right,top_clay_sandy,0)]
    clay_sandy_mid = [(mid_right,bot_clay_sandy,0), (mid_left,bot_clay_sandy,0), (mid_left,top_clay_sandy,0), (mid_right,top_clay_sandy,0)]

    top_sand_clean_mid = bot_clay_sandy
    bot_sand_clean_mid_part1 = 1.5
    bot_sand_clean_mid_part2 = -6

    sand_clean_mid_part_1_left_coordinates = [(-26.8,bot_sand_clean_mid_part1,0), (mid_left,bot_sand_clean_mid_part1,0), (mid_left,top_sand_clean_mid,0), (-22,top_sand_clean_mid,0)]
    sand_clean_mid_part_2_left_coordinates = [(-52.3,bot_sand_clean_mid_part2,0), (mid_left,bot_sand_clean_mid_part2,0), (mid_left,bot_sand_clean_mid_part1,0), (-26.8,bot_sand_clean_mid_part1,0)]
    sand_clean_mid_right_coordinates = [(mid_right,bot_sand_clean_mid_part2,0), (right_coord,bot_sand_clean_mid_part2,0), (right_coord,top_sand_clean_mid,0), (mid_right,top_sand_clean_mid,0)]
    sand_clean_mid_mid_coordinates = [(mid_right,bot_sand_clean_mid_part2,0), (mid_left,bot_sand_clean_mid_part2,0), (mid_left,top_sand_clean_mid,0), (mid_right,top_sand_clean_mid,0)]

    sand_clayey_top = bot_sand_clean_mid_part2
    sand_clayey_bot = -14.5

    sand_clayey_part_1_left_coordinates =[(-81.3,sand_clayey_bot,0), (-30,sand_clayey_bot,0), (-7.5,-9,0), (mid_left,-9,0), (mid_left,sand_clayey_top,0),
                                          (-52.3,sand_clayey_top,0)]
    sand_clayey_part_2_left_coordinates = [(-30,sand_clayey_bot,0), (mid_left,sand_clayey_bot,0), (mid_left,-9,0), (-7.5,-9,0)]

    sand_clayey_right_coordinates = [(mid_right,sand_clayey_bot,0), (right_coord,sand_clayey_bot,0), (right_coord,sand_clayey_top,0), (mid_right,sand_clayey_top,0)]
    sand_clayey_mid_coordinates = [(mid_right,sand_clayey_bot,0), (mid_left,sand_clayey_bot,0), (mid_left,sand_clayey_top,0), (mid_right,sand_clayey_top,0)]

    sand_clean_mid_low_top = sand_clayey_bot
    sand_clean_mid_low_bot = -18.8
    sand_clean_mid_low_part_1_coordinates =[ [-95.9, sand_clean_mid_low_bot, 0], [-46.5, sand_clean_mid_low_bot, 0], [-30,sand_clean_mid_low_top,0], [-81.3,sand_clean_mid_low_top,0]]
    sand_clean_mid_low_part_2_coordinates = [(-46.5, sand_clean_mid_low_bot, 0), (mid_left, sand_clean_mid_low_bot, 0), (mid_left, sand_clean_mid_low_top, 0), (-30, sand_clean_mid_low_top, 0)]
    sand_clean_mid_low_right_coordinates = [(mid_right, sand_clean_mid_low_bot, 0), (right_coord, sand_clean_mid_low_bot, 0), (right_coord, sand_clean_mid_low_top, 0), (mid_right, sand_clean_mid_low_top, 0)]
    sand_clean_mid_low_mid_coordinates = [(mid_right, sand_clean_mid_low_bot, 0), (mid_left, sand_clean_mid_low_bot, 0), (mid_left, sand_clean_mid_low_top, 0), (mid_right, sand_clean_mid_low_top, 0)]

    clay_clean_mid_top = sand_clean_mid_low_bot
    clay_clean_mid_bot = -21.8

    clay_clean_mid_part_1_coordinates = [[left_coord, clay_clean_mid_bot, 0], [-58.3, clay_clean_mid_bot, 0], [-46.5,clay_clean_mid_top,0], [-95.9,clay_clean_mid_top,0], [left_coord,-20,0]]
    clay_clean_mid_part_2_coordinates = [(-58.3, clay_clean_mid_bot, 0), (mid_left, clay_clean_mid_bot, 0), (mid_left, clay_clean_mid_top, 0), (-46.5, clay_clean_mid_top, 0)]
    clay_clean_mid_right_coordinates = [(mid_right, clay_clean_mid_bot, 0), (right_coord, clay_clean_mid_bot, 0), (right_coord, clay_clean_mid_top, 0), (mid_right, clay_clean_mid_top, 0)]
    clay_clean_mid_mid_coordinates = [(mid_right, clay_clean_mid_bot, 0), (mid_left, clay_clean_mid_bot, 0), (mid_left, clay_clean_mid_top, 0), (mid_right, clay_clean_mid_top, 0)]

    sand_clean_dense_top = clay_clean_mid_bot
    sand_clean_dense_bot = bottom_coordinate
    bottom_sheetpile_coord = -22.3

    sand_clean_dense_coordinates = [[left_coord, sand_clean_dense_bot, 0], [right_coord, sand_clean_dense_bot, 0], [right_coord, sand_clean_dense_top, 0], [mid_right, sand_clean_dense_top, 0], (mid_right, bottom_sheetpile_coord, 0), (mid_left, bottom_sheetpile_coord, 0), (mid_left,sand_clean_dense_top,0), (left_coord,sand_clean_dense_top,0)]
    sand_clean_dense_mid_coordinates = [(mid_right, bottom_sheetpile_coord, 0), (mid_left, bottom_sheetpile_coord, 0), (mid_left, sand_clean_dense_top, 0), (mid_right, sand_clean_dense_top, 0)]

    model.add_soil_layer_by_coordinates(sand_clean_loose_left_coordinates, material_sand_clean_loose, name="sand_clean_loose_left")
    model.add_soil_layer_by_coordinates(sand_clean_loose_right_coordinates, material_sand_clean_loose, name="sand_clean_loose_right")
    model.add_soil_layer_by_coordinates(sand_clean_loose_mid, material_sand_clean_loose, name="sand_clean_loose_mid")
    model.add_soil_layer_by_coordinates(clay_sandy_left_coordinates, material_clay_sandy, name="clay_sandy_left")
    model.add_soil_layer_by_coordinates(clay_sandy_right_coordinates, material_clay_sandy, name="clay_sandy_right")
    model.add_soil_layer_by_coordinates(clay_sandy_mid, material_clay_sandy, name="clay_sandy_mid")
    model.add_soil_layer_by_coordinates(sand_clean_mid_part_1_left_coordinates, material_sand_clean_mid, name="sand_clean_mid_part_1_left")
    model.add_soil_layer_by_coordinates(sand_clean_mid_part_2_left_coordinates, material_sand_clean_mid, name="sand_clean_mid_part_2_left")
    model.add_soil_layer_by_coordinates(sand_clean_mid_right_coordinates, material_sand_clean_mid, name="sand_clean_mid_right")
    model.add_soil_layer_by_coordinates(sand_clean_mid_mid_coordinates, material_sand_clean_mid, name="sand_clean_mid_mid")
    model.add_soil_layer_by_coordinates(sand_clayey_part_1_left_coordinates, material_sand_clayey, name="sand_clayey_part_1_left")
    model.add_soil_layer_by_coordinates(sand_clayey_part_2_left_coordinates, material_sand_clayey, name="sand_clayey_part_2_left")
    model.add_soil_layer_by_coordinates(sand_clayey_right_coordinates, material_sand_clayey, name="sand_clayey_right")
    model.add_soil_layer_by_coordinates(sand_clayey_mid_coordinates, material_sand_clayey, name="sand_clayey_mid")
    model.add_soil_layer_by_coordinates(sand_clean_mid_low_part_1_coordinates, material_sand_clean_mid, name="sand_clean_mid_low_part_1")
    model.add_soil_layer_by_coordinates(sand_clean_mid_low_part_2_coordinates, material_sand_clean_mid, name="sand_clean_mid_low_part_2")
    model.add_soil_layer_by_coordinates(sand_clean_mid_low_right_coordinates, material_sand_clean_mid, name="sand_clean_mid_low_right")
    model.add_soil_layer_by_coordinates(sand_clean_mid_low_mid_coordinates, material_sand_clean_mid, name="sand_clean_mid_low_mid")
    model.add_soil_layer_by_coordinates(clay_clean_mid_part_1_coordinates, material_clay_clean, name="clay_clean_mid_part_1")
    model.add_soil_layer_by_coordinates(clay_clean_mid_part_2_coordinates, material_clay_clean, name="clay_clean_mid_part_2")
    model.add_soil_layer_by_coordinates(clay_clean_mid_right_coordinates, material_clay_clean, name="clay_clean_mid_right")
    model.add_soil_layer_by_coordinates(clay_clean_mid_mid_coordinates, material_clay_clean, name="clay_clean_mid_mid")
    model.add_soil_layer_by_coordinates(sand_clean_dense_coordinates, material_sand_clean_dense, name="sand_clean_dense")
    model.add_soil_layer_by_coordinates(sand_clean_dense_mid_coordinates, material_sand_clean_dense, name="sand_clean_dense_mid")


    model.set_interface_between_model_parts(["sand_clean_loose_left","sand_clean_loose_right" ], ["sand_clean_loose_mid"], interface_sand_clean_loose_material, "interface_sand_clean_loose")
    model.set_interface_between_model_parts(["clay_sandy_left","clay_sandy_right" ], ["clay_sandy_mid"], interface_clay_sandy_material, "interface_clay_sandy")
    model.set_interface_between_model_parts(["sand_clean_mid_part_1_left","sand_clean_mid_part_2_left", "sand_clean_mid_right"], ["sand_clean_mid_mid"], interface_sand_clean_mid_material, "interface_sand_clean_mid")
    model.set_interface_between_model_parts(["sand_clayey_part_1_left","sand_clayey_part_2_left", "sand_clayey_right"], ["sand_clayey_mid"], interface_sand_clayey_material, "interface_sand_clayey", {"water_normal_load_soil": [True,False],
                                                                                                                                                                                                       "water_normal_load_beam": [False, True]})
    model.set_interface_between_model_parts(["sand_clean_mid_low_part_1","sand_clean_mid_low_part_2", "sand_clean_mid_low_right"], ["sand_clean_mid_low_mid"], interface_sand_clean_mid_material, "interface_sand_clean_mid_low")
    model.set_interface_between_model_parts(["clay_clean_mid_part_2", "clay_clean_mid_right"], ["clay_clean_mid_mid"], interface_clay_clean_material, "interface_clay_clean_mid")
    # model.set_interface_between_model_parts(["sand_clean_dense"], ["sand_clean_dense_mid"], interface_sand_clean_dense_material, "interface_sand_clean_dense")



    sheetpile_coordinates_part_1 = [(0, 5.5,0),(0, 1.5,0)]
    sheetpile_coordinates_part_2 = [(0, 1.5,0),(0, -15,0)]
    sheetpile_coordinates_part_3 = [(0, -15,0),(0, bottom_sheetpile_coord,0)]

    bottom_plat_coordinates = [(mid_right, bottom_sheetpile_coord, 0),(mid_left, bottom_sheetpile_coord, 0)]

    anchor_coordinates = [(0,2,0),(24.8,-22.8,0)]
    grout_coordinates = [(24.8,-22.8,0), (36.82,-34.82,0)]

    phreatic_level = 0.24

    def initial_vertical_effective_stress(depth):
        """
        Vertical effective stress [Pa] (compression positive) at the given vertical coordinate, at the right of the
        sheetpile (the side of the grout), from the weight of the soil layers above and the hydrostatic water pressure
        below the phreatic level.
        """
        gamma_w = 9.81e3
        layers = [(top_coordinate, bot_sand, "sand_clean_loose"),
                  (bot_sand, bot_clay_sandy, "clay_sandy"),
                  (bot_clay_sandy, bot_sand_clean_mid_part2, "sand_clean_mid"),
                  (sand_clayey_top, sand_clayey_bot, "sand_clayey"),
                  (sand_clean_mid_low_top, sand_clean_mid_low_bot, "sand_clean_mid"),
                  (clay_clean_mid_top, clay_clean_mid_bot, "clay_clean"),
                  (sand_clean_dense_top, sand_clean_dense_bot, "sand_clean_dense")]
        total_stress = 0.0
        for layer_top, layer_bottom, soil_name in layers:
            top = min(layer_top, top_coordinate)
            bottom = max(layer_bottom, depth)
            if bottom >= top:
                continue
            # unsaturated above and saturated below the phreatic level
            dry_part = max(0.0, top - max(bottom, phreatic_level))
            wet_part = (top - bottom) - dry_part
            total_stress += (dry_part * prob_parameters[soil_name]["unsat_unit_weight"]["sample"]
                             + wet_part * prob_parameters[soil_name]["sat_unit_weight"]["sample"])
        return total_stress - gamma_w * max(0.0, phreatic_level - depth)

    # interface between the embedded grout body and the soil (sand_clean_dense), smeared over the anchor spacing.
    # Stiffness as for a Plaxis embedded beam row: K = ISF * G_soil / spacing, with the default Plaxis interface
    # stiffness factors ISF_RS = ISF_RN = 2.5 * (spacing / D)^-0.75 = 0.2698.
    # G_soil is the elastic (unloading-reloading) shear modulus of the hardening soil model at the in-situ stress level
    # at the middle of the grout, with sigma'_3 = K0nc * sigma'_v:
    #   E_ur = E_ur_ref * ((c cos(phi) + sigma'_3 sin(phi)) / (c cos(phi) + p_ref sin(phi)))^m,  G = E_ur / (2 (1 + nu_ur))
    prm_grout_soil = prob_parameters["sand_clean_dense"]
    grout_mid_depth = 0.5 * (grout_coordinates[0][1] + grout_coordinates[1][1])
    grout_sigma_3 = prm_grout_soil["k0_nc"]["sample"] * initial_vertical_effective_stress(grout_mid_depth)
    phi_grout_soil = math.radians(prm_grout_soil["friction_angle"]["sample"])
    c_cos_phi = prm_grout_soil["cohesion"]["sample"] * math.cos(phi_grout_soil)
    eur_grout_soil = prm_grout_soil["eur_ref"]["sample"] * (
        (c_cos_phi + grout_sigma_3 * math.sin(phi_grout_soil))
        / (c_cos_phi + HS_P_REF * math.sin(phi_grout_soil))) ** prm_grout_soil["power_m"]["sample"]
    shear_modulus_grout_soil = eur_grout_soil / (2 * (1 + soil_poisson_ratio))
    axial_stiffness_factor = 0.2698
    lateral_stiffness_factor = 0.2698

    # axial skin resistance: linear with T_skin,start,max = T_skin,end,max = 170 kN/m per anchor, i.e. a constant
    # maximum shear force per unit length of T_skin / spacing per metre width. The 2D interface element limits the
    # shear force per unit length to 2 * GEO_COHESION (two faces), without friction the limit is stress independent.
    # Lateral resistance is unlimited (the normal component of the interface is linear elastic) and there is no base
    # resistance (F_max = 0).
    skin_resistance_max = 170e3                         # T_skin,max [N/m]
    grout_cohesion = skin_resistance_max / (2 * spacing)

    grout_interface_material = StructuralMaterial("grout_interface", EmbeddedBeamInterface(
        INTERFACE_SHEAR_STIFFNESS=axial_stiffness_factor * shear_modulus_grout_soil / spacing,
        INTERFACE_NORMAL_STIFFNESS=lateral_stiffness_factor * shear_modulus_grout_soil / spacing,
        GEO_COHESION=grout_cohesion))
    print(f"grout interface: sigma'_3 = {grout_sigma_3 / 1e3:.1f} kPa, G_soil = {shear_modulus_grout_soil / 1e6:.1f} MPa, "
          f"K = {axial_stiffness_factor * shear_modulus_grout_soil / spacing / 1e6:.2f} MN/m2")
    #
    SheetPileUtils.add_sheetpile_by_coordinates(sheetpile_coordinates_part_1,sheetpile_E357, "sheetpile_part_1", model.gmsh_io,model.body_model_parts)
    SheetPileUtils.add_sheetpile_by_coordinates(sheetpile_coordinates_part_2,sheetpile_E15, "sheetpile_part_2", model.gmsh_io,model.body_model_parts)
    SheetPileUtils.add_sheetpile_by_coordinates(sheetpile_coordinates_part_3,sheetpile_E357, "sheetpile_part_3", model.gmsh_io,model.body_model_parts)

    SheetPileUtils.add_sheetpile_by_coordinates(bottom_plat_coordinates,bottom_plate, "bottom_plate", model.gmsh_io,model.body_model_parts)

    SheetPileUtils.add_anchor_by_coordinates(anchor_coordinates,anchor,0.0, "anchor", model.gmsh_io,model.body_model_parts)
    # the grout is an embedded beam, connected to the soil by an interface (model part "grout_interface"). It is
    # meshed independently of the soil, the anchor is connected to the head of the grout after meshing
    model.add_embedded_beam_by_coordinates(grout_coordinates, grout_material, "grout", element_size=0.5,
                                           interface_material=grout_interface_material,
                                           host_model_part_names=["sand_clean_dense"])
    # model.show_geometry(show_line_ids=True)

    # model.set_interface_between_model_parts(["grout"],["sand_clean_dense"], interface_sand_clean_dense_material, "interface_sand_clean_dense_grout")
    # # model.set_interface_between_model_parts(["sand_clean_dense"], ["sand_clean_dense_mid","sheetpile_part_3"], interface_sand_clean_dense_material, "interface_sand_clean_dense")
    model.set_interface_between_model_parts(["sand_clean_dense"], ["sand_clean_dense_mid", "bottom_plate","sheetpile_part_3"],
                                            interface_sand_clean_dense_material, "interface_sand_clean_dense")
    # model.set_interface_between_model_parts(["sand_clean_dense"], ["sand_clean_dense_mid"], interface_sand_clean_dense_material, "interface_sand_clean_dense")


    excavation_building_pit_left_1 = Excavation(deactivate_body_model_part=False)
    excavation_building_pit_left_2_1 = Excavation(deactivate_body_model_part=False)
    excavation_building_pit_left_2_2 = Excavation(deactivate_body_model_part=False)
    excavation_building_pit_left_2_3 = Excavation(deactivate_body_model_part=False)
    excavation_building_pit_left_2_4 = Excavation(deactivate_body_model_part=False)

    excavation_sheetpiles = Excavation(deactivate_body_model_part=True)
    excavation_anchor = Excavation(deactivate_body_model_part=True)

    interface_excavation = Excavation(deactivate_body_model_part=True)

    # excavation_building_pit_right = Excavation(deactivate_body_model_part=False)
    # excavation_sheetpile = Excavation(deactivate_body_model_part=True)
    # excavation_anchor = Excavation(deactivate_body_model_part=True)
    # excavation_concrete_cover = Excavation(deactivate_body_model_part=False)
    #
    model.apply_additional_process(excavation_building_pit_left_1, "sand_clean_loose_left")
    model.apply_additional_process(excavation_building_pit_left_1, "clay_sandy_left")
    model.apply_additional_process(excavation_building_pit_left_1, "sand_clean_mid_part_1_left")

    model.apply_additional_process(excavation_building_pit_left_2_1, "sand_clean_mid_part_2_left")
    model.apply_additional_process(excavation_building_pit_left_2_2, "sand_clayey_part_1_left")
    model.apply_additional_process(excavation_building_pit_left_2_3, "sand_clean_mid_low_part_1")
    model.apply_additional_process(excavation_building_pit_left_2_4, "clay_clean_mid_part_1")

    model.apply_additional_process(excavation_sheetpiles, "sheetpile_part_1")
    model.apply_additional_process(excavation_sheetpiles, "sheetpile_part_2")
    model.apply_additional_process(excavation_sheetpiles, "sheetpile_part_3")
    model.apply_additional_process(excavation_sheetpiles, "bottom_plate")

    model.apply_additional_process(excavation_anchor, "anchor")
    model.apply_additional_process(excavation_anchor, "grout")
    model.apply_additional_process(excavation_anchor, "grout_interface")

    model.apply_additional_process(interface_excavation, "interface_sand_clean_loose")
    model.apply_additional_process(interface_excavation, "interface_clay_sandy")
    model.apply_additional_process(interface_excavation, "interface_sand_clean_mid")
    model.apply_additional_process(interface_excavation, "interface_sand_clayey")
    model.apply_additional_process(interface_excavation, "interface_sand_clean_mid_low")
    model.apply_additional_process(interface_excavation, "interface_clay_clean_mid")
    model.apply_additional_process(interface_excavation, "interface_sand_clean_dense")

    # K0 procedure on all soil layers in stage 1, it is removed in stage 2
    soil_layer_names = [mp.name for mp in model.body_model_parts
                        if isinstance(mp.material, SoilMaterial) and mp.material.name in hs_laws]
    model.apply_additional_process(K0Procedure(model_part_name_list=soil_layer_names))


    # vertical_point_load = PointLoad(active=[True, True, True], value=[0, -177.2e3, 0])
    # point_moment = PointMoment(active=[False, False, True], value=[0, 0, 90.4e3])
    vertical_point_load = PointLoad(active=[True, True, True], value=[0, 0, 0])
    point_moment = PointMoment(active=[False, False, True], value=[0, 0, 0])
    model.add_load_by_coordinates([(0.0,4.0,0.0)], vertical_point_load, "vertical_point_load")
    model.add_load_by_coordinates([(0.0,4.0,0.0)], point_moment, "point_moment")

    top_line_load_1 = LineLoad(active=[True, True, True], value=[0, 0.0, 0])
    model.add_load_by_coordinates([(0.4,5.5,0.0),(20,5.5,0.0)], top_line_load_1, "top_line_load_1")

    phreatic_line = PhreaticLineWaterPressure([(-100, phreatic_level, 0), (100, phreatic_level, 0)],
                                              is_fixed=True)
    model.add_phreatic_line(phreatic_line)
    #
    model.show_geometry(show_line_ids=True)

    water_load = WaterLineLoad(active=False, reference_coordinate=[-100, phreatic_level, 0])
    model.add_load_by_geometry_ids([48,49,40,29,30], water_load, "water_normal_load_soil")
    model.add_load_by_geometry_ids([31, 22], water_load, "water_normal_load_beam")

    init_water_load = WaterLineLoad(active=True, reference_coordinate=[-100, phreatic_level, 0])
    model.add_load_by_geometry_ids([23,32,41,50], init_water_load, "init_water_normal_load_soil")


    model.add_boundary_condition_by_geometry_ids(1, [51,63,59,55,45,36,25,15,6], DisplacementConstraint([True,False,True], [0,0,0]), "sides")
    model.add_boundary_condition_by_geometry_ids(1, [58], DisplacementConstraint([True,True,True], [0,0,0]), "fixed_support")

    # define the output settings in vtk format
    model.add_output_settings(
        part_name="porous_computational_model_part",
        output_dir=results_dir,
        output_name="vtk_output",
        output_parameters=VtkOutputParameters(
            file_format="ascii",
            output_interval=1,
            nodal_results=[NodalOutput.TOTAL_DISPLACEMENT, NodalOutput.DISPLACEMENT, NodalOutput.WATER_PRESSURE],
            gauss_point_results=[GaussPointOutput.CAUCHY_STRESS_VECTOR],
            output_control_type="step"))

    model.add_output_settings(part_name="sheetpile_part_1",
                              output_dir=results_dir,
                              output_name="json_output_sheetpile_part_1",
                                output_parameters=JsonOutputParameters(
                                    output_interval=1,
                                    nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.BENDING_MOMENT, NodalOutput.SHEAR_FORCE,NodalOutput.AXIAL_FORCE]))


    extrapolate_process = ExtrapolateIntegrationPointToNodesParameters(model_part_name_list=["sheetpile_part_1", "sheetpile_part_2", "sheetpile_part_3", "bottom_plate"],
                                                                       list_of_variables=[NodalOutput.BENDING_MOMENT.name, NodalOutput.SHEAR_FORCE.name, NodalOutput.AXIAL_FORCE.name])
    model.apply_additional_process(extrapolate_process)


    # stage 1 (K0 procedure) is linear and is calculated in one step
    end_time = 1
    delta_time = 1
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=end_time,
                                       delta_time=delta_time,
                                       reduction_factor=0.5,
                                       increase_factor=1.5,
                                       max_delta_time_factor=1000)

    # set convergence criteria
    # convergence_criterion = ResidualConvergenceCriteria(residual_relative_tolerance=5.0e-3,
    #                                                         residual_absolute_tolerance=5.0e-1)
    # convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-5,
    #                                                         displacement_absolute_tolerance=1.0e-12)
    convergence_criterion = GlobalForceErrorConvergenceCriteria(force_relative_tolerance =0.01,
                                                                force_absolute_tolerance =1.0e-9)

    linear_solver = Lu()
    # linear_solver = Cg(scaling=False, preconditioner_type="none")
    # linear_solver = Amgcl(smoother_type="damped_jacobi", krylov_type="cg", tolerance=1e-12, max_iteration=1000)
    # set solver settings stage 1 (K0 procedure, see K0Procedure)
    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.QUASI_STATIC,
                                     stress_initialisation_type=StressInitialisationType.K0_PROCEDURE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=False,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     local_error_criteria= LocalErrorCriteria(),
                                     strategy_type=NewtonRaphsonStrategy(),
                                     linear_solver_settings=linear_solver)

    # Set up problem data
    problem = Problem(problem_name="sheetpile", number_of_threads=16, settings=solver_settings)
    model.project_parameters = problem

    model.mesh_settings.element_size=2
    model.set_element_size_of_group(0.5, "sand_clean_loose_mid")
    model.set_element_size_of_group(0.5, "clay_sandy_mid")
    model.set_element_size_of_group(0.5, "sand_clean_mid_mid")
    model.set_element_size_of_group(0.5, "sand_clayey_mid")
    model.set_element_size_of_group(0.5, "sand_clean_mid_low_mid")
    model.set_element_size_of_group(0.5, "clay_clean_mid_mid")
    model.set_element_size_of_group(0.5, "sand_clean_dense_mid")
    model.set_element_size_of_group(0.5, "sand_clayey_part_2_left")

    # model.set_element_size_of_group(0.5, "water_normal_load_soil")

    # model.apply_additional_process(FindNeighboursOfInterfacesProcess(model_part_name_list=["interface_sand_clean_loose", "interface_clay_sandy", "interface_sand_clean_mid"
    # , "interface_sand_clayey", "interface_sand_clean_mid_low", "interface_clay_clean_mid", "interface_sand_clean_dense"]))

    # model.project_parameters.settings.quasi_newton_type = "none"
    model.project_parameters.settings.strategy_type.max_iterations = 1000
    stem = Stem(model, input_files_dir)

    # the embedded grout does not share nodes with the soil, connect the anchor to the head node of the grout, such
    # that the anchor force is transferred to the soil through the grout interface. This is done before the stages
    # are created, such that all stages share the connection.
    connect_anchor_to_embedded_beam(model, "anchor", "grout", grout_coordinates[0])

    try:

        # stage 2: nil step (as in PLAXIS). The soil is switched from the K0 procedure to hardening soil, without any
        # other changes, such that equilibrium is found from the K0 stresses. The displacements of the K0 procedure
        # have no physical meaning and are reset.
        stage2 = stem.create_new_stage(0.1,1)
        stage2.project_parameters.settings.stress_initialisation_type = StressInitialisationType.NONE
        stage2.project_parameters.settings.reset_displacements = True
        stage2.remove_model_part_by_name_and_type("", K0Procedure)
        for mp in stage2.body_model_parts:
            if isinstance(mp.material, SoilMaterial) and mp.material.name in hs_laws:
                mp.material.constitutive_law = copy.deepcopy(hs_laws[mp.material.name])
        # the global force error is relative to the internal forces of the full self weight, so 1% leaves a large
        # part of the out-of-balance of the K0 stresses, which is then released in stage 3 (slip plane of the slope).
        # The nil step must therefore reach equilibrium with a much smaller tolerance.
        # stage2.project_parameters.settings.convergence_criteria = GlobalForceErrorConvergenceCriteria(
        #     force_relative_tolerance=1.0e-4, force_absolute_tolerance=1.0e-9)
        stem.add_calculation_stage(stage2)

        # stage 3 install sheetpile, with the original tolerance (stages are copied from the previous stage)
        stage3 = stem.create_new_stage(0.1,1)
        stage3.project_parameters.settings.convergence_criteria = convergence_criterion
        # stage 4.project_parameters.settings.convergence_criteria.displacement_relative_tolerance= 5e-4
        # stage 4.project_parameters.settings.quasi_newton_type = "none"
        # stage 4.project_parameters.settings.strategy_type.max_iterations = 1000
        sheetpile_casing_mps = []
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("sand_clean_loose_mid"))
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("clay_sandy_mid"))
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("sand_clean_mid_mid"))
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("sand_clayey_mid"))
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("sand_clean_mid_low_mid"))
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("clay_clean_mid_mid"))
        sheetpile_casing_mps.append(stage3.get_model_part_by_name("sand_clean_dense_mid"))

        for mp in sheetpile_casing_mps:
            # mp.material = SoilMaterial(f"material_concrete_{mp.name}", copy.deepcopy(mp.material.soil_formulation),
            #                            constitutive_law_concrete, copy.deepcopy(mp.material.retention_parameters))
            mp.material = material_concrete

        stage3.get_additional_process_part_by_name_and_type("sheetpile_part_1", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("sheetpile_part_2", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("sheetpile_part_3", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("bottom_plate", Excavation).parameters.deactivate_body_model_part = False

        import copy



        def set_interface_strength(interface_name, interface_law, soil_name):
            """
            Assigns the Mohr-Coulomb law to the interface, with the strength of the soil reduced by R_inter as in
            PLAXIS: c_i = R_inter * c_soil, tan(phi_i) = R_inter * tan(phi_soil) and psi_i = 0 (for R_inter < 1). The
            tensile strength of the interface law is 0, such that R_inter * tensile strength stays 0.
            """
            r_inter = prob_parameters[soil_name]["r_inter"]["sample"]
            law = copy.deepcopy(interface_law)
            law.GEO_FRICTION_ANGLE = math.degrees(math.atan(r_inter * math.tan(math.radians(interface_law.GEO_FRICTION_ANGLE))))
            law.GEO_COHESION = r_inter * interface_law.GEO_COHESION
            law.GEO_DILATANCY_ANGLE = 0.0
            stage3.get_model_part_by_name(interface_name).material.constitutive_law = law

        set_interface_strength("interface_sand_clean_loose", interface_law_sand_clean_loose, "sand_clean_loose")
        set_interface_strength("interface_clay_sandy", interface_law_clay_sandy, "clay_sandy")
        set_interface_strength("interface_sand_clean_mid", interface_law_sand_clean_mid, "sand_clean_mid")
        set_interface_strength("interface_sand_clayey", interface_law_sand_clayey, "sand_clayey")
        set_interface_strength("interface_sand_clean_mid_low", interface_law_sand_clean_mid, "sand_clean_mid")
        set_interface_strength("interface_clay_clean_mid", interface_law_clay_clean, "clay_clean")
        set_interface_strength("interface_sand_clean_dense", interface_law_sand_clean_dense, "sand_clean_dense")


        stage3.get_additional_process_part_by_name_and_type("interface_sand_clean_loose", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("interface_clay_sandy", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("interface_sand_clean_mid", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("interface_sand_clayey", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("interface_sand_clean_mid_low", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("interface_clay_clean_mid", Excavation).parameters.deactivate_body_model_part = False
        stage3.get_additional_process_part_by_name_and_type("interface_sand_clean_dense", Excavation).parameters.deactivate_body_model_part = False

        # remove master slave, since interface is active
        stage3.remove_model_part_by_name_and_type("", MasterSlaveConstraint)


        stage3.apply_additional_process(FindNeighboursOfInterfacesProcess(model_part_name_list=["interface_sand_clean_loose", "interface_clay_sandy", "interface_sand_clean_mid"
        , "interface_sand_clayey", "interface_sand_clean_mid_low", "interface_clay_clean_mid", "interface_sand_clean_dense"]))

        stem.add_calculation_stage(stage3)

        # stage 5 excavation part 1
        stage4 = stem.create_new_stage(0.1,1)
        stage4.project_parameters.settings.quasi_newton_type = "lbfgs"
        stage4.get_additional_process_part_by_name_and_type("sand_clean_loose_left", Excavation).parameters.deactivate_body_model_part = True
        stage4.get_additional_process_part_by_name_and_type("clay_sandy_left", Excavation).parameters.deactivate_body_model_part = True
        stage4.get_additional_process_part_by_name_and_type("sand_clean_mid_part_1_left", Excavation).parameters.deactivate_body_model_part = True
        #
        #
        #
        move_stress_sheetpile = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["sheetpile_part_1", "sheetpile_part_2", "sheetpile_part_3", "bottom_plate"])
        stage4.apply_additional_process(move_stress_sheetpile)

        move_stress_interface = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["interface_sand_clean_loose", "interface_clay_sandy", "interface_sand_clean_mid",
                                                                                            "interface_sand_clayey", "interface_sand_clean_mid_low", "interface_clay_clean_mid","interface_sand_clean_dense" ])
        stage4.apply_additional_process(move_stress_interface)
        stage4.remove_model_part_by_name_and_type("", FindNeighboursOfInterfacesProcess)

        stem.add_calculation_stage(stage4)


        # stage 5 add anchor and grout
        stage5 = stem.create_new_stage(0.1,1)
        stage5.project_parameters.settings.quasi_newton_type = "lbfgs"
        stage5.get_additional_process_part_by_name_and_type("anchor", Excavation).parameters.deactivate_body_model_part = False
        stage5.get_additional_process_part_by_name_and_type("grout", Excavation).parameters.deactivate_body_model_part = False
        stage5.get_additional_process_part_by_name_and_type("grout_interface", Excavation).parameters.deactivate_body_model_part = False

        # stage 6.get_model_part_by_name("interface_sand_clean_dense_grout").material.constitutive_law = constitutive_law_sand_clean_dense
        # stage 6.get_model_part_by_name("interface_sand_clean_dense_grout").material.constitutive_law.GEO_COHESION = 50000
        # stage 6.get_model_part_by_name("interface_sand_clean_dense_grout").material.constitutive_law.YOUNG_MODULUS = 500e6
        stem.add_calculation_stage(stage5)

        # stage 6 add point load and moment
        stage6 = stem.create_new_stage(0.1,1)
        move_stress_anchor = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["anchor", "grout"])
        stage6.apply_additional_process(move_stress_anchor)
        anchor_model_part = stage6.get_model_part_by_name("anchor")

        # prestress is set to half because here, because the stress initialisation is wrong. In this case, the prestress stays
        # as it should be
        anchor_model_part.material.material_parameters.PRESTRESS = anchor_model_part.material.material_parameters.PRESTRESS/2


        stage6.get_model_part_by_name("vertical_point_load").parameters.value = [0, -177.2e3, 0]
        stage6.get_model_part_by_name("point_moment").parameters.value = [0, 0, 90.4e3]
        # stage 7.project_parameters.settings.convergence_criteria.displacement_relative_tolerance= 1e-4

        stem.add_calculation_stage(stage6)

        # stage 7 excavation part 2
        stage7 = stem.create_new_stage(0.01,1)
        stage7.get_additional_process_part_by_name_and_type("sand_clean_mid_part_2_left", Excavation).parameters.deactivate_body_model_part = True
        stage7.get_additional_process_part_by_name_and_type("sand_clayey_part_1_left", Excavation).parameters.deactivate_body_model_part = True
        stage7.get_additional_process_part_by_name_and_type("sand_clean_mid_low_part_1", Excavation).parameters.deactivate_body_model_part = True
        stage7.get_additional_process_part_by_name_and_type("clay_clean_mid_part_1", Excavation).parameters.deactivate_body_model_part = True

        #set water load
        stage7.get_model_part_by_name("water_normal_load_soil").parameters.active = True
        stage7.get_model_part_by_name("water_normal_load_beam").parameters.active = True
        # stage 8.get_model_part_by_name("water_normal_load_soil").parameters.reference_coordinate = [0,5,0]
        # stage 8.get_model_part_by_name("init_water_normal_load_soil").parameters.active = False

        anchor_model_part = stage7.get_model_part_by_name("anchor")
        # prestress is set to half because here, because the stress initialisation is wrong. In this case, the prestress stays
        # as it should be
        anchor_model_part.material.material_parameters.PRESTRESS = anchor_model_part.material.material_parameters.PRESTRESS/2

        stage7.get_model_part_by_name("top_line_load_1").parameters.value = [0, -10e3, 0]


        # this step is close to instable, so we need to relax the convergence criteria
        # stage 8.project_parameters.settings.convergence_criteria.displacement_relative_tolerance= 1e-3
        # stage 8.project_parameters.settings.quasi_newton_max_rank = 3
        stage7.project_parameters.settings.quasi_newton_restart_interval=1000
        # stage 8.project_parameters.settings.quasi_newton_type = "none"
        stage7.project_parameters.settings.strategy_type.min_iterations = 6
        stage7.project_parameters.settings.strategy_type.max_iterations = 200

        # convergence_criterion = ResidualConvergenceCriteria(residual_relative_tolerance=5.0e-2,
        #                                                         residual_absolute_tolerance=1.0e-2)

        stage7.project_parameters.settings.convergence_criteria = convergence_criterion

        stem.add_calculation_stage(stage7)


        stem.write_all_input_files()

        import time

        start_time_calc = time.time()
        stem.run_calculation()
        print(f"Total calculation time: {time.time() - start_time_calc} seconds")

        with open(os.path.join(res_dir, "json_output_sheetpile_part_1_stage_7.json"), 'r') as f:
            data = json.load(f)

        bending_moments = []
        for node_key in data.keys():
            node_data = data[node_key]
            if not isinstance(node_data, dict) or "BENDING_MOMENT" not in node_data:
                continue
            bending_moment = node_data["BENDING_MOMENT"]
            bending_moments.append(bending_moment)
            # shear_force = node_data["SHEAR_FORCE"]
            # axial_force = node_data["AXIAL_FORCE"]

        max_bending_moment = float(np.max(np.abs(bending_moments)))
        sampled_parameters = collect_sampled_parameters(prob_parameters)

        sample_record = {
            "sample_number": active_sample_number,
            "parameters": sampled_parameters,
            "max_bending_moment": max_bending_moment,
        }
        if replay_parameters is not None:
            sample_record["replay_of_sample_number"] = replay_sample_number
        results_dict["samples"].append(sample_record)
        append_sample_to_json(results_file, sample_record)


        print("max_bending_moment: ", max_bending_moment)
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        # make sure to change back to the original working directory and finalize gmsh even if an error occurs
        os.chdir(cwd)
        model.gmsh_io.finalize_gmsh()

        sample_record = {
            "sample_number": active_sample_number,
            "parameters": collect_sampled_parameters(prob_parameters),
            "max_bending_moment": None,
            "failure_stage": stem._Stem__last_ran_stage_number + 1
        }
        if replay_parameters is not None:
            sample_record["replay_of_sample_number"] = replay_sample_number
        results_dict["samples"].append(sample_record)
        append_sample_to_json(results_file, sample_record)

