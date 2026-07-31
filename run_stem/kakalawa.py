input_files_dir = "kakalawa"
results_dir = "output"

import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw, SmallStrainUmatLaw, MohrCoulombLaw, SmallStrainUdsmLaw, InterfaceMaterial
from stem.structural_material import ElasticSpringDamper, NodalConcentrated, EulerBeam, Anchor
from stem.additional_processes import Excavation, ApplyFinalStressesOfPreviousStageToInitialState, \
    FindNeighboursOfInterfacesProcess, MasterSlaveConstraint
# from stem.default_materials import DefaultMaterial
from stem.load import LineLoad, WaterLineLoad, PointLoad, PointMoment
from stem.water_processes import PhreaticLineWaterPressure
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg, Amgcl, Lu, LineSearchStrategy, ResidualConvergenceCriteria
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters, GaussPointOutput, GiDOutputParameters
from stem.table import Table
from stem.stem import Stem

from stem.sheetpile_utils import SheetPileUtils

ndim = 2
model = Model(ndim)
model.mesh_settings.element_order = 2
gravity_constant = 9.81
# umat_name = r"C:\software_development\ConstitutiveModels2\build_C\lib\matsuoka_nakai.dll"


formulation_clay_clean = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=17e3/ gravity_constant, POROSITY=0.0)
constitutive_law_clay_clean = MohrCoulombLaw(YOUNG_MODULUS=14e6,
                                             POISSON_RATIO=0.2,
                                             GEO_FRICTION_ANGLE=25.2,
                                             GEO_DILATANCY_ANGLE=0.0,
                                             GEO_COHESION=28.2e3,
                                             GEO_TENSILE_STRENGTH=10e10) # change

# constitutive_law_clay_clean = SmallStrainUmatLaw(UMAT_NAME=umat_name,
#                                         IS_FORTRAN_UMAT=False,
#                                         UMAT_PARAMETERS=[14e6, 0.2, 28.2e3, 25.2, 0.0],
#                                         STATE_VARIABLES=[0])

material_clay_clean = SoilMaterial("material_clay_clean", formulation_clay_clean, constitutive_law_clay_clean,
                                   SaturatedBelowPhreaticLevelLaw())


formulation_clay_sandy = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=15e3/ gravity_constant, POROSITY=0.0)
constitutive_law_clay_sandy = MohrCoulombLaw(YOUNG_MODULUS=9e6,
                                             POISSON_RATIO=0.2,
                                             GEO_FRICTION_ANGLE=22.5,
                                             GEO_DILATANCY_ANGLE=0.0,
                                             GEO_COHESION=5.0e3,
                                             GEO_TENSILE_STRENGTH=10e10)
# constitutive_law_clay_sandy = SmallStrainUmatLaw(UMAT_NAME=umat_name,
#                                         IS_FORTRAN_UMAT=False,
#                                         UMAT_PARAMETERS=[9e6, 0.2, 5.0e6, 22.5, 0.0],
#                                         STATE_VARIABLES=[0])


material_clay_sandy = SoilMaterial("material_clay_sandy", formulation_clay_sandy, constitutive_law_clay_sandy,
                                   SaturatedBelowPhreaticLevelLaw())

gamma_unsat, gamma_sat , gamma_w = 17.4e3, 19.4e3, 9.81e3
porosity = (gamma_sat - gamma_unsat) / gamma_w
formulation_sand_clayey = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=gamma_unsat/ gravity_constant, POROSITY=porosity)
constitutive_law_sand_clayey = MohrCoulombLaw(YOUNG_MODULUS=135e6,
                                             POISSON_RATIO=0.2,
                                             GEO_FRICTION_ANGLE=37.7,
                                             GEO_DILATANCY_ANGLE=15.0,
                                             GEO_COHESION=0e0, # change
                                             GEO_TENSILE_STRENGTH=10e10)
# constitutive_law_sand_clayey = SmallStrainUmatLaw(UMAT_NAME=umat_name,
#                                         IS_FORTRAN_UMAT=False,
#                                         UMAT_PARAMETERS=[135e6, 0.2, 0, 37.7, 15.0],
#                                         STATE_VARIABLES=[0])
material_sand_clayey = SoilMaterial("material_sand_clayey", formulation_sand_clayey, constitutive_law_sand_clayey,
                                   SaturatedBelowPhreaticLevelLaw())

gamma_unsat, gamma_sat , gamma_w = 17.0e3, 19.0e3, 9.81e3
porosity = (gamma_sat - gamma_unsat) / gamma_w
formulation_sand_clean_loose = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=gamma_unsat/ gravity_constant, POROSITY=porosity)
constitutive_law_sand_clean_loose = MohrCoulombLaw(YOUNG_MODULUS=61.5e6, POISSON_RATIO=0.2, GEO_FRICTION_ANGLE=32.6,
                                                   GEO_DILATANCY_ANGLE=2.6, GEO_COHESION=0.0, GEO_TENSILE_STRENGTH=10e10)

# constitutive_law_sand_clean_loose = SmallStrainUmatLaw(UMAT_NAME=umat_name,
#                                         IS_FORTRAN_UMAT=False,
#                                         UMAT_PARAMETERS=[61.5e6, 0.2, 0, 32.6, 2.6],
#                                         STATE_VARIABLES=[0])
material_sand_clean_loose = SoilMaterial("material_sand_clean_loose", formulation_sand_clean_loose, constitutive_law_sand_clean_loose,
                                            SaturatedBelowPhreaticLevelLaw())

gamma_unsat, gamma_sat , gamma_w = 18.0e3, 20.0e3, 9.81e3
porosity = (gamma_sat - gamma_unsat) / gamma_w
formulation_sand_clean_mid = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=gamma_unsat/ gravity_constant, POROSITY=porosity)
constitutive_law_sand_clean_mid = MohrCoulombLaw(YOUNG_MODULUS=123.3e6, POISSON_RATIO=0.2, GEO_FRICTION_ANGLE=38.3,
                                                 GEO_DILATANCY_ANGLE=8.3, GEO_COHESION=0e0, GEO_TENSILE_STRENGTH=10e10) # change
# constitutive_law_sand_clean_mid = SmallStrainUmatLaw(UMAT_NAME=umat_name,
#                                         IS_FORTRAN_UMAT=False,
#                                         UMAT_PARAMETERS=[123.3e6, 0.2, 0, 38.3, 8.3],
#                                         STATE_VARIABLES=[0])
material_sand_clean_mid = SoilMaterial("material_sand_clean_mid", formulation_sand_clean_mid, constitutive_law_sand_clean_mid,
                                            SaturatedBelowPhreaticLevelLaw())

gamma_unsat, gamma_sat , gamma_w = 19.0e3, 21.0e3, 9.81e3
porosity = (gamma_sat - gamma_unsat) / gamma_w
formulation_sand_clean_dense = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=gamma_unsat/ gravity_constant, POROSITY=porosity)
constitutive_law_sand_clean_dense = MohrCoulombLaw(YOUNG_MODULUS=226.5e6, POISSON_RATIO=0.2, GEO_FRICTION_ANGLE=37.8,
                                                    GEO_DILATANCY_ANGLE=7.8, GEO_COHESION=0.0, GEO_TENSILE_STRENGTH=10e10)
# constitutive_law_sand_clean_dense = SmallStrainUmatLaw(UMAT_NAME=umat_name,
#                                         IS_FORTRAN_UMAT=False,
#                                         UMAT_PARAMETERS=[226.5e6, 0.2, 0, 37.8, 7.8],
#                                         STATE_VARIABLES=[0])

material_sand_clean_dense = SoilMaterial("material_sand_clean_dense", formulation_sand_clean_dense, constitutive_law_sand_clean_dense,
                                         SaturatedBelowPhreaticLevelLaw())


formulation_concrete = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=0.1/ gravity_constant, POROSITY=0.0)
constitutive_law_concrete = MohrCoulombLaw(YOUNG_MODULUS=500e6, POISSON_RATIO=0.0, GEO_FRICTION_ANGLE=0.0,
                                                    GEO_DILATANCY_ANGLE=0.0, GEO_COHESION=10e6, GEO_TENSILE_STRENGTH=10e10)

material_concrete = SoilMaterial("material_concrete", formulation_concrete, constitutive_law_concrete,
                                   SaturatedBelowPhreaticLevelLaw())


w = 12e3
EA = 12e9
EI = 640e6
d = 0.8

E = EA / d
I = EI / E
cross_area = d * 1
density = w/ (d * 1 * gravity_constant)
sheetpile_E15 = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.15, CROSS_AREA=cross_area, I33=I, DENSITY=density)

w = 12e3
EA = 28.56e9
EI = 1.523e9
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
bottom_plate = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.15, CROSS_AREA=cross_area, I33=I, DENSITY=density)


EA = 441e6
cross_area = cross_area
spacing = 3.6
E = EA / spacing / cross_area
anchor = Anchor(YOUNG_MODULUS=E, CROSS_AREA=cross_area, PRESTRESS=1000e3/cross_area/spacing)
# anchor = Anchor(YOUNG_MODULUS=E, CROSS_AREA=cross_area)

gamma_grout = 26e3
spacing=3.6
cross_area = 0.02688
I33 = 0.05750e-3
E = 16.41e9 / spacing

grout = EulerBeam(2, YOUNG_MODULUS=E, POISSON_RATIO=0.15, CROSS_AREA=cross_area, I33=I33, DENSITY=gamma_grout/ (cross_area * gravity_constant)/ spacing)



no_interface_constitutive_law = LinearElasticSoil(YOUNG_MODULUS=1e15, POISSON_RATIO=0.2)

# interfaces
# interface_sand_clayey_material = InterfaceMaterial("interface_sand_clayey_material", constitutive_law=constitutive_law_sand_clayey, soil_formulation=formulation_sand_clayey, retention_parameters=SaturatedBelowPhreaticLevelLaw())
# interface_clay_clean_material = InterfaceMaterial("interface_clay_clean_material", constitutive_law=constitutive_law_clay_clean, soil_formulation=formulation_clay_clean, retention_parameters=SaturatedBelowPhreaticLevelLaw())
# interface_clay_sandy_material = InterfaceMaterial("interface_clay_sandy_material", constitutive_law=constitutive_law_clay_sandy, soil_formulation=formulation_clay_sandy, retention_parameters=SaturatedBelowPhreaticLevelLaw())
# interface_sand_clean_loose_material = InterfaceMaterial("interface_sand_clean_loose_material", constitutive_law=constitutive_law_sand_clean_loose, soil_formulation=formulation_sand_clean_loose, retention_parameters=SaturatedBelowPhreaticLevelLaw())
# interface_sand_clean_mid_material = InterfaceMaterial("interface_sand_clean_mid_material", constitutive_law=constitutive_law_sand_clean_mid, soil_formulation=formulation_sand_clean_mid, retention_parameters=SaturatedBelowPhreaticLevelLaw())
# interface_sand_clean_dense_material = InterfaceMaterial("interface_sand_clean_dense_material", constitutive_law=constitutive_law_sand_clean_dense, soil_formulation=formulation_sand_clean_dense, retention_parameters=SaturatedBelowPhreaticLevelLaw())

interface_sand_clayey_material = InterfaceMaterial("interface_sand_clayey_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clayey, retention_parameters=SaturatedBelowPhreaticLevelLaw())
interface_clay_clean_material = InterfaceMaterial("interface_clay_clean_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_clay_clean, retention_parameters=SaturatedBelowPhreaticLevelLaw())
interface_clay_sandy_material = InterfaceMaterial("interface_clay_sandy_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_clay_sandy, retention_parameters=SaturatedBelowPhreaticLevelLaw())
interface_sand_clean_loose_material = InterfaceMaterial("interface_sand_clean_loose_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clean_loose, retention_parameters=SaturatedBelowPhreaticLevelLaw())
interface_sand_clean_mid_material = InterfaceMaterial("interface_sand_clean_mid_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clean_mid, retention_parameters=SaturatedBelowPhreaticLevelLaw())
interface_sand_clean_dense_material = InterfaceMaterial("interface_sand_clean_dense_material", constitutive_law=no_interface_constitutive_law, soil_formulation=formulation_sand_clean_dense, retention_parameters=SaturatedBelowPhreaticLevelLaw())
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

bottom_plat_coordinates = [(mid_left, bottom_sheetpile_coord, 0), (mid_right, bottom_sheetpile_coord, 0)]

anchor_coordinates = [(0,2,0),(24.8,-22.8,0)]
grout_coordinates = [(24.8,-22.8,0), (36.82,-34.82,0)]
#
SheetPileUtils.add_sheetpile_by_coordinates(sheetpile_coordinates_part_1,sheetpile_E357, "sheetpile_part_1", model.gmsh_io,model.body_model_parts)
SheetPileUtils.add_sheetpile_by_coordinates(sheetpile_coordinates_part_2,sheetpile_E15, "sheetpile_part_2", model.gmsh_io,model.body_model_parts)
SheetPileUtils.add_sheetpile_by_coordinates(sheetpile_coordinates_part_3,sheetpile_E357, "sheetpile_part_3", model.gmsh_io,model.body_model_parts)

# SheetPileUtils.add_sheetpile_by_coordinates(bottom_plat_coordinates,bottom_plate, "bottom_plate", model.gmsh_io,model.body_model_parts)

SheetPileUtils.add_anchor_by_coordinates(anchor_coordinates,anchor,0.0, "anchor", model.gmsh_io,model.body_model_parts)
SheetPileUtils.add_grout_by_coordinates(grout_coordinates, grout,None, "grout", model.gmsh_io,model.body_model_parts)
# model.show_geometry(show_line_ids=True)

# model.set_interface_between_model_parts(["grout"],["sand_clean_dense"], interface_sand_clean_dense_material, "interface_sand_clean_dense_grout")
model.set_interface_between_model_parts(["sand_clean_dense"], ["sand_clean_dense_mid","sheetpile_part_3"], interface_sand_clean_dense_material, "interface_sand_clean_dense")
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
# model.apply_additional_process(excavation_sheetpiles, "bottom_plate")

model.apply_additional_process(excavation_anchor, "anchor")
model.apply_additional_process(excavation_anchor, "grout")

model.apply_additional_process(interface_excavation, "interface_sand_clean_loose")
model.apply_additional_process(interface_excavation, "interface_clay_sandy")
model.apply_additional_process(interface_excavation, "interface_sand_clean_mid")
model.apply_additional_process(interface_excavation, "interface_sand_clayey")
model.apply_additional_process(interface_excavation, "interface_sand_clean_mid_low")
model.apply_additional_process(interface_excavation, "interface_clay_clean_mid")
model.apply_additional_process(interface_excavation, "interface_sand_clean_dense")


# vertical_point_load = PointLoad(active=[True, True, True], value=[0, -177.2e3, 0])
# point_moment = PointMoment(active=[False, False, True], value=[0, 0, 90.4e3])
vertical_point_load = PointLoad(active=[True, True, True], value=[0, 0, 0])
point_moment = PointMoment(active=[False, False, True], value=[0, 0, 0])
model.add_load_by_coordinates([(0.0,4.0,0.0)], vertical_point_load, "vertical_point_load")
model.add_load_by_coordinates([(0.0,4.0,0.0)], point_moment, "point_moment")

top_line_load_1 = LineLoad(active=[True, True, True], value=[0, 0.0, 0])
model.add_load_by_coordinates([(0.4,5.5,0.0),(20,5.5,0.0)], top_line_load_1, "top_line_load_1")

phreatic_level = 0.24
phreatic_line = PhreaticLineWaterPressure([(-100, phreatic_level, 0), (100, phreatic_level, 0)],
                                          is_fixed=True)
model.add_phreatic_line(phreatic_line)
#
# model.show_geometry(show_line_ids=True)

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



end_time = 1
delta_time = 0.01
time_integration = TimeIntegration(start_time=0.0,
                                   end_time=end_time,
                                   delta_time=delta_time,
                                   reduction_factor=0.5,
                                   increase_factor=1.5,
                                   max_delta_time_factor=1000)

# set convergence criteria
# convergence_criterion = ResidualConvergenceCriteria(residual_relative_tolerance=1.0e-3,
#                                                         residual_absolute_tolerance=1.0e-12)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-5,
                                                        displacement_absolute_tolerance=1.0e-12)

linear_solver = Lu()
# linear_solver = Cg(scaling=False, preconditioner_type="none")
# linear_solver = Amgcl(smoother_type="damped_jacobi", krylov_type="cg", tolerance=1e-12, max_iteration=1000)
# set solver settings stage 1
solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                 solution_type=SolutionType.QUASI_STATIC,
                                 stress_initialisation_type=StressInitialisationType.GRAVITY_LOADING,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False,
                                 are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=NewtonRaphsonStrategy(),
                                 linear_solver_settings=linear_solver)

# Set up problem data
problem = Problem(problem_name="sheetpile", number_of_threads=16, settings=solver_settings)
model.project_parameters = problem

model.mesh_settings.element_size=3
model.set_element_size_of_group(1, "sand_clean_loose_mid")
model.set_element_size_of_group(1, "clay_sandy_mid")
model.set_element_size_of_group(1, "sand_clean_mid_mid")
model.set_element_size_of_group(1, "sand_clayey_mid")
model.set_element_size_of_group(1, "sand_clean_mid_low_mid")
model.set_element_size_of_group(1, "clay_clean_mid_mid")
model.set_element_size_of_group(1, "sand_clean_dense_mid")
model.set_element_size_of_group(1, "sand_clayey_part_2_left")

# model.set_element_size_of_group(0.5, "water_normal_load_soil")

# model.apply_additional_process(FindNeighboursOfInterfacesProcess(model_part_name_list=["interface_sand_clean_loose", "interface_clay_sandy", "interface_sand_clean_mid"
# , "interface_sand_clayey", "interface_sand_clean_mid_low", "interface_clay_clean_mid", "interface_sand_clean_dense"]))

stem = Stem(model, input_files_dir)


# stage 2 install sheetpile
stage2 = stem.create_new_stage(0.1,1)
stage2.project_parameters.settings.convergence_criteria.displacement_relative_tolerance= 5e-4

sheetpile_casing_mps = []
sheetpile_casing_mps.append(stage2.get_model_part_by_name("sand_clean_loose_mid"))
sheetpile_casing_mps.append(stage2.get_model_part_by_name("clay_sandy_mid"))
sheetpile_casing_mps.append(stage2.get_model_part_by_name("sand_clean_mid_mid"))
sheetpile_casing_mps.append(stage2.get_model_part_by_name("sand_clayey_mid"))
sheetpile_casing_mps.append(stage2.get_model_part_by_name("sand_clean_mid_low_mid"))
sheetpile_casing_mps.append(stage2.get_model_part_by_name("clay_clean_mid_mid"))
sheetpile_casing_mps.append(stage2.get_model_part_by_name("sand_clean_dense_mid"))

for mp in sheetpile_casing_mps:
    mp.material = material_concrete

stage2.get_additional_process_part_by_name_and_type("sheetpile_part_1", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("sheetpile_part_2", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("sheetpile_part_3", Excavation).parameters.deactivate_body_model_part = False
# stage2.get_additional_process_part_by_name_and_type("bottom_plate", Excavation).parameters.deactivate_body_model_part = False

import copy



stage2.get_model_part_by_name("interface_sand_clean_loose").material.constitutive_law = copy.deepcopy(constitutive_law_sand_clean_loose)
r_inter = 0.57
stage2.get_model_part_by_name("interface_sand_clean_loose").material.constitutive_law.GEO_FRICTION_ANGLE =r_inter * constitutive_law_sand_clean_loose.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_sand_clean_loose").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_sand_clean_loose.GEO_COHESION

r_inter = 0.88
stage2.get_model_part_by_name("interface_clay_sandy").material.constitutive_law = copy.deepcopy(constitutive_law_clay_sandy)
stage2.get_model_part_by_name("interface_clay_sandy").material.constitutive_law.GEO_FRICTION_ANGLE = r_inter * constitutive_law_clay_sandy.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_clay_sandy").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_clay_sandy.GEO_COHESION

r_inter = 0.46
stage2.get_model_part_by_name("interface_sand_clean_mid").material.constitutive_law = copy.deepcopy(constitutive_law_sand_clean_mid)
stage2.get_model_part_by_name("interface_sand_clean_mid").material.constitutive_law.GEO_FRICTION_ANGLE = r_inter * constitutive_law_sand_clean_mid.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_sand_clean_mid").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_sand_clean_mid.GEO_COHESION

r_inter = 0.47
stage2.get_model_part_by_name("interface_sand_clayey").material.constitutive_law = copy.deepcopy(constitutive_law_sand_clayey)
stage2.get_model_part_by_name("interface_sand_clayey").material.constitutive_law.GEO_FRICTION_ANGLE =r_inter * constitutive_law_sand_clayey.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_sand_clayey").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_sand_clayey.GEO_COHESION

r_inter = 0.46
stage2.get_model_part_by_name("interface_sand_clean_mid_low").material.constitutive_law = copy.deepcopy(constitutive_law_sand_clean_mid)
stage2.get_model_part_by_name("interface_sand_clean_mid_low").material.constitutive_law.GEO_FRICTION_ANGLE =r_inter * constitutive_law_sand_clean_mid.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_sand_clean_mid_low").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_sand_clean_mid.GEO_COHESION

r_inter = 0.77
stage2.get_model_part_by_name("interface_clay_clean_mid").material.constitutive_law = copy.deepcopy(constitutive_law_clay_clean)
stage2.get_model_part_by_name("interface_clay_clean_mid").material.constitutive_law.GEO_FRICTION_ANGLE = r_inter * constitutive_law_clay_clean.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_clay_clean_mid").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_clay_clean.GEO_COHESION

r_inter = 0.47
stage2.get_model_part_by_name("interface_sand_clean_dense").material.constitutive_law = copy.deepcopy(constitutive_law_sand_clean_dense)
stage2.get_model_part_by_name("interface_sand_clean_dense").material.constitutive_law.GEO_FRICTION_ANGLE =r_inter * constitutive_law_sand_clean_dense.GEO_FRICTION_ANGLE
stage2.get_model_part_by_name("interface_sand_clean_dense").material.constitutive_law.GEO_COHESION = r_inter * constitutive_law_sand_clean_dense.GEO_COHESION


stage2.get_additional_process_part_by_name_and_type("interface_sand_clean_loose", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("interface_clay_sandy", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("interface_sand_clean_mid", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("interface_sand_clayey", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("interface_sand_clean_mid_low", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("interface_clay_clean_mid", Excavation).parameters.deactivate_body_model_part = False
stage2.get_additional_process_part_by_name_and_type("interface_sand_clean_dense", Excavation).parameters.deactivate_body_model_part = False

# remove master slave, since interface is active
stage2.remove_model_part_by_name_and_type("", MasterSlaveConstraint)


stage2.apply_additional_process(FindNeighboursOfInterfacesProcess(model_part_name_list=["interface_sand_clean_loose", "interface_clay_sandy", "interface_sand_clean_mid"
, "interface_sand_clayey", "interface_sand_clean_mid_low", "interface_clay_clean_mid", "interface_sand_clean_dense"]))

stem.add_calculation_stage(stage2)

#stage3 excavation part 1
stage3 = stem.create_new_stage(0.1,1)
stage3.get_additional_process_part_by_name_and_type("sand_clean_loose_left", Excavation).parameters.deactivate_body_model_part = True
stage3.get_additional_process_part_by_name_and_type("clay_sandy_left", Excavation).parameters.deactivate_body_model_part = True
stage3.get_additional_process_part_by_name_and_type("sand_clean_mid_part_1_left", Excavation).parameters.deactivate_body_model_part = True
#
#
#
move_stress_sheetpile = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["sheetpile_part_1", "sheetpile_part_2", "sheetpile_part_3"])
stage3.apply_additional_process(move_stress_sheetpile)

move_stress_interface = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["interface_sand_clean_loose", "interface_clay_sandy", "interface_sand_clean_mid",
                                                                                    "interface_sand_clayey", "interface_sand_clean_mid_low", "interface_clay_clean_mid","interface_sand_clean_dense" ])
stage3.apply_additional_process(move_stress_interface)
stage3.remove_model_part_by_name_and_type("", FindNeighboursOfInterfacesProcess)

stem.add_calculation_stage(stage3)


# stage 4 add anchor and grout
stage4 = stem.create_new_stage(0.1,1)
stage4.get_additional_process_part_by_name_and_type("anchor", Excavation).parameters.deactivate_body_model_part = False
stage4.get_additional_process_part_by_name_and_type("grout", Excavation).parameters.deactivate_body_model_part = False

# stage4.get_model_part_by_name("interface_sand_clean_dense_grout").material.constitutive_law = constitutive_law_sand_clean_dense
# stage4.get_model_part_by_name("interface_sand_clean_dense_grout").material.constitutive_law.GEO_COHESION = 50000
# stage4.get_model_part_by_name("interface_sand_clean_dense_grout").material.constitutive_law.YOUNG_MODULUS = 500e6
stem.add_calculation_stage(stage4)

#stage 5 add point load and moment
stage5 = stem.create_new_stage(0.01,1)
move_stress_anchor = ApplyFinalStressesOfPreviousStageToInitialState(model_part_name_list=["anchor", "grout"])
stage5.apply_additional_process(move_stress_anchor)
anchor_model_part = stage5.get_model_part_by_name("anchor")

# prestress is set to half because here, because the stress initialisation is wrong. In this case, the prestress stays
# as it should be
anchor_model_part.material.material_parameters.PRESTRESS = anchor_model_part.material.material_parameters.PRESTRESS/2


stage5.get_model_part_by_name("vertical_point_load").parameters.value = [0, -177.2e3, 0]
stage5.get_model_part_by_name("point_moment").parameters.value = [0, 0, 90.4e3]
# stage5.project_parameters.settings.convergence_criteria.displacement_relative_tolerance= 1e-4

stem.add_calculation_stage(stage5)

#stage 6 excavation part 2
stage6 = stem.create_new_stage(0.1,1)
stage6.get_additional_process_part_by_name_and_type("sand_clean_mid_part_2_left", Excavation).parameters.deactivate_body_model_part = True
stage6.get_additional_process_part_by_name_and_type("sand_clayey_part_1_left", Excavation).parameters.deactivate_body_model_part = True
stage6.get_additional_process_part_by_name_and_type("sand_clean_mid_low_part_1", Excavation).parameters.deactivate_body_model_part = True
stage6.get_additional_process_part_by_name_and_type("clay_clean_mid_part_1", Excavation).parameters.deactivate_body_model_part = True

#set water load
stage6.get_model_part_by_name("water_normal_load_soil").parameters.active = True
stage6.get_model_part_by_name("water_normal_load_beam").parameters.active = True
# stage6.get_model_part_by_name("water_normal_load_soil").parameters.reference_coordinate = [0,5,0]
# stage6.get_model_part_by_name("init_water_normal_load_soil").parameters.active = False

anchor_model_part = stage6.get_model_part_by_name("anchor")
# prestress is set to half because here, because the stress initialisation is wrong. In this case, the prestress stays
# as it should be
anchor_model_part.material.material_parameters.PRESTRESS = anchor_model_part.material.material_parameters.PRESTRESS/2

stage6.get_model_part_by_name("top_line_load_1").parameters.value = [0, -10e3, 0]



# this step is close to instable, so we need to relax the convergence criteria
stage6.project_parameters.settings.convergence_criteria.displacement_relative_tolerance= 5e-3
# stage6.project_parameters.settings.quasi_newton_max_rank = 3
# stage6.project_parameters.settings.quasi_newton_restart_interval=50
stage6.project_parameters.settings.quasi_newton_type = "none"
stage6.project_parameters.settings.strategy_type.max_iterations = 1000
stem.add_calculation_stage(stage6)


stem.write_all_input_files()
stem.run_calculation()

