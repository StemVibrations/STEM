from enum import Enum
from stem.soil_material import (
    SoilMaterial,
    TwoPhaseSoil,
    LinearElasticSoil,
    SaturatedBelowPhreaticLevelLaw,
    FluidProperties,
)
from stem.structural_material import StructuralMaterial, EulerBeam


def default_peat_material() -> SoilMaterial:
    return SoilMaterial(
        name="default_elastic_peat",
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


def default_clay_material() -> SoilMaterial:
    return SoilMaterial(
        name="default_elastic_clay",
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


def default_sand_material() -> SoilMaterial:
    return SoilMaterial(
        name="default_elastic_sand",
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


def default_steel_beam_material() -> StructuralMaterial:
    return StructuralMaterial(
        name="default_elastic_steel_beam",
        material_parameters=EulerBeam(
            ndim=3,
            DENSITY=1.0,
            YOUNG_MODULUS=2.1e11,
            POISSON_RATIO=0.2,
            CROSS_AREA=1.0,
            I33=1,
            TORSIONAL_INERTIA=1,
            I22=1
        ),
    )


class DefaultMaterial(Enum):
    Peat = default_peat_material()
    Clay = default_clay_material()
    Sand = default_sand_material()
    SteelBeam = default_steel_beam_material()
