from enum import Enum
from stem.soil_material import (
    SoilMaterial,
    TwoPhaseSoil,
    LinearElasticSoil,
    SaturatedBelowPhreaticLevelLaw,
    FluidProperties,
)
from stem.structural_material import StructuralMaterial, EulerBeam


class RailTypes(Enum):
    rail_46E3 = 1
    rail_54E1 = 2
    rail_60E1 = 3


def default_peat_material() -> SoilMaterial:
    """
    Function to define the default elastic material for peat.

    Returns:
        - :class:`stem.structural_material.SoilMaterial`
    """
    # TODO fill the right materials for PEAT
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
    """
    Function to define the default elastic material for clay.

    Returns:
        - :class:`stem.structural_material.SoilMaterial`
    """
    # TODO fill the right materials for CLAY
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
    """
    Function to define the default elastic material for a sandy soil.

    Returns:
        - :class:`stem.structural_material.SoilMaterial`
    """
    # TODO fill the right materials for SAND
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


def default_steel_rail_material(ndim: int, rail_type: RailTypes) -> StructuralMaterial:
    """
    Function to define the default elastic material for a steel beam of the given geometry.
    Currently, the most common rails adopted in the Netherlands are implemented (46E3, 54E1 and 60E1).

    Args:
        - ndim (int): number of dimensions of the problem (either 2 or 3)
        - rail_type (:class:`RailTypes`): instance of the enumeration to describe the rail type

    Returns:
        - :class:`stem.structural_material.StructuralMaterial`
    """
    if rail_type.value == 1:
        parameters = dict(CROSS_AREA=0.005944,I33=1.606e-05)
        if ndim == 3:
            parameters["I22"] = 3.075e-06
            parameters["TORSIONAL_INERTIA"] = 1.635e-05

    elif rail_type.value == 2:
        parameters = dict(CROSS_AREA=0.006977, I33=2.3372e-05)
        if ndim == 3:
            parameters["I22"] = 2.787e-06
            parameters["TORSIONAL_INERTIA"] = 2.38e-05

    elif rail_type.value == 3:
        parameters = dict(CROSS_AREA=0.00767, I33=3.038e-05)
        if ndim == 3:
            parameters["I22"] = 5.123e-06
            parameters["TORSIONAL_INERTIA"] = 3.522e-05
    else:
        raise ValueError

    name = f"default_elastic_{rail_type.name}_{ndim}D"
    beam_object = EulerBeam(
        ndim=ndim, DENSITY=7850, YOUNG_MODULUS=2.1e11, POISSON_RATIO=0.3, **parameters
    )
    return StructuralMaterial(name=name, material_parameters=beam_object)


class DefaultMaterial(Enum):
    """
    Enumeration class to retrieve default soil materials to help the user in making the model.
    """

    Peat = default_peat_material()
    Clay = default_clay_material()
    Sand = default_sand_material()

    Rail_45E3_2D = default_steel_rail_material(ndim=2, rail_type=RailTypes.rail_46E3)
    Rail_45E3_3D = default_steel_rail_material(ndim=3, rail_type=RailTypes.rail_46E3)

    Rail_54E1_2D = default_steel_rail_material(ndim=2, rail_type=RailTypes.rail_54E1)
    Rail_54E1_3D = default_steel_rail_material(ndim=3, rail_type=RailTypes.rail_54E1)

    Rail_60E1_2D = default_steel_rail_material(ndim=2, rail_type=RailTypes.rail_60E1)
    Rail_60E1_3D = default_steel_rail_material(ndim=3, rail_type=RailTypes.rail_60E1)
