import os
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import PointLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
        LinearNewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output, GaussPointOutput
from stem.stem import Stem
from stem.additional_processes import ParameterFieldParameters
from random_fields.geostatistical_cpt_interpretation import ElasticityFieldsFromCpt, RandomFieldProperties


ndim = 3
model = Model(ndim)

solid_density_1 = 2650
porosity_1 = 0.3
young_modulus_1 = 0.
poisson_ratio_1 = 0.2
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity_1)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio_1)
retention_parameters_1 = SaturatedBelowPhreaticLevelLaw()
material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, retention_parameters_1)
soil1_coordinates = [( 0.0, -25.0, -25.0),
                        ( 20.0, -25.0, -25.0),
                        ( 20.0,  -1.0, -25.0),
                        ( 1.0,  -1.0, -25.0),
                        ( 0.0,  -1.0, -25.0)]

model.extrusion_length = 50.

model.set_mesh_size(element_size=1.)

model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")

cpt_folder = r'./tests/test_data/CPTs'
orientation_x_axis = 75

elastic_field_generator_cpt = ElasticityFieldsFromCpt(
    cpt_file_folder=cpt_folder,
    based_on_midpoint=True,
    max_conditioning_points=1000,
    orientation_x_axis=orientation_x_axis,
    poisson_ratio=material_soil_1.constitutive_law.POISSON_RATIO,
    porosity=material_soil_1.POROSITY,
    water_density=material_soil_1.fluid_properties.DENSITY_FLUID,
    return_property=[RandomFieldProperties.YOUNG_MODULUS,
    RandomFieldProperties.DENSITY_SOLID],
)

elastic_field_generator_cpt.calibrate_geostat_model(calibration_indices=(1, 2), v_dim=0)

field_parameters_json = ParameterFieldParameters(
            property_name=["YOUNG_MODULUS", "DENSITY_SOLID"],
            function_type="json_file",
            field_generator=elastic_field_generator_cpt)