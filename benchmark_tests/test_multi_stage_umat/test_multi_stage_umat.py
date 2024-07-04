import os

import numpy as np

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw, \
    SmallStrainUmatLaw
from stem.load import LineLoad, MovingLoad, PointLoad
from stem.table import Table
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def test_stem():
    """
    Test STEM: 2D block with distributed loading with multistage for the umat using umat and changing the stiffness
    of the material in the second stage (halved).

    It still needs to be checked for the different formulation type of the elements in 2D (incremental, required
    for K_0 stress initialisation) and 3D (non-incremental).
    """

    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model_stage_1 = Model(ndim)

    # Specify material model
    # Linear elastic drained soil with a Density of 2700, a Young's modulus of 50e6,
    # a Poisson ratio of 0.3 & a Porosity of 0.3 is specified.
    DENSITY_SOLID = 2700
    POROSITY = 0.3
    YOUNG_MODULUS = 50e6
    POISSON_RATIO = 0.3
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:1m x y:0.5m
    layer1_coordinates = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.0, 0.5, 0.0)]

    # Create the soil layer
    model_stage_1.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_block")

    # Boundary conditions and Loads
    # a sinus load at 20Hz (load period T=0.05s)
    load_frequency = 20  # Hz
    delta_time = 0.005  # s = 10 points per cycle
    total_simulation_time = 0.5  # s =  10 cycles
    load_pulse = load_frequency * (2 * np.pi)  # rad/s

    t = np.arange(0, 0.5 + delta_time, delta_time)  # s
    values = -1000 * np.sin(load_pulse * t)  # N

    LOAD_Y = Table(times=t, values=values)
    # Add line load
    load_coordinates = [(0.0, 0.5, 0.0), (1.0, 0.5, 0.0)]
    load = LineLoad(value=[0, LOAD_Y, 0], active=[False, True, False])
    model_stage_1.add_load_by_coordinates(load_coordinates, load, "point_load")

    # Define boundary conditions
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    sym_parameters = DisplacementConstraint(active=[True, False, True], is_fixed=[True, False, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model_stage_1.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model_stage_1.add_boundary_condition_by_geometry_ids(1, [2, 4], sym_parameters, "side_rollers")

    # Synchronize geometry
    model_stage_1.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model_stage_1.set_mesh_size(element_size=0.5)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC

    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=total_simulation_time / 2,
                                       delta_time=delta_time,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0E-12,
                                                            displacement_absolute_tolerance=1.0E-6)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=1e-03,
                                     rayleigh_m=0.02)

    # Set up problem data
    problem = Problem(problem_name="test_multi_stage_umat", number_of_threads=2, settings=solver_settings)
    model_stage_1.project_parameters = problem

    # Define the results to be written to the output file
    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

    # Define the output process
    model_stage_1.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                            output_interval=1,
                                                                            nodal_results=nodal_results,
                                                                            gauss_point_results=[]),
                                      output_dir="output",
                                      output_name="vtk_output")

    # define the STEM instance
    input_folder = "benchmark_tests/test_multi_stage_umat/inputs_kratos"
    stem = Stem(model_stage_1, input_folder)

    # create new stage:
    # the new material parameters have a Young's modulus half of the stage 1 material
    model_stage_2 = stem.create_new_stage(delta_time, total_simulation_time / 2)

    YOUNG_MODULUS_2 = YOUNG_MODULUS / 2
    SHEAR_MODULUS = YOUNG_MODULUS_2 / (2 * (1 + POISSON_RATIO))

    # soil_formulation_stage_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    # constitutive_law_stage_2 = SmallStrainUmatLaw(UMAT_NAME="../../MohrCoulombUMAT.dll",
    #                                        IS_FORTRAN_UMAT=True,
    #                                        UMAT_PARAMETERS=[1e8,
    #                                             0.0,
    #                                             1e6,
    #                                             30,
    #                                             0.0,
    #                                             1e6,
    #                                             1,
    #                                             0.0],
    #                                        STATE_VARIABLES=[0.0])

    soil_formulation_stage_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law_stage_2 = SmallStrainUmatLaw(UMAT_NAME="../linear_elastic.dll",
                                                  IS_FORTRAN_UMAT=True,
                                                  UMAT_PARAMETERS=[SHEAR_MODULUS, POISSON_RATIO],
                                                  STATE_VARIABLES=[0.0])

    retention_parameters_stage_2 = SaturatedBelowPhreaticLevelLaw()
    material_stage_2 = SoilMaterial("soil2", soil_formulation_stage_2, constitutive_law_stage_2,
                                    retention_parameters_stage_2)

    model_stage_2.body_model_parts[0].material = material_stage_2

    # add the new stage to the calculation
    stem.add_calculation_stage(model_stage_2)

    # write the kratos input files
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    result = assert_files_equal("benchmark_tests/test_multi_stage_umat/output_/output_vtk_full_model",
                                os.path.join(input_folder, "output/output_vtk_full_model"))

    assert result is True
    rmtree(input_folder)
