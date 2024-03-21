import os

from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import *
from stem.load import MovingLoad
from stem.boundary import RotationConstraint
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.stem import Stem

from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    # Specify soil material model
    DENSITY_SOLID = 2650
    POROSITY = 0.3
    YOUNG_MODULUS = 0.5e6
    POISSON_RATIO = 0.2
    soil_formulation1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law1 = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters1 = SaturatedBelowPhreaticLevelLaw()
    material1 = SoilMaterial("soil", soil_formulation1, constitutive_law1, retention_parameters1)

    # Specify the coordinates for the column: x:5m x y:1m
    layer1_coordinates = [(0.0, -0.1, 0.0), (1.0, -0.1, 0.0), (1.0, 0, 0.0), (0.0, 0, 0.0)]

    # Create the soil layer
    model.add_soil_layer_by_coordinates(layer1_coordinates, material1, "soil_layer")

    # Specify beam material model
    YOUNG_MODULUS = 210000000000
    POISSON_RATIO = 0.3
    DENSITY = 7850
    CROSS_AREA = 0.01
    I22 = 0.00001
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I22)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)

    # Specify the coordinates for the beam: x:1m x y:0m
    beam_coordinates = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    # Create the beam
    gmsh_input = {name: {"coordinates": beam_coordinates, "ndim": 1}}
    model.gmsh_io.generate_geometry(gmsh_input, "")
    #
    # create body model part
    body_model_part = BodyModelPart(name)
    body_model_part.material = structural_material

    # set the geometry of the body model part
    body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, name)
    model.body_model_parts.append(body_model_part)

    # Synchronize geometry
    # model.synchronise_geometry()
    # model.show_geometry(show_line_ids=True, show_point_ids=True)

    # Define moving load
    load_coordinates = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

    moving_load = MovingLoad(load=["0.0", "-10000*t", "0.0"], direction=[1, 0, 0], velocity=1.0, origin=[0.0, 0.0, 0.0])
    model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

    # Define rotation boundary condition
    rotation_boundaries_parameters = RotationConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(0, [4], rotation_boundaries_parameters, "rotation")
    model.add_boundary_condition_by_geometry_ids(1, [1, 4, 2], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()

    # Show geometry and geometry ids
    # model.show_geometry(show_line_ids=True, show_point_ids=True)

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=0.05)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL_GROUNDWATER_FLOW
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=1.0,
                                       delta_time=0.01,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=False,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.001,
                                     rayleigh_m=0.1)

    # Set up problem data
    problem = Problem(problem_name="calculate_moving_load_on_beam", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
                                                                    output_interval=10,
                                                                    nodal_results=nodal_results,
                                                                    gauss_point_results=gauss_point_results,
                                                                    output_control_type="step"),
                              output_dir="output",
                              output_name="vtk_output")

    input_folder = "benchmark_tests/test_moving_load_on_beam_on_soil/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    assert assert_files_equal("benchmark_tests/test_moving_load_on_beam_on_soil/output_/output_vtk_full_model",
                              os.path.join(input_folder, "output/output_vtk_full_model"))

    rmtree(input_folder)
