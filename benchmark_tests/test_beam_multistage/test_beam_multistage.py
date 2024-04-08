import os
import numpy as np
import pytest
from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import EulerBeam, StructuralMaterial
from stem.load import MovingLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output, JsonOutputParameters
from stem.stem import Stem
from shutil import rmtree, copytree

from benchmark_tests.utils import assert_floats_in_directories_almost_equal

PLOT_RESULTS = True


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    velocity = 100 / 3.6

    # Specify beam material model
    YOUNG_MODULUS = 2.87e9
    POISSON_RATIO = 0.30000
    DENSITY = 2303
    CROSS_AREA = 0.1
    I22 = 0.29
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I22)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    beam_coordinates = [(0, 0, 0), (25, 0, 0)]
    # Create the beam
    gmsh_input = {name: {"coordinates": beam_coordinates, "ndim": 1}}
    # check if extrusion length is specified in 3D
    model.gmsh_io.generate_geometry(gmsh_input, "")
    #
    # create body model part
    body_model_part = BodyModelPart(name)
    body_model_part.material = structural_material

    # set the geometry of the body model part
    body_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, name)
    model.body_model_parts.append(body_model_part)

    moving_load = MovingLoad(load=[0, -80000, 0], velocity=velocity, origin=[0, 0, 0], direction=[1, 0, 0])
    model.add_load_on_line_model_part("beam", moving_load, "point_load")

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()
    # model.show_geometry(show_line_ids=True, show_point_ids=True)

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=0.1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0, end_time=0.45, delta_time=0.05, reduction_factor=1.0,
                                       increase_factor=1.0, max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.NONE
    solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True, are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.000,
                                     rayleigh_m=0.00)

    # Set up problem data
    problem = Problem(problem_name="point_load_on_beam", number_of_threads=1, settings=solver_settings)
    model.project_parameters = problem

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = []

    # Define the output process
    vtk_output_process = Output(
        output_name="vtk_output",
        output_dir="output",
        output_parameters=VtkOutputParameters(
            file_format="ascii",
            output_interval=1,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
            output_control_type="step"
        )
    )

    json_output = JsonOutputParameters(0.049, nodal_results = [NodalOutput.DISPLACEMENT_Y])

    model.add_output_settings(json_output, "point_load" )
    # model.output_settings = [vtk_output_process]
    #
    input_folder = r"benchmark_tests/test_beam_multistage/input_kratos"


    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    model_stage_2 = stem.create_new_stage(0.005, 0.45)
    stem.add_calculation_stage(model_stage_2)

    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()



    # # # test output
    # assert_floats_in_files_almost_equal("benchmark_tests/test_sdof_uvec_beam/output_/output_vtk_full_model",
    #                                     os.path.join(input_folder, "output/output_vtk_full_model"), decimal=3)
    #
    # rmtree(input_folder)

    # assert_files_equal("benchmark_tests/test_sdof_uvec_beam/output_/output_vtk_full_model",
