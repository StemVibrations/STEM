import os
from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import *
from stem.load import MovingLoad
from stem.boundary import RotationConstraint
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, GaussPointOutput, JsonOutputParameters
from stem.stem import Stem


def run_moving_load_on_beam(input_folder, ndim):
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    model = Model(ndim)

    # Specify beam material model
    YOUNG_MODULUS = 210e9
    POISSON_RATIO = 0.30000
    DENSITY = 7850
    CROSS_AREA = 0.01
    I22 = 0.0001
    I33 = 0.0001

    velocity_moving_load = 10
    length_beam = 25

    TORTIONAL_INERTIA = I22 + I33
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I33, I22, TORTIONAL_INERTIA)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: x:1m x y:0m
    beam_coordinates = [(0, 0, 0), (length_beam, 0, 0)]
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

    # Show geometry and geometry ids
    # model.show_geometry(show_point_ids=True, show_line_ids=True)

    # Define moving load
    moving_load = MovingLoad(load=[0.0, 1000.0, 0.0],
                             direction_signs=[1, 1, 1],
                             velocity=velocity_moving_load,
                             origin=[0.0, 0.0, 0.0],
                             offset=0.0)

    model.add_load_by_geometry_ids([1], moving_load, "moving_load")

    # Define rotation boundary condition
    no_torsion_parameters = RotationConstraint(is_fixed=[True, False, False], value=[0, 0, 0])

    # Define displacement conditions
    no_vert_displacement = DisplacementConstraint(is_fixed=[False, True, False], value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(0, [1, 2], no_torsion_parameters, "no_torsion")
    model.add_boundary_condition_by_geometry_ids(0, [1, 2], no_vert_displacement, "no_vert_displacement")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size
    # --------------------------------
    model.set_mesh_size(element_size=2.5)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC

    end_time = length_beam / velocity_moving_load
    dt = end_time / 1000
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=end_time,
                                       delta_time=dt,
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
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.000,
                                     rayleigh_m=0.0)

    # Set up problem data
    problem = Problem(problem_name="calculate_moving_load_on_beam3D", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT]
    # Gauss point results
    gauss_point_results = [GaussPointOutput.FORCE]

    JsonOutputParameters(dt, nodal_results)
    model.add_output_settings_by_coordinates([(length_beam / 2, 0, 0)],
                                             output_parameters=JsonOutputParameters(dt, nodal_results),
                                             part_name=f"json_output_{ndim}D")

    # # Define the output process
    # model.add_output_settings(output_parameters=VtkOutputParameters(file_format="ascii",
    #                                                                 output_interval=10,
    #                                                                 nodal_results=nodal_results,
    #                                                                 gauss_point_results=gauss_point_results,
    #                                                                 output_control_type="step"),
    #                           output_dir="output",
    #                           output_name="vtk_output")

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()
