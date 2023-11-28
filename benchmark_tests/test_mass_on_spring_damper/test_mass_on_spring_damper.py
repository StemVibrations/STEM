import os
from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import *
from stem.load import PointLoad
from stem.boundary import RotationConstraint
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, GaussPointOutput, VtkOutputParameters, Output
from stem.stem import Stem
from benchmark_tests.utils import assert_files_equal
from shutil import rmtree


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)


    spring_damper_material_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 10000, 0],
                                                            NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                                            NODAL_DAMPING_COEFFICIENT=[0, 0, 0],
                                                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    spring_damper_material = StructuralMaterial(name="spring_damper", material_parameters=spring_damper_material_parameters)

    spring_damper_coordinates = [(0, 0, 0), (0, -3, 0)]
    gmsh_input = {"spring_damper": {"coordinates": spring_damper_coordinates, "ndim": 1}}
    model.gmsh_io.generate_geometry(gmsh_input, "")

    spring_damper_model_part = BodyModelPart("spring_damper")
    spring_damper_model_part.material = spring_damper_material
    spring_damper_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "spring_damper")

    model.body_model_parts.append(spring_damper_model_part)

    # create mass element
    mass_material_parameters = NodalConcentrated(NODAL_MASS=10,
                                                 NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                                 NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    mass_material = StructuralMaterial(name="mass", material_parameters=mass_material_parameters)

    mass_coordinates = [(0, -3, 0)]
    gmsh_input = {"mass": {"coordinates": mass_coordinates, "ndim": 0}}
    model.gmsh_io.generate_geometry(gmsh_input, "")

    mass_model_part = BodyModelPart("mass")
    mass_model_part.material = mass_material
    mass_model_part.get_geometry_from_geo_data(model.gmsh_io.geo_data, "mass")

    model.body_model_parts.append(mass_model_part)

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(0, [1], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size and generate mesh
    # --------------------------------

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0, end_time=1.00, delta_time=0.01, reduction_factor=1.0,
                                       increase_factor=1.0, max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.GRAVITY_LOADING

    solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                    stress_initialisation_type=stress_initialisation_type,
                                    time_integration=time_integration,
                                    is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                    convergence_criteria=convergence_criterion, rayleigh_k=0.001,
                                    rayleigh_m=0.1)

    # Set up problem data
    problem = Problem(problem_name="calculate_mass_on_spring_damper", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

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

    model.output_settings = [vtk_output_process]

    model.post_setup()
    model.set_mesh_size(element_size=3)
    model.generate_mesh()

    input_folder = "benchmark_tests/test_mass_on_spring_damper/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()
    #
    # assert assert_files_equal("benchmark_tests/test_mass_on_sring_damper/output_/output_vtk_full_model",
    #                           os.path.join(input_folder, "output/output_vtk_full_model"))
    #
    # rmtree(input_folder)