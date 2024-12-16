import os
import json

import numpy.testing as npt
from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import *
from stem.boundary import DisplacementConstraint
from stem.solver import (AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
                         StressInitialisationType, SolverSettings, Problem)
from stem.output import NodalOutput, JsonOutputParameters
from stem.stem import Stem

from benchmark_tests.analytical_solutions.linear_spring_damper_mass import LinearSpringDamperMass
from shutil import rmtree

SHOW_RESULTS = False


def test_stem():
    # Define geometry, conditions and material parameters
    # --------------------------------

    # Specify dimension and initiate the model
    ndim = 2
    model = Model(ndim)

    spring_damper_material_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 10000, 0],
                                                            NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                                            NODAL_DAMPING_COEFFICIENT=[0, 100, 0],
                                                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    spring_damper_material = StructuralMaterial(name="spring_damper",
                                                material_parameters=spring_damper_material_parameters)

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
    displacementXYZ_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True],
                                                        value=[0, 0, 0])

    # Add boundary conditions to the model (geometry ids are shown in the show_geometry)
    model.add_boundary_condition_by_geometry_ids(0, [1], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    delta_time = 0.001
    time_integration = TimeIntegration(start_time=0.0 - delta_time,
                                       end_time=1.00,
                                       delta_time=delta_time,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-6,
                                                            displacement_absolute_tolerance=1.0e-12)
    stress_initialisation_type = StressInitialisationType.GRAVITY_LOADING

    solver_settings = SolverSettings(analysis_type=analysis_type,
                                     solution_type=solution_type,
                                     stress_initialisation_type=stress_initialisation_type,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=False,
                                     convergence_criteria=convergence_criterion,
                                     rayleigh_k=0.0,
                                     rayleigh_m=0.0)

    # Set up problem data
    problem = Problem(problem_name="calculate_mass_on_spring_damper", number_of_threads=2, settings=solver_settings)
    model.project_parameters = problem

    # Define the results to be written to the output file

    # Nodal results
    nodal_results = [NodalOutput.DISPLACEMENT_Y]
    # Gauss point results
    gauss_point_results = []

    # write output to json file
    model.add_output_settings(output_dir=".",
                              part_name="mass",
                              output_name="output_mass",
                              output_parameters=JsonOutputParameters(output_interval=delta_time,
                                                                     nodal_results=nodal_results,
                                                                     gauss_point_results=gauss_point_results))

    model.set_mesh_size(element_size=1)

    input_folder = "benchmark_tests/test_mass_on_spring_damper/inputs_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    # Run Kratos calculation
    # --------------------------------
    stem.run_calculation()

    # Assert results
    with open(r"benchmark_tests/test_mass_on_spring_damper/output_/expected_output_mass.json") as f:
        expected_data = json.load(f)

    with open(os.path.join(input_folder, "./output_mass.json")) as f:
        calculated_data = json.load(f)

    # Check if the expected displacement is equal to the calculated displacement
    npt.assert_almost_equal(calculated_data["NODE_2"]["DISPLACEMENT_Y"], expected_data["NODE_2"]["DISPLACEMENT_Y"])

    # Only calculate analytical solution and show results if SHOW_RESULTS is True
    if SHOW_RESULTS:
        import matplotlib.pyplot as plt
        # calculate spring damper mass system analytically
        end_time = time_integration.end_time
        nsteps = int(end_time / time_integration.delta_time) + 1
        analytical_solution = LinearSpringDamperMass(
            k=spring_damper_material_parameters.NODAL_DISPLACEMENT_STIFFNESS[1],
            c=spring_damper_material_parameters.NODAL_DAMPING_COEFFICIENT[1],
            m=mass_material_parameters.NODAL_MASS,
            g=9.81,
            end_time=end_time,
            n_steps=nsteps)

        analytical_solution.solve()

        # start at 0 displacement
        amplitude = analytical_solution.displacement[0]
        analytical_solution.displacement -= amplitude

        plt.plot(analytical_solution.time, analytical_solution.displacement)
        plt.plot(calculated_data["TIME"], calculated_data["NODE_2"]["DISPLACEMENT_Y"])
        plt.show()

    rmtree(input_folder)
