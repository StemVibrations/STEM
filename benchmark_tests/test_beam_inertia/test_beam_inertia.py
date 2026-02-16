import json
import numpy as np
import pytest
from shutil import rmtree

from stem.model import Model
from stem.model_part import BodyModelPart
from stem.structural_material import EulerBeam, StructuralMaterial
from stem.load import PointLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
      StressInitialisationType, SolverSettings, Problem, NewtonRaphsonStrategy
from stem.output import NodalOutput, JsonOutputParameters
from stem.stem import Stem

# Pameter definitions for the beam and load
BEAM_LENGTH = 25.0
LOAD_MAGNITUDE = -1000.0


@pytest.mark.parametrize("beam_coordinates, expected_node_coord",
                         [([(0, 0, 0), (0, 0, BEAM_LENGTH)], [0, 0, BEAM_LENGTH / 2]),
                          ([(0, 0, 0), (BEAM_LENGTH, 0, 0)], [BEAM_LENGTH / 2, 0, 0])])
def test_stem(beam_coordinates, expected_node_coord):
    """
    Test STEM: Point load on beam to test the inertia properties of the beam element.
    """

    ndim = 3
    model = Model(ndim)

    # Specify beam material model
    YOUNG_MODULUS = 2.87e9
    POISSON_RATIO = 0.30000
    DENSITY = 2303
    CROSS_AREA = 0.1
    I22 = 0.29
    I33 = 0.5
    torsional_inertia = 1
    beam_material = EulerBeam(ndim, YOUNG_MODULUS, POISSON_RATIO, DENSITY, CROSS_AREA, I33, I22, torsional_inertia)
    name = "beam"
    structural_material = StructuralMaterial(name, beam_material)
    # Specify the coordinates for the beam: parametrized
    node_coordinates = [expected_node_coord]

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

    # Define loads
    load = PointLoad(active=[True, True, True], value=[0, LOAD_MAGNITUDE, 0])

    model.add_load_by_coordinates(node_coordinates, load, "point_load")

    # Define displacement conditions
    displacementXYZ_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(0, [1, 2], displacementXYZ_parameters, "displacementXYZ")

    # Synchronize geometry
    model.synchronise_geometry()

    # Set mesh size and generate mesh
    # --------------------------------
    model.set_mesh_size(element_size=0.1)

    # Define project parameters
    # --------------------------------

    # Set up solver settings
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.QUASI_STATIC
    strategy = NewtonRaphsonStrategy()
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.45,
                                       delta_time=0.05,
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
                                     are_mass_and_damping_constant=False,
                                     strategy_type=strategy,
                                     convergence_criteria=convergence_criterion)

    # Set up problem data
    problem = Problem(problem_name="point_load_on_beam", number_of_threads=8, settings=solver_settings)
    model.project_parameters = problem

    # Define the output process
    json_output = JsonOutputParameters(0.05, nodal_results=[NodalOutput.DISPLACEMENT_Y])
    model.add_output_settings(json_output, "point_load")

    input_folder = r"benchmark_tests/test_beam_inertia/input_kratos"

    # Write KRATOS input files
    # --------------------------------
    stem = Stem(model, input_folder)
    stem.write_all_input_files()

    stem.run_calculation()

    # # Compare results
    calculated_output_files = ["point_load.json"]
    with open(f"{input_folder}/point_load.json", "r") as f:
        calculated_data = json.load(f)

    assert calculated_data["NODE_3"]["COORDINATES"] == expected_node_coord

    max_disp = LOAD_MAGNITUDE * BEAM_LENGTH**3 / (48 * YOUNG_MODULUS * I33)

    assert all(max_disp - np.array(calculated_data["NODE_3"]["DISPLACEMENT_Y"]) < 1e-12)

    rmtree(input_folder)
