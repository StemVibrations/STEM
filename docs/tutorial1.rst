.. _tutorial1:

Lamb's problem in 3D
====================

Overview
--------
This tutorial shows a step-by-step guide on how to set up and run a 3D Lamb problem, that
consists on the application of a point load on the surface, and computing the wave propagation
towards the free field.
To avoid reflections at the model edges, absorbing boundaries are used.

Imports and setup
-----------------
First the necessary packages are imported and the input folder is defined.

.. code-block:: python

    input_files_dir = "lamb"

    from stem.model import Model
    from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
    from stem.load import PointLoad
    from stem.boundary import DisplacementConstraint, AbsorbingBoundary
    from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
        StressInitialisationType, SolverSettings, Problem, LinearNewtonRaphsonStrategy, Cg
    from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
    from stem.stem import Stem

    # END CODE BLOCK

For setting up the model, ``Model`` is imported from ``stem.model``.
For the soil material, ``OnePhaseSoil``, ``LinearElasticSoil``, ``SoilMaterial``,
and ``SaturatedBelowPhreaticLevelLaw`` are imported from ``stem.soil_material``.
In this case, a point load is applied, therefore ``PointLoad`` is imported from ``stem.load``.
Boundary conditions are set using ``DisplacementConstraint`` and ``AbsorbingBoundary``.
Solver settings are defined with classes imported from ``stem.solver``.
For output, ``NodalOutput``, ``VtkOutputParameters``, and ``JsonOutputParameters`` are imported.
Finally, ``Stem`` is imported from ``stem.stem`` to write input files and run the calculation.

Geometry and material
---------------------
In this step, the geometry and material are defined.
First the model dimension is set to 3 and the model is initialised.

.. code-block:: python

    ndim = 3
    model = Model(ndim)

    # END CODE BLOCK

The soil is modelled as linear elastic, drained, and one-phase.

.. code-block:: python

    DENSITY_SOLID = 2000
    POROSITY = 0
    YOUNG_MODULUS = 30e6
    POISSON_RATIO = 0.2

    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=DENSITY_SOLID, POROSITY=POROSITY)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=YOUNG_MODULUS, POISSON_RATIO=POISSON_RATIO)
    retention_parameters = SaturatedBelowPhreaticLevelLaw()
    material = SoilMaterial("soil", soil_formulation, constitutive_law, retention_parameters)

    # END CODE BLOCK

A rectangular soil domain is created in the x-y plane and extruded in z-direction.

.. code-block:: python

    x_max = 10
    y_max = 5
    z_max = 10

    layer_coordinates = [(0.0, 0.0, 0.0), (x_max, 0.0, 0.0), (x_max, y_max, 0.0), (0.0, y_max, 0.0)]
    model.extrusion_length = z_max

    model.add_soil_layer_by_coordinates(layer_coordinates, material, "soil")

    # END CODE BLOCK

Load
----
A point load is applied at the surface corner (x=0, y=y_max, z=0), acting in the negative y-direction.

.. code-block:: python

    force = -1e6
    node_coordinates = [(0.0, y_max, 0.0)]

    point_load = PointLoad(active=[True, True, True], value=[0, force, 0])
    model.add_load_by_coordinates(node_coordinates, point_load, "point_load")

    # END CODE BLOCK

Boundary conditions
-------------------
Below the boundary conditions are defined.
The base is fully fixed.
Roller boundaries are applied on x=0 and z=0 planes.
Absorbing boundaries are applied on x=x_max and z=z_max planes.

.. code-block:: python

    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])

    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                                 roller_displacement_parameters, "sides_roller")

    # END CODE BLOCK

Mesh and geometry synchronisation
---------------------------------
The mesh size and element order are defined.
After assigning geometry and conditions, the geometry is synchronised.

.. code-block:: python

    model.set_mesh_size(element_size=0.25)
    model.mesh_settings.element_order = 2

    model.synchronise_geometry()

    # END CODE BLOCK

Solver settings
---------------
Now that the model is defined, the solver settings are set.
A dynamic mechanical analysis is used with constant time step.
Linear-Newton-Raphson is used as strategy and Cg as linear solver.

.. code-block:: python

    time_step = 0.001

    time_integration = TimeIntegration(start_time=0.0,
                                       end_time=0.08,
                                       delta_time=time_step,
                                       reduction_factor=1.0,
                                       increase_factor=1.0,
                                       max_delta_time_factor=1000)

    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)

    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.DYNAMIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True,
                                     are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=LinearNewtonRaphsonStrategy(),
                                     linear_solver_settings=Cg(),
                                     rayleigh_k=7.86e-5,
                                     rayleigh_m=0.248)

    # END CODE BLOCK

Problem and output
------------------
The problem definition is added to the model.
In this example, JSON output is requested at four surface points and VTK output
is written for the full computational model part.

.. code-block:: python

    problem = Problem(problem_name="Pekeris", number_of_threads=44, settings=solver_settings)
    model.project_parameters = problem

    json_output_parameters = JsonOutputParameters(time_step, [NodalOutput.DISPLACEMENT], [])

    model.add_output_settings_by_coordinates([
        (0, y_max, 0),
        (1, y_max, 0),
        (2, y_max, 0),
        (3, y_max, 0),
    ], json_output_parameters, "json_output")

    model.add_output_settings(
        output_parameters=VtkOutputParameters(
            file_format="ascii",
            output_interval=1,
            nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY],
            gauss_point_results=[],
            output_control_type="step"
        ),
        part_name="porous_computational_model_part",
        output_dir="output",
        output_name="vtk_output"
    )

    # END CODE BLOCK

Run
---
Now that the model is set up, the calculation is ready to run.

.. code-block:: python

    stem = Stem(model, input_files_dir)

    # END CODE BLOCK

Write inputs
------------
The Kratos input files are written to the input folder.

.. code-block:: python

    stem.write_all_input_files()

    # END CODE BLOCK

Run calculation
---------------
The calculation is run by calling `run_calculation`.

.. code-block:: python

    stem.run_calculation()

    # END CODE BLOCK

.. seealso::

    - Next: :ref:`tutorial2`