Quickstart
==========

This short guide gets you from installation to your first simulation input files.
For complete walkthroughs, see the Tutorials section.

Install
-------

- User install (latest release):

  .. code-block::

     pip install STEM-Vibrations

- Developer install (editable with tests):

  .. code-block::

     git clone https://github.com/StemVibrations/STEM.git
     cd STEM
     pip install -e '.[testing]'

Minimal example
---------------

The script below creates a simple 3D embankment with two soil layers, applies fixed and roller
boundary conditions, adds a line load, writes Kratos input files, and (optionally) runs Kratos
if available. Save as ``quickstart.py`` and run inside your Python environment.

.. code-block:: python

   from stem.model import Model
   from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
   from stem.load import LineLoad
   from stem.boundary import DisplacementConstraint
   from stem.solver import (
       AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
       LinearNewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType,
       SolverSettings, Problem
   )
   from stem.output import NodalOutput, VtkOutputParameters
   from stem.stem import Stem

   # 1) Geometry and materials
   ndim = 3
   model = Model(ndim)

   # Soil layers
   soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
   mat_1 = SoilMaterial(
       "soil_1",
       soil_formulation_1,
       LinearElasticSoil(YOUNG_MODULUS=30e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

   soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2550, POROSITY=0.3)
   mat_2 = SoilMaterial(
       "soil_2",
       soil_formulation_2,
       LinearElasticSoil(YOUNG_MODULUS=30e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

   emb_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
   mat_emb = SoilMaterial(
       "embankment",
       emb_formulation,
       LinearElasticSoil(YOUNG_MODULUS=10e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

   # 2) Coordinates (extruded in z)
   soil1 = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
   soil2 = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
   emb   = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
   model.extrusion_length = 20.0

   model.add_soil_layer_by_coordinates(soil1, mat_1, "soil_layer_1")
   model.add_soil_layer_by_coordinates(soil2, mat_2, "soil_layer_2")
   model.add_soil_layer_by_coordinates(emb,   mat_emb, "embankment_layer")

   # 3) Load and mesh
   load_coords = [(0.75, 3.0, 0.0), (0.75, 3.0, 20.0)]
   line_load = LineLoad(active=[False, True, False], value=[0, -1000, 0])
   model.add_load_by_coordinates(load_coords, line_load, "line_load")

   model.set_mesh_size(element_size=2.0)

   # 4) Boundary conditions (use model.show_geometry(show_surface_ids=True) to inspect IDs)
   fixed = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True], value=[0, 0, 0])
   roller = DisplacementConstraint(active=[True, True, True], is_fixed=[True, False, True], value=[0, 0, 0])

   model.synchronise_geometry()
   model.add_boundary_condition_by_geometry_ids(2, [1], fixed, "base_fixed")
   model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17], roller, "sides_roller")

   # 5) Solver and problem
   time = TimeIntegration(start_time=0.0, end_time=0.05, delta_time=0.01, reduction_factor=1.0, increase_factor=1.0)
   conv = DisplacementConvergenceCriteria(1.0e-4, 1.0e-9)
   solver = SolverSettings(
       analysis_type=AnalysisType.MECHANICAL,
       solution_type=SolutionType.DYNAMIC,
       stress_initialisation_type=StressInitialisationType.NONE,
       time_integration=time,
       is_stiffness_matrix_constant=True,
       are_mass_and_damping_constant=True,
       convergence_criteria=conv,
       strategy_type=LinearNewtonRaphsonStrategy(),
       scheme=NewmarkScheme(),
       linear_solver_settings=Amgcl(),
       rayleigh_k=2.0e-4,
       rayleigh_m=0.6,
   )
   problem = Problem(problem_name="quickstart_example", number_of_threads=1, settings=solver)
   model.project_parameters = problem

   # 6) Output
   nodal = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
   vtk = VtkOutputParameters(output_interval=1, nodal_results=nodal, gauss_point_results=[], output_control_type="step")
   model.add_output_settings(
       part_name="porous_computational_model_part",
       output_name="vtk_output",
       output_dir="output",
       output_parameters=vtk,
   )

   # 7) Generate inputs and (optionally) run
   stem = Stem(model, input_files_dir="quickstart_inputs")
   stem.write_all_input_files()
   # stem.run_calculation()  # requires Kratos Multiphysics with Railway + GeoMechanics apps

Tips
----
- Units: SI (N, m, kg, s, Pa).
- Use the geometry inspection helpers to discover surface/line/point IDs before assigning BCs.
- For quasi-static problems, set ``solution_type=SolutionType.QUASI_STATIC``; Rayleigh parameters are not required.
- Start with a coarse mesh and short end time to verify the setup before refining.
