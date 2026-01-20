Formulation
===========

This page summarizes the mathematical and numerical methods used by STEM to simulate
railway-induced ground vibrations. STEM constructs and executes simulations using
`Kratos Multiphysics <https://github.com/KratosMultiphysics/Kratos>`_, primarily the
GeoMechanics and Railway applications, while the user-facing Python API lives in this
package.

Governing equations
-------------------
STEM solves the (small-strain) linear momentum balance in the solid skeleton and, when
requested, the groundwater flow equation for a porous medium (in coupled analyses):

- Mechanical (quasi-static/dynamic):

  .. math:: \n
     \n     \mathbf{M} \, \ddot{\mathbf{u}} + \mathbf{C} \, \dot{\mathbf{u}} + \mathbf{K} \, \mathbf{u} = \mathbf{f}(t)

  where :math:`\mathbf{u}` is the nodal displacement vector, :math:`\mathbf{M}` the mass matrix,
  :math:`\mathbf{C}` the damping matrix, :math:`\mathbf{K}` the stiffness matrix, and :math:`\mathbf{f}` the load vector.

- Groundwater flow (optional): Richards/Darcy-type flow within the GeoMechanicsApplication; STEM exposes
  the configuration through its high-level API when using coupled analyses (:class:`stem.solver.AnalysisType`).

Spatial discretisation (FEM)
----------------------------
- Geometry and mesh: STEM builds geometry using gmsh (via ``gmsh_utils``) and assigns physical groups
  to volumes/surfaces/lines for materials, boundaries and loads. The mesh is exported to Kratos mdpa
  via :mod:`stem.IO.kratos_io`.
- Elements and conditions are provided by Kratos; STEM maps gmsh element types to Kratos types automatically.
- Materials can be linear elastic or user-defined. STEM supports:
  - Built-in laws for soil and structural materials, e.g. :mod:`stem.soil_material` and :mod:`stem.structural_material`.
  - External constitutive models through UMAT/UDSM interfaces (see :doc:`API_definition`).

Time integration and solution strategies
----------------------------------------
- Solution type: :class:`stem.solver.SolutionType` supports quasi-static and dynamic analyses.
- Time integration:
  - Dynamic analyses default to the :class:`stem.solver.NewmarkScheme` with parameters (:pyattr:`beta`, :pyattr:`gamma`),
    which is unconditionally stable for standard choices (e.g. 0.25/0.5 for linear problems).
  - Quasi-static analyses internally use a static scheme (:class:`stem.solver.StaticScheme`).
- Nonlinear solution: Newton-type strategies in :mod:`stem.solver`:
  - :class:`stem.solver.NewtonRaphsonStrategy` (default),
  - :class:`stem.solver.LineSearchStrategy`,
  - :class:`stem.solver.ArcLengthStrategy`.
- Linear solvers: configurable via :class:`stem.solver.LinearSolverSettingsABC`, including
  :class:`stem.solver.Amgcl`, :class:`stem.solver.Cg`, :class:`stem.solver.Lu`, :class:`stem.solver.SparseCg`.
- Damping: Rayleigh damping is supported in dynamics via ``rayleigh_m`` and ``rayleigh_k``
  in :class:`stem.solver.SolverSettings`.

Boundary conditions and processes
---------------------------------
- Displacement constraints (fixed and rollers) are defined via :mod:`stem.boundary` and mapped to Kratos conditions.
- Absorbing boundaries (Lysmer-type) and other processes are handled through Kratos processes; STEM exposes
  these through :mod:`stem.additional_processes` and :mod:`stem.water_processes` when applicable.

Loads and train–track interaction
---------------------------------
- Standard loads include line/point/surface loads defined in :mod:`stem.load`.
- Vehicle–track interaction can be provided by a user-defined vehicle model (UVEC)
  as described in :doc:`API_definition`. At each time step, STEM exchanges kinematic
  state near the wheel–rail contact and expects wheel loads back from the UVEC function.

Problem and solver settings
---------------------------
- The overall configuration is collected in :class:`stem.solver.SolverSettings` and
  :class:`stem.solver.Problem`.
- Output configuration supports VTK, GiD, and JSON via :mod:`stem.output` and the corresponding
  writers in :mod:`stem.IO.kratos_output_io`.

References
----------
For background on soil/structural dynamics and train–track interaction:

- :cite:`Verruijt_2010` — Soil dynamics fundamentals.
- :cite:`Biggs_1964` — Structural dynamics.
- :cite:`Zhang_2001`, :cite:`Lei_Noda_2002`, :cite:`Kabo_2006` — Examples of vehicle–track interaction modelling.

See :doc:`bibliography` for the complete list.
