Boundary conditions
===================

This page shows how to define common boundary conditions in STEM and provides a few tips
for assigning them to the correct geometric entities.

Inspect geometry IDs
--------------------
Boundary conditions are applied to geometry IDs produced by gmsh. Use these helpers
after creating the geometry to see IDs:

.. code-block:: python

   model.synchronise_geometry()
   model.show_geometry(show_surface_ids=True)  # for 3D models (surfaces)
   # model.show_geometry(show_line_ids=True)   # for 2D models (lines)
   # model.show_geometry(show_point_ids=True)  # to see points

Fixed base and roller sides (3D example)
----------------------------------------
- Fixed base: zero displacement in x, y, z on bottom surfaces.
- Roller sides: zero displacement in x and z, free in y on side surfaces.

.. code-block:: python

   from stem.boundary import DisplacementConstraint

   # Define constraint presets
   fixed = DisplacementConstraint(active=[True, True, True], is_fixed=[True, True, True], value=[0, 0, 0])
   roller = DisplacementConstraint(active=[True, True, True], is_fixed=[True, False, True], value=[0, 0, 0])

   # Apply to surface IDs (dimension=2 for surfaces in 3D)
   model.add_boundary_condition_by_geometry_ids(2, [1], fixed, "base_fixed")
   model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17], roller, "sides_roller")

2D models
---------
In 2D, boundary conditions are assigned to lines (dimension=1). Example:

.. code-block:: python

   # Bottom line fixed; lateral lines roller-like
   model.add_boundary_condition_by_geometry_ids(1, [bottom_line_id], fixed, "base_fixed_2d")
   model.add_boundary_condition_by_geometry_ids(1, [left_line_id, right_line_id], roller, "sides_roller_2d")

Absorbing boundaries (tips)
---------------------------
- Use Lysmer-type boundaries or equivalent absorbing conditions to mitigate spurious reflections.
- These are configured through Kratos processes; STEM exposes helpers in :mod:`stem.additional_processes`.
- Place absorbing boundaries sufficiently far from sources or refine the mesh near boundaries to improve performance.

General tips
------------
- Verify the entity dimension: 2 for surfaces in 3D, 1 for lines in 2D.
- Keep boundary condition names unique and descriptive (used downstream in IO).
- Visualize geometry IDs early to avoid mis-assignments.
- Avoid over-constraining the model; prefer rollers where applicable.
