Boundary conditions
===================
This page shows how to define boundary conditions in STEM and provides a few tips
for assigning them to the correct geometric entities.

Types of boundary conditions
----------------------------
STEM supports the following types of boundary conditions:
- Displacement constraints (fixed, roller, etc.)
- Absorbing boundaries (Lysmer-type :cite:`Lysmer_Kuhlemeyer_1969`)

In STEM the displacement boundary conditions are defined as:

.. code-block:: python

   from stem.boundary import DisplacementConstraint

   # Define a fixed boundary condition (zero displacement in all directions)
   fixed = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])

   # Define a roller boundary condition (zero displacement in x and z, free in y)
   roller = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])

The 'is_fixed' array indicates which directions have fixed displacement (True means fixed, False means free).
The value is used to specify the displacement value for fixed directions (0 means zero displacement).
For roller conditions, the free direction can have a value of 0 or any other value since it is not constrained.

The absorbing boundaries are defined as:

.. code-block:: python

   from stem.boundary import AbsorbingBoundary

   absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0],
                                                       virtual_thickness=10.0)


The `absorbing_factors` are the Lysmer scaling factors for the normal and tangential directions, respectively.
A value of 1.0 corresponds to impedance-matched (optimal) absorption for normally incident waves,
while a value of 0.0 disables absorption.

The `virtual_thickness` defines a virtual boundary layer used to compute elastic stiffness terms that stabilize
the model against rigid body motion.

Application of boundary conditions
----------------------------------
In STEM boundary conditions are applied to geometric entities (points, lines, surfaces) defined in the geometry and mesh.
Boundary conditions can be added to the model by specifying a plane dimension (only valid for 3D models),
or by specifying a list of geometry IDs (valid for 2D and 3D).

To assign boundary conditions on a plane, specify three points that define the plane, assign the
boundary condition (as shown above), and give it a name:

.. code-block:: python
   # Define the plane by three points
   model.add_boundary_condition_on_plane([(0, 0, 0), (x_max, 0, 0), (x_max, 0, z_max)], fixed, "base_fixed")


To assign boundary conditions by geometry IDs, specify the dimension of the boundary condition (1 for lines in 2D,
2 for surfaces in 3D), a list of geometry IDs, the boundary condition (as shown above), and a name:

.. code-block:: python
   # Apply to surface IDs (dimension=2 for surfaces in 3D)
   model.add_boundary_condition_by_geometry_ids(2, [1], fixed, "base_fixed")
   model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17], roller, "sides_roller")

Inspect geometry IDs
--------------------
When assigning boundary conditions by geometry IDs, it is crucial to ensure that the correct IDs are used.
After creating the geometry, use the following helper functions to visualize the geometry and inspect the IDs:

.. code-block:: python

   model.synchronise_geometry()
   model.show_geometry(show_surface_ids=True)  # for 3D models (surfaces)
   # model.show_geometry(show_line_ids=True)   # for 2D models (lines)
   # model.show_geometry(show_point_ids=True)  # to see points

In this way it is possible to identify the correct geometry IDs for the surfaces or lines
where the boundary conditions should be applied.

Practical tips
--------------
- Boundary condition dimensions: 2 for surfaces in 3D, 1 for lines in 2D.
- Keep boundary condition names unique.
- When assigning boundary conditions, by geometry IDs, visualise the geometry to avoid mis-assignments.
- Place absorbing boundaries sufficiently far from sources to avoid spurious reflections.
- Use Lysmer absorbing boundaries conditions to mitigate spurious reflections.
