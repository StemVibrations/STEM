Loads
=====

STEM supports a range of load types via :mod:`stem.load`. This page shows quick examples
and tips to choose the right parameters and entities.

Basics
------
- Use ``active=[bool, bool, bool]`` to enable directions (x, y, z).
- The ``value`` can be a float (time-independent) or a :class:`stem.table.Table` (time-dependent).
- Apply loads to coordinates or geometry groups depending on the load type.
- Units: SI (N, N/m, Pa).

Point load
----------
.. code-block:: python

   from stem.load import PointLoad

   p = PointLoad(active=[False, True, False], value=[0.0, -1e4, 0.0])  # N
   model.add_load_by_coordinates([(x0, y0, z0)], p, "point_load")

Line load
---------
.. code-block:: python

   from stem.load import LineLoad

   line = LineLoad(active=[False, True, False], value=[0.0, -1e3, 0.0])  # N/m
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], line, "line_load")

Surface load (pressure)
-----------------------
.. code-block:: python

   from stem.load import SurfaceLoad

   surf = SurfaceLoad(active=[False, True, False], value=[0.0, -50e3, 0.0])  # Pa
   # Apply to known surface group by name or ids (depending on how the geometry was created)
   model.add_load_by_geometry_ids(2, [surface_id], surf, "surface_load")

Moving load
-----------
.. code-block:: python

   from stem.load import MovingLoad

   mload = MovingLoad(
       load=[0.0, -1.2e5, 0.0],   # N per wheel or axle representation
       direction=[1, 0, 0],       # +x travel
       velocity=50.0,             # m/s
       origin=[x_start, y_level, z_track],
       offset=0.0,
   )
   # Apply along a line (2 nodes define the track path); use model dimension accordingly
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], mload, "moving_load")

Gravity
-------
.. code-block:: python

   from stem.load import GravityLoad

   gravity = GravityLoad()  # uses global gravity; ensure vertical axis matches project convention
   model.add_gravity(gravity)

UVEC vehicle coupling
---------------------
- Use :class:`stem.load.UvecLoad` to drive wheel loads from a user-defined vehicle model.
- See :doc:`API_definition` for the UVEC function signature and data exchange.

Time-dependent loads with Table
-------------------------------
.. code-block:: python

   from stem.table import Table
   from stem.load import LineLoad

   # Define a ramp in time for the y-direction
   ramp = Table(x_values=[0.0, 1.0, 2.0], y_values=[0.0, 1.0, 0.0])  # time [s], amplitude [-]
   line = LineLoad(active=[False, True, False], value=[0.0, ramp, 0.0])

Tips
----
- Pick the correct units: N for point; N/m for line; Pa for surface.
- For quadratic line/surface elements, STEM maps to Kratos "DiffOrder" conditions where needed.
- Moving loads and UVEC need a path/line to move along; make sure the orientation matches the intended travel direction.
- For coupled analyses (mechanical + groundwater), ensure AnalysisType is set accordingly so the right Kratos conditions are used.
