Loads
=====
This page shows how to define loads in STEM and provides a few tips
for assigning them to the correct geometric entities.


Point load
----------
In STEM, point loads are defined as concentrated forces applied at specific points in the model.
To define a point load:

.. code-block:: python

   from stem.load import PointLoad

   p = PointLoad(active=[False, True, False], value=[0.0, -1e4, 0.0])
   model.add_load_by_coordinates([(x, y, z)], p, "point_load")

The ``active`` parameter is a list of three booleans that indicate whether the load is active in the
x, y, and z directions, respectively.
The ``value`` parameter is a list of three values that specify the magnitude of the load in each direction.
The load then is applied to to model, by specifying a list of the node coordinates where the load should be applied,
the load object, and the load name.


Line load
---------
In STEM, line loads are defined as distributed forces applied along a line in the model.
To define a line load:

.. code-block:: python

   from stem.load import LineLoad

   line = LineLoad(active=[False, True, False], value=[0.0, -1e3, 0.0])
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], line, "line_load")

The ``active`` parameter is a list of three booleans that indicate whether the load is active in the
x, y, and z directions, respectively.
The ``value`` parameter is a list of three values that specify the magnitude of the load in each direction.
The load then is applied to to model, by specifying a list of the node coordinates where the load should be applied,
the load object, and the load name.


Surface load
------------
In STEM, surface loads are defined as distributed forces applied over a surface in the model.
To define a surface load:

.. code-block:: python

   from stem.load import SurfaceLoad

   surf = SurfaceLoad(active=[False, True, False], value=[0.0, -50e3, 0.0])
   # Apply to surface by geometry ID (e.g., from Gmsh physical group):
   model.add_load_by_geometry_ids(2, [surface_id], surf, "surface_load")
   # Or, if you have the coordinates of the surface nodes:
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)], surf, "surface_load")


The ``active`` parameter is a list of three booleans that indicate whether the load is active in the
x, y, and z directions, respectively.
The ``value`` parameter is a list of three values that specify the magnitude of the load in each direction.

The load can be applied on a surface by specifying the geometry IDs, or
by specifying the coordinates of the surface nodes.
When applying by geometry ID, make sure to use the correct geometry IDs (see :ref:`geometry_id_bc`).
When applying by coordinates, make sure to specify the nodes in either clockwise or anti-clockwise order.

Moving load
-----------
In STEM, moving loads are defined as Point load forces that move along a specified path in the model.
To define a moving load:

.. code-block:: python

   from stem.load import MovingLoad

   mload = MovingLoad(
       load=[0.0, -1.2e5, 0.0],
       direction=[1, 0, 0],
       velocity=50.0,
       origin=[x_start, y_level, z_track],
       offset=0.0,
   )
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], mload, "moving_load")


The ``load`` parameter is a list of three values that specify the magnitude of the load in each direction.
The ``direction`` parameter is a list of three values that specify the direction of the load
(e.g., [1, 0, 0] for x-direction).
The ``velocity`` parameter specifies the speed at which the load moves along the path (in m/s).
The ``origin`` parameter specifies the starting point of the load,
and the ``offset`` parameter can be used to specify an initial offset along the path.


UVEC load
---------
In STEM it is possible to model an external load (UVEC) that is defined by a user-defined function.
This is the most flexible way to define loads, as it allows you to implement any kind of load that can be expressed
as a function of time and/or other parameters.
This is  the method that it is used to simulate the train and the train-track interaction in STEM.
More details on how to define a UVEC load can be found in the :doc:`API_definition` page.

UVEC load are defined as:

.. code-block:: python

   from stem.load import UvecLoad

   # Define UVEC load
   uvec_parameters = {
        ... # user specific parameters for the UVEC function
   }

   uvec_load = UvecLoad(direction_signs=[1, 1, 1],
                        velocity=velocity,
                        origin=[0.0, 0, 0],
                        wheel_configuration=[0.0],
                        uvec_file="uvec.py",
                        uvec_function_name="uvec",
                        uvec_parameters=uvec_parameters)
   model.add_load_by_geometry_ids([1], uvec_load, "uvec_load")

   # UVEC can be added to coordinates; geometry IDs or model parts.
   # For example, to add it to coordinates:
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], uvec_load, "train_load")
   # Or to a geometry ID:
   model.add_load_by_geometry_ids([1], uvec_load, "uvec_load")
   # Or to a model part:
   model.add_load_on_line_model_part("rail_track_1", uvec_load, "uvec_load")


Time-dependent loads with Table
-------------------------------
In STEM, it is possible to define time-dependent loads using the Tables.

A Table is a class that allows you to define a function of time (or any other parameter)
by specifying a list of x and y values.

.. code-block:: python

   from stem.table import Table
   from stem.load import LineLoad

   # Define a ramp in time for the y-direction
   ramp = Table(x_values=[0.0, 1.0, 2.0], y_values=[0.0, 1.0, 0.0])  # time [s], amplitude [-]
   line = LineLoad(active=[False, True, False], value=[0.0, ramp, 0.0])


Practical tips
--------------
- Units: SI (N, N/m, Pa).
- Moving loads and UVEC need a path/line to move along; make sure the orientation matches the intended travel direction.
- For surface loads, ensure the node ordering is consistent (clockwise or anti-clockwise) to avoid incorrect load directions.
