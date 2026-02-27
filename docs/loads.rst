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
x, y, and z-directions, respectively.
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
x, y, and z-directions, respectively.
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
x, y, and z-directions, respectively.
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
       direction_signs=[1, 0, 0],
       velocity=50.0,
       origin=[x_start, y_level, z_track],
       offset=0.0,
   )
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], mload, "moving_load")


The ``load`` parameter is a list of three values that specify the magnitude of the load in each direction.
The ``direction_signs`` parameter is a list of three values that specify the direction of the load
(e.g., [1, 0, 0] for x-direction).
The ``velocity`` parameter specifies the speed at which the load moves along the path (in m/s).
The ``origin`` parameter specifies the starting coordinate of the load, and it is required that the origin is located
along the path.
The ``offset`` parameter can be used to specify an initial offset along the path.


UVEC load
---------
In STEM it is possible to model an external load (UVEC) that is defined by a user-defined function.
This is the most flexible way to define loads, as it allows the implementation of point loads that can be expressed
as a function of time and or displacement at its location.
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
                        velocity=40,
                        origin=[0, 0, 0],
                        wheel_configuration=[0.0, 2.5, 19.9, 22.4],
                        uvec_file="uvec.py",  # the file where the UVEC function is defined
                        uvec_function_name="uvec",  # the name of the function that defines the UVEC load
                        uvec_parameters=uvec_parameters)

Similar to the moving load, the UVEC load also needs a path to move along;
make sure the orientation matches the intended travel direction.
The ``wheel_configuration`` parameter is used to specify the offsets of each wheel with respect to the ``origin``.

The UVEC load can be added to coordinates, geometry IDs or model parts:

.. code-block:: python

   # Add UVEC by coordinates:
   model.add_load_by_coordinates([(x1, y1, z1), (x2, y2, z2)], uvec_load, "train_load")
   # Add UVEC by geometry ID:
   model.add_load_by_geometry_ids([1], uvec_load, "uvec_load")
   # Add UVEC to model part:
   model.add_load_on_line_model_part("rail_track_1", uvec_load, "uvec_load")


Default train
.............
The default train in STEM (2D with 10 degrees-of-freedom per cart) is defined as a UVEC load.
In order to use the default train, the user can simply import ``uvec``.
This function defines contains the train definition and the train-track interaction model as described in
:ref:`uvec_formulation`.

To define the UVEC train it is required to define the wheel configuration `wheel_configuration`
and train parameters `uvec_parameters`.
In this example a train with two carts is defined, where each cart has two bogies and each bogie has two wheels.

.. code-block:: python

   from stem.load import UvecLoad
   import UVEC.uvec_ten_dof_vehicle_2D as uvec

   # define uvec parameters
   wheel_configuration=[0.0, 2.5, 19.9, 22.4, 23.5, 26.0, 43.4, 45.9] # wheel configuration [m]
   uvec_parameters = {"n_carts": 2, # number of carts [-]
                      "cart_inertia": (1128.8e3) / 2, # mass moment of inertia of the cart [kgm2]
                      "cart_mass": (50e3) / 2, # mass of the cart [kg]
                      "cart_stiffness": 2708e3, # stiffness between the cart and bogies [N/m]
                      "cart_damping": 64e3, # damping coefficient between the cart and bogies [Ns/m]
                      "bogie_distances": [-9.95, 9.95], # distances of the bogies from the centre of the cart [m]
                      "bogie_inertia": (0.31e3) / 2, # mass moment of inertia of the bogie [kgm2]
                      "bogie_mass": (6e3) / 2, # mass of the bogie [kg]
                      "wheel_distances": [-1.25, 1.25], # distances of the wheels from the centre of the bogie [m]
                      "wheel_mass": 1.5e3, # mass of the wheel [kg]
                      "wheel_stiffness": 4800e3, # stiffness between the wheel and the bogie [N/m]
                      "wheel_damping": 0.25e3, # damping coefficient between the wheel and the bogie [Ns/m]
                      "gravity_axis": 1, # axis on which gravity works [x =0, y = 1, z = 2]
                      "contact_coefficient": 9.1e-7, # Hertzian contact coefficient between the wheel and the rail [N/m]
                      "contact_power": 1.5, # Hertzian contact power between the wheel and the rail [-]
                      "static_initialisation": False, # True if the analysis of the UVEC is static
                      "wheel_configuration": wheel_configuration,
                      "velocity": 40,
                      }

    uvec_load = UvecLoad(direction_signs=[1, 1, 1],
                        velocity=40,
                        origin=wheel_configuration,
                        wheel_configuration=wheel_configuration,
                        uvec_parameters=uvec_parameters,
                        uvec_model=uvec,
                        )

In this case, the ``uvec_model`` parameter is used to specify the UVEC function that defines the train and the
train-track interaction model.


Time-dependent loads with Table
-------------------------------
In STEM, it is possible to define time-dependent loads using the Tables.

A Table is a class that allows you to define a function of time (or any other parameter)
by specifying a list of times and values.

.. code-block:: python

   from stem.table import Table
   from stem.load import LineLoad

   # Define a ramp in time for the y-direction
   ramp = Table(times=[0.0, 1.0, 2.0], values=[0.0, 1.0, 0.0])
   line = LineLoad(active=[False, True, False], value=[0.0, ramp, 0.0])


Practical tips
--------------
- Units: SI (N, N/m, Pa).
- Moving loads and UVEC need a path/line to move along; make sure the orientation matches the intended travel direction.
- For surface loads, ensure the node ordering is consistent (clockwise or anti-clockwise) to avoid incorrect load directions.
