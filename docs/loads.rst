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


Default trains
..............
The default train in STEM (2D with 10 degrees-of-freedom per cart) is defined as a UVEC load.
In order to use the default train, the user can simply import ``uvec``.
This function defines contains the train definition and the train-track interaction model as described in
:ref:`uvec_formulation`.

To define the UVEC train it is required to define the wheel configuration `wheel_configuration`
and train parameters `uvec_parameters`.
In this example a train with two carts is defined, where each cart has two bogies and each bogie has two wheels.

.. code-block:: python

   from stem.load import UvecLoad, TrainType
   import UVEC.uvec_ten_dof_vehicle_2D as uvec

   # define uvec parameters
   wheel_configuration=[0.0, 2.5, 19.9, 22.4, 23.5, 26.0, 43.4, 45.9] # wheel configuration [m]
   uvec_parameters = {
                      "cart_mass": (50e3) / 2, # mass of half the cart [kg]
                      "bogie_mass": (6e3) / 2, # mass of half one bogie / primary suspension mass [kg]
                      "wheel_mass": 1.5e3, # mass of one wheel / secondary suspension mass [kg]
                      "cart_inertia": (1128.8e3) / 2, # mass moment of inertia of half the cart [kgm2]
                      "bogie_inertia": (0.31e3) / 2, # mass moment of inertia of half one bogie / primary suspension inertia [kgm2]
                      "cart_stiffness": 2708e3, # stiffness between the cart and bogies for 1 spring / primary suspension stiffness [N/m]
                      "wheel_stiffness": 4800e3, # stiffness between the wheel and the bogie for 1 spring / secondary suspension stiffness [N/m]
                      "cart_damping": 64e3, # damping coefficient between the cart and bogies for 1 damper / primary suspension damping [Ns/m]
                      "wheel_damping": 0.25e3, # damping coefficient between the wheel and the bogie for 1 damper / secondary suspension damping [Ns/m]
                      "bogie_distances": [-9.95, 9.95], # distances of the bogies from the centre of the cart [m]
                      "wheel_distances": [-1.25, 1.25], # distances of the wheels from the centre of the bogie [m]
                      "cart_length": 22.4,  # length of the cart [m]
                      "gravity_axis": 1, # axis on which gravity works [x =0, y = 1, z = 2]
                      "contact_coefficient": 9.1e-7, # Hertzian contact coefficient between the wheel and the rail [N/m]
                      "contact_power": 1.5, # Hertzian contact power between the wheel and the rail [-]
                      "wheel_configuration": wheel_configuration,
                      }

    uvec_load = UvecLoad(direction_signs=[1, 1, 1],
                         velocity=40,
                         origin=wheel_configuration,
                         uvec_model=uvec,
                         nb_carts=2,
                         offset=0,
                         train_type=TrainType.CUSTOM,
                         uvec_parameters=uvec_parameters,
                         static_vehicle_calculation=False,
                         irregularities=None,
                         rail_joint=None,
                        )

In this case, the ``uvec_model`` parameter is used to specify the UVEC function that defines the train and the
train-track interaction model.
The ``train_type`` parameter is set to ``TrainType.CUSTOM`` to indicate that the train is defined by the user and not
by the default parameters. When using custom train types, it is required to define the ``uvec_parameters``
to specify the parameters of the train and the train-track interaction model.
The use of irregularities and rail joints is optional and can be defined by the user as needed
(see :ref:`irregularities_track` and :ref:`rail_joints` for more details on how to define irregularities
and rail joints).

Alternatively, the user can also use the default train without specifying the ``uvec_parameters``.
In this case, the default parameters for the train and the train-track interaction model will be used.
To use the default train, simply specify the train type as follows:

.. code-block:: python

   from stem.load import UvecLoad, TrainType
   import UVEC.uvec_ten_dof_vehicle_2D as uvec

    uvec_load = UvecLoad(direction_signs=[1, 1, 1],
                        velocity=40,
                        origin=wheel_configuration,
                        uvec_parameters=uvec_parameters,
                        uvec_model=uvec,
                        train_type=TrainType.PASSENGER_HEAVY,
                        irregularities=None,
                        rail_joint=None,
                        static_vehicle_calculation=False,
                        )


Table :ref:`default_train_parameters` shows the default parameters for the different train types that are
available in STEM. The values of the parameters are based on :cite:`Ricardo_2025`
and can be used as a reference for defining custom trains. These values concern a half model train, where the
values for the cart correspond to half the mass and inertia of the cart, the values of the secondary suspension
correspond to one spring-damper system, the values for the bogie correspond to half the mass and inertia of one bogie,
the values for the primary suspension correspond to one spring-damper system, and the values for
the wheel correspond to the mass and inertia of one wheel.

The values presented in the table are for reference only and can be adjusted by the user as needed to define custom
trains.

.. _default_train_parameters:

.. table:: Default train parameters

   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | Parameter                              | Locomotive | Passenger train (heavy)   | Passenger train (light)   | Freight train (loaded)   | Freight train (unloaded) |
   +========================================+============+===========================+===========================+==========================+==========================+
   | cart mass [kg]                         | 27500      | 19000                     | 14500                     | 40500                    |     6500                 |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | bogie mass [kg]                        | 3000       | 1550                      | 1300                      | 900                      |     900                  |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | wheel mass [kg]                        | 2250       | 950                       | 850                       | 700                      |     700                  |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | cart inertia [kgm²]                    | 485000     | 1150000                   | 900000                    | 365000                   |     115000               |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | bogie inertia [kgm²]                   | 3900       | 1200                      | 1150                      | 850                      |     850                  |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | cart stiffness [N/m]                   | 8.0e5      | 3.5e5                     | 4.0e5                     | 6.0e6                    |     6.0e6                |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | cart damping [Ns/m]                    | 4.5e4      | 3.2e4                     | 2.2e4                     | 6.5e4                    |     6.5e4                |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | wheel stiffness [N/m]                  | 2.0e6      | 1.2e6                     | 7.4e5                     | 2.6e6                    |     5.0e5                |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | wheel damping [Ns/m]                   | 3.3e4      | 2.5e3                     | 6.4e3                     | 1.5e4                    |     5.7e3                |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | bogie distance to center               | 10.4       | 10.0                      | 9.5                       | 7.0                      |     7.0                  |
   | of cart [m]                            |            |                           |                           |                          |                          |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | wheel distance to center               | 1.3        | 1.25                      | 1.28                      | 0.9                      |     0.9                  |
   | of bogie [m]                           |            |                           |                           |                          |                          |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+
   | cart length [m]                        | 26.8       | 27.0                      | 25.0                      | 17.0                     |     17.0                 |
   +----------------------------------------+------------+---------------------------+---------------------------+--------------------------+--------------------------+

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
