.. _tutorial5:

Variation in the Z-direction
============================
Overview
--------
This tutorial shows a step-by-step guide on how to set up a 3D geometry where the soil layers vary in both Y and Z
directions. Furthermore, a track is defined which lays on top of the soil and extends outside of the soil domain. The
problem will be calculated in a multistage analysis.

Imports and setup
-----------------
First the necessary packages are imported and paths are defined.

.. code-block:: python

    input_files_dir = "variation_z"
    results_dir = "output"

    from stem.model import Model
    from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
    from stem.structural_material import ElasticSpringDamper, NodalConcentrated
    from stem.default_materials import DefaultMaterial
    from stem.load import MovingLoad
    from stem.boundary import DisplacementConstraint, AbsorbingBoundary
    from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
        LinearNewtonRaphsonStrategy, StressInitialisationType, SolverSettings, Problem, Cg
    from stem.output import NodalOutput, VtkOutputParameters
    from stem.stem import Stem

    # END CODE BLOCK

Geometry, track and materials
----------------------------
For setting up the model, Model class is imported from stem.model. And for setting up the soil material, OnePhaseSoil,
LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw classes are imported.
For the structural elements, the ElasticSpringDamper and NodalConcentrated classes are imported. The rail material will
be retrieved from the DefaultMaterial class. In this case, there is a moving load on top a track. MovingLoad class is
imported from stem.load. As for setting the boundary conditions, DisplacementConstraint class and the AbsorbingBoundary
are imported from stem.boundary. For setting up the solver settings, necessary classes are imported from stem.solver.
Classes needed for the output, are NodalOutput, VtkOutputParameters and Output which are imported from stem.output.
Lastly, Stem class is imported from stem.stem, in order to run the simulation.

First the dimension of the model is indicated which in this case is 3. After which the model can be initialised.

.. code-block:: python

    ndim = 3
    model = Model(ndim)

    # END CODE BLOCK

In this tutorial, different soil layers will be added in the vertical direction and in the extruded out of plane
direction. In order to extrude different parts of the geometry differently, it is required to divide the model in groups.
Below, two groups with unique names are created. Later on, soil layers will be added to these groups. For the creation
of the groups, the 'reference_depth' (reference out of plane coordinate) and the extrusion length have to be given. In
this case, "group_1" has a reference z-coordinate of 0.0 and is extruded for 5 meter along the z-axis; "group_2" starts
at a reference z-coordinate of 5.0 and is extruded for 3 meter. In total, the soil domain will be extruded for 8 meter.

.. code-block:: python

    model.add_group_for_extrusion("group_1", reference_depth=0.0, extrusion_length=5.0)
    model.add_group_for_extrusion("group_2", reference_depth=5.0, extrusion_length=3.0)

    # END CODE BLOCK

Specification of the soil material is defined afterwards.
The bottom soil layer is defined as a material with the name "soil_1".
It's a Linear elastic material model with the solid density (rho) of 2650 kg/m3,
the Young's modulus is 30e6 Pa and the Poisson's ratio is 0.2.
The soil is dry above the phreatic level and wet below the phreatic level. A porosity of 0.3 is specified.
The soil is a one-phase soil, meaning that the flow of water through the soil is not computed.

.. code-block:: python

    solid_density_1 = 2650
    porosity_1 = 0.3
    young_modulus_1 = 30e6
    poisson_ratio_1 = 0.2
    soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity_1)
    constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio_1)
    retention_parameters_1 = SaturatedBelowPhreaticLevelLaw()
    material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, retention_parameters_1)

    # END CODE BLOCK

The second soil layer is defined as a material with the name "soil_2".
It's a Linear elastic material model with the solid density (rho) of 2550 kg/m3,
the Young's modulus is 30e6 Pa and the Poisson's ratio is 0.2.
The soil is dry above the phreatic level and wet below the phreatic level. A porosity of 0.3 is specified.
The soil is a one-phase soil, meaning that the flow of water through the soil is not computed.

.. code-block:: python

    solid_density_2 = 2550
    porosity_2 = 0.3
    young_modulus_2 = 30e6
    poisson_ratio_2 = 0.2
    soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_2, POROSITY=porosity_2)
    constitutive_law_2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_2, POISSON_RATIO=poisson_ratio_2)
    retention_parameters_2 = SaturatedBelowPhreaticLevelLaw()
    material_soil_2 = SoilMaterial("soil_2", soil_formulation_2, constitutive_law_2, retention_parameters_2)

    # END CODE BLOCK

The coordinates of the model are defined in the following way. Each of the layers are defined by a list of coordinates,
defined on an x-y plane. For 3D models, x-y planes are extruded in the z-direction. Since in this case, two groups are
created, the soil layers are added to "group_1" and "group_2". It is important that all soil layers have a unique name.

.. code-block:: python

    soil_bottom_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    soil_top_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]

    second_section_bottom_coordinates = [(0.0, 1.5, 5.0), (5.0, 1.5, 5.0), (5.0, 2.0, 5.0), (0.0, 2.0, 5.0)]
    second_section_top_coordinates=  [(0.0, 0.0, 5.0), (5.0, 0.0, 5.0), (5.0, 1.5, 5.0), (0.0, 1.5, 5.0)]

    model.add_soil_layer_by_coordinates(soil_bottom_coordinates, material_soil_1, "soil_layer_1", "group_1")
    model.add_soil_layer_by_coordinates(soil_top_coordinates, material_soil_2, "soil_layer_2", "group_1")

    model.add_soil_layer_by_coordinates(second_section_bottom_coordinates, material_soil_1, "soil_layer_1_group_2", "group_2")
    model.add_soil_layer_by_coordinates(second_section_top_coordinates, material_soil_2, "soil_layer_2_group_2", "group_2")

    # END CODE BLOCK

The geometry is shown in the figures below.

.. image:: _static/double_extrusion.png

Now that the soil layers are defined, the track will be defined. The track consists of a rail, railpads and sleepers.
Furthermore, the track is extended outside of the 3D soil domain. On the extended part of the track, the track is supported
by 1D elastic spring damper elements which simulate the soil behaviour. The rail parameters are retrieved from the
DefaultMaterial class, where default properties are given to a beam element. The railpad parameters are defined as an
ElasticSpringDamper with a nodal displacement stiffness of 750e6 N/m, a nodal damping coefficient of 750e3 Ns/m.
The sleeper parameters are defined as a NodalConcentrated with a nodal mass of 140 kg. The soil equivalent parameters
are defined as an ElasticSpringDamper with a nodal displacement stiffness of 8e6 N/m and a nodal damping coefficient
of 1 Ns/m.

.. code-block:: python

    rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters

    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                                              NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                              NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],
                                              NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    soil_equivalent_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[0, 8e6, 0],
                                                     NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                                                     NODAL_DAMPING_COEFFICIENT=[0, 1, 0],
                                                     NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0])

    # END CODE BLOCK

Now that the track materials are defined, the track can be added to the model. The track has equal distance between the
sleepers of 0.5 meters. The number of sleepers is calculated based on the distance between the sleepers and the total
length of the track. The rail pad thickness is set to 0.025 meters. The track has an origin point at coordinates [0.75, 2.0, -5.0].
From this point, the track follows the direction of the 'direction_vector' [0, 0, 1] (following the z-axis). The extension
is supported with 1D soil equivalent elements with a length of 2 meters.

.. code-block:: python

    model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
    model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
    model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")

    # END CODE BLOCK


Generating the train track
--------------------------
STEM provides two options to generate a straight track (see :doc:`Tutorial 3 </tutorial3>`).
In this tutorial the track is generated on top of the embankment layer.

The track is added by specifying the origin point of the track and the direction for the extrusion that creates
the rail as well as rail pads and sleepers. Important is that the origin point and the end of the track lie on
geometry edges.

In this tutorial, a straight track is generated parallel to the z-axis at 0.75 m distance from the x-axis,
on top of the embankment. To do this, the origin point of the track is set with coordinates [0.75, 3.0, 0.0] and the
extrusion is done parallel to the positive z-axis, i.e. with a direction vector of [0, 0, 1].
The length of the track is defined by the number of sleepers and their spacing.
In this tutorial, 101 sleepers are placed which are connected by to the rail by 0.025m thick railpads. The sleepers
are spaced 0.5m from each others which results in a 50m straight track, with part name "rail_track".

.. code-block:: python

    origin_point = [0.75, 3.0, 0.0]
    direction_vector = [0, 0, 1]
    number_of_sleepers = 101
    sleeper_spacing = 0.5
    rail_pad_thickness = 0.025

    model.generate_straight_track(sleeper_spacing, number_of_sleepers, rail_parameters,
                                  sleeper_parameters, rail_pad_parameters,
                                  rail_pad_thickness, origin_point,
                                  direction_vector, "rail_track")

    # END CODE BLOCK


The rail joint is modelled by adding a hinge on the rail track.
The hinge requires the definition of the distance to the joint, starting from the origin point of the track and
the rotational stiffness in the y and z direction.
The hinge is added to the model by specifying the name of the track (in this case "rail_track"), the coordinates
of the joint, the hinge parameters and the name of the hinge.

.. code-block:: python

    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True],
                                                        value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True],
                                                            value=[0, 0, 0])
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=3.0)

    # set hinge rotation stiffness
    distance_joint = 35.75
    hinge_stiffness_y = 37.8e7
    hinge_stiffness_z = 37.8e7

    model.add_hinge_on_beam("rail_track", [(0.75, 3 + rail_pad_thickness, distance_joint)],
                            HingeParameters(hinge_stiffness_y, hinge_stiffness_z), "hinge")

    # END CODE BLOCK

The UVEC model is then defined using the UvecLoad class. The train moves in positive direction from the origin, this is
defined in `direction=[1, 1, 1]`, values greater than 0 indicate positive direction, values smaller than 0 indicate
negative direction.

In this tutorial the train is statically initialised therefore the velocity is set to 0 m/s.
This means that the train is not moving, but the static load of the train is applied to the model, on top of the track,
that includes an extra thickness of the rail-pad, as shown above in `rail_pad_thickness`.

The wheel configuration is defined as a list of distances from the origin point to the wheels. The `uvec_model` is the
imported UVEC train model. The `uvec_parameters` parameter is a dictionary which contains the parameters of the
UVEC model. The UVEC load is added on top of the previously defined track with the name "rail_track".
And the name of the load is set to "train_load".
Because a rail joint is present in the model, the "joint_parameters" key needs to be defined in the `uvec_parameters`
dictionary. If not, the joint will not be taken into account in the UVEC model.
The joint is modelled following the model of dipped joint :cite:`Kabo_2006`, and the parameters are defined as a
dictionary with the following keys:

- "location_joint": the distance from the origin point to the joint in meters
- "depth_joint": the depth of the joint in meters
- "width_joint": the width of the joint in meters.

A schematisation of the UVEC model and the rail joint as defined in this tutorial, is shown below.

.. |uvec_model| image:: _static/figure_uvec.png
    :width: 60%

.. |joint_model| image:: _static/figure_joint.png
    :width: 39%

|uvec_model| |joint_model|


Below the uvec parameters are defined.

.. code-block:: python

    # define uvec parameters
    wheel_configuration=[0.0, 2.5, 19.9, 22.4] # wheel configuration [m]
    velocity = 0 # velocity of the UVEC [m/s]
    uvec_parameters = {"n_carts": 1, # number of carts [-]
                       "cart_inertia": (1128.8e3) / 2, # inertia of the cart [kgm2]
                       "cart_mass": (50e3) / 2, # mass of the cart [kg]
                       "cart_stiffness": 2708e3, # stiffness between the cart and bogies [N/m]
                       "cart_damping": 64e3, # damping coefficient between the cart and bogies [Ns/m]
                       "bogie_distances": [-9.95, 9.95], # distances of the bogies from the centre of the cart [m]
                       "bogie_inertia": (0.31e3) / 2, # inertia of the bogie [kgm2]
                       "bogie_mass": (6e3) / 2, # mass of the bogie [kg]
                       "wheel_distances": [-1.25, 1.25], # distances of the wheels from the centre of the bogie [m]
                       "wheel_mass": 1.5e3, # mass of the wheel [kg]
                       "wheel_stiffness": 4800e3, # stiffness between the wheel and the bogie [N/m]
                       "wheel_damping": 0.25e3, # damping coefficient between the wheel and the bogie [Ns/m]
                       "gravity_axis": 1, # axis on which gravity works [x =0, y = 1, z = 2]
                       "contact_coefficient": 9.1e-7, # Hertzian contact coefficient between the wheel and the rail [N/m]
                       "contact_power": 1.0, # Hertzian contact power between the wheel and the rail [-]
                       "static_initialisation": True, # True if the analysis of the UVEC is static
                       "wheel_configuration": wheel_configuration,
                       "velocity": velocity,
                       "joint_parameters": {"location_joint": distance_joint,  # joint location [m]
                                            "depth_joint": 0.01,  # depth of the joint [m]
                                            "width_joint": 0.25},  # width of the joint [m]
                       }

    # define the UVEC load
    uvec_load = UvecLoad(direction_signs=[1, 1, 1], velocity=velocity, origin=[0.75, 3+rail_pad_thickness, 0],
                         wheel_configuration=wheel_configuration,
                         uvec_model=uvec,
                         uvec_parameters=uvec_parameters)

    # add the load on the tracks
    model.add_load_on_line_model_part("rail_track", uvec_load, "train_load")

    # END CODE BLOCK

The boundary conditions are defined on planes using "DisplacementConstraint" and "AbsorbingBoundary" classes.
The base of the model is fixed in all directions with the name "base_fixed".
For the surfaces at the symmetry plane, roller boundary condition is applied with the name "sides_roller".
To prevent reflections from the sides of the model, absorbing boundaries are applied with virtual thickness of 40 meters.
The boundary conditions are added to the model on the edge surfaces, i.e. the boundary conditions are applied to a list
of surface ids (which can be visualised using: "model.show_geometry(show_surface_ids=True)")  with the corresponding
surface-dimension, "2".

.. code-block:: python

    # define BC
    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])
    absorbing_boundaries_parameters = AbsorbingBoundary(absorbing_factors=[1.0, 1.0], virtual_thickness=40.0)

    model.add_boundary_condition_on_plane([(0, 0, 0), (0, 0, extrusion_length), (5, 0, 0)], no_displacement_parameters,"base_fixed")
    model.add_boundary_condition_on_plane([(0, 0, 0), (0, 0, extrusion_length), (0, 3, 0)], roller_displacement_parameters, "sides_roller")
    #
    model.add_boundary_condition_on_plane([(0, 0, 0), (5, 0, 0), (5, 3, 0)],absorbing_boundaries_parameters,"abs")
    model.add_boundary_condition_on_plane([(0, 0, extrusion_length), (5, 0, extrusion_length), (5, 3, extrusion_length)],absorbing_boundaries_parameters,"abs")
    model.add_boundary_condition_on_plane([(5, 0, 0), (5, 3, 0), (5, 0, extrusion_length)], absorbing_boundaries_parameters, "abs")

    # END CODE BLOCK

Now that the geometry is generated, materials, loads and boundary conditions are assigned. The mesh specifications can
be defined. In this case, the general element size is set to 1.0 and the element size of the soil layer "soil_layer_1_group_2"
is set to 0.2.

.. code-block:: python

    model.set_mesh_size(element_size=1.0)
    model.set_element_size_of_group(element_size=0.2, group_name="soil_layer_1_group_2")

    # END CODE BLOCK

Below it is shown how the solver settings are defined. The analysis type is set to "MECHANICAL" and the solution type of
the first stage is set to "QUASI_STATIC". The start time is set to 0.0 second and the end time is set to 0.1 second. The
time step size is set to 0.025 second. Furthermore, the reduction factor and increase factor are set to 1.0, such that the
time step size is constant throughout the simulation. Displacement convergence criteria is set to 1.0e-4 for the relative
tolerance and 1.0e-12 for the absolute tolerance. No stress initialisation is used. Furthemore, all matrices are assumed
to be constant. Cg is used as a linear solver. Further solver settings are set to the default settings.

.. code-block:: python

    # set time integration parameters
    end_time = 0.1
    delta_time = 0.025
    time_integration = TimeIntegration(start_time=0.0, end_time=end_time, delta_time=delta_time,
                                       reduction_factor=1, increase_factor=1, max_delta_time_factor=1000)

    # set convergence criteria
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-12)

    # set solver settings
    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL,
                                     solution_type=SolutionType.QUASI_STATIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     linear_solver_settings=Cg())

    # END CODE BLOCK

Now the problem data should be set up. The problem should be given a name, in this case it is
"variation_z". The problem will be solved on 4 threads. Then the solver settings are added to the problem. And the problem
definition is added to the model.

.. code-block:: python

    # Set up problem data
    problem = Problem(problem_name="variation_z", number_of_threads=4,
                      settings=solver_settings)
    model.project_parameters = problem

    # END CODE BLOCK

Before starting the calculation, it is required to specify why output is desired. In this case, displacement,
velocity and acceleration is given on the nodes and written to the output file. In this test case, gauss point results
are left empty.

.. code-block:: python

    nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
    gauss_point_results = []

    # END CODE BLOCK

The output process is added to the model using the `Model.add_output_settings` method. The results will be then written to the output directory in vtk
format. In this case, the output interval is set to 1 and the output control type is set to "step", meaning that the
results will be written every time step. The vtk files will be written in binary format in order to save space.

.. code-block:: python

    model.add_output_settings(
        part_name="porous_computational_model_part",
        output_dir=results_dir,
        output_name="vtk_output",
        output_parameters=VtkOutputParameters(
            file_format="binary",
            output_interval=1,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results,
            output_control_type="step"
        )
    )

    # END CODE BLOCK

Now that the the first stage is set up, the calculation is almost ready to be ran.

Firstly the Stem class is initialised, with the model and the directory where the input files will be written to.
While initialising the Stem class, the mesh will be generated.

.. code-block:: python

    stem = Stem(model, input_files_dir)

    # END CODE BLOCK

The second stage can easily be created  by calling the "create_new_stage" function, this function requires the delta time
and the duration of the stage, for the rest, the latest added stage is coppied. In the second stage, the solution type is
set to "DYNAMIC" and the Rayleigh damping coefficients are set to 0.0002 for the stiffness matrix and 0.6 for the mass
matrix. Since the problem is linear elastic, the Linear-Newton-Raphson strategy is used. Furthermore, the velocity of the
moving load is set to move with a velocity of 18 m/s. After the stage is created, and the settings are set, the stage is
added to the calculation.

.. code-block:: python

    delta_time_stage_2 = 0.01
    duration_stage_2 = 1.0
    stage2 = stem.create_new_stage(delta_time_stage_2,duration_stage_2)
    stage2.project_parameters.settings.solution_type = SolutionType.DYNAMIC
    stage2.project_parameters.settings.strategy_type = LinearNewtonRaphsonStrategy()
    stage2.project_parameters.settings.rayleigh_k = 0.0002
    stage2.project_parameters.settings.rayleigh_m = 0.6
    stage2.get_model_part_by_name("moving_load").parameters.velocity = 18.0
    stem.add_calculation_stage(stage2)

    # END CODE BLOCK

The Kratos input files are then written. The project settings and output definitions are written to
ProjectParameters_stage_1.json file. The mesh is written to the .mdpa file and the material parameters are
written to the MaterialParameters_stage_1.json file.
All of the input files are then written to the input files directory.

.. code-block:: python

    stem.write_all_input_files()

    # END CODE BLOCK

The calculation is then ran by calling the run_calculation function within the stem class.

.. code-block:: python

    stem.run_calculation()

    # END CODE BLOCK

.. seealso::

    - Previous: :ref:`tutorial4`
    - Next: :ref:`tutorial6`