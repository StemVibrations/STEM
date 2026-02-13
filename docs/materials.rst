Materials
=========
This page outlines how to define soil and structural materials in STEM and highlights
practical tips for choosing parameters.

.. _soil_material:
Soil materials
--------------
Soil materials are defined by combining a soil formulation, a soil law, and a saturation law.
Hereby an example of how to define two soil material layers.
Further details about the soil formulation can be found in :ref:`linear_elastic_material`.

.. code-block:: python

   from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw

   ndim = 3

   soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
   material_1 = SoilMaterial(
       "soil_1",
       soil_formulation_1,
       LinearElasticSoil(YOUNG_MODULUS=30e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

   soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2550, POROSITY=0.3)
   material_2 = SoilMaterial(
       "soil_2",
       soil_formulation_2,
       LinearElasticSoil(YOUNG_MODULUS=30e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )


Railway track
-------------
To model the railway track, the user can define materials for the rail, railpad and sleepers.

The rail consists of a Euler-Bernoulli beam element. Since the rail properties are well-know,
STEM provides a default material for the rail, which can be used as follows:

.. code-block:: python

    from stem.default_materials import DefaultMaterial

    # Rail and sleeper parameters
    rail_parameters = DefaultMaterial.Rail_54E1_3D.value.material_parameters

STEM supports the following types of rails:

- Rail_46E3_3D
- Rail_54E1_3D
- Rail_60E1_3D

Custom rail materials can be defined by defining a Euler-Bernoulli beam material as described in :doc:`api`.

The railpad is modelled as a spring-damper system, which can be defined as follows:

.. code-block:: python

    from stem.structural_material import ElasticSpringDamper

    rail_pad_parameters = ElasticSpringDamper(
                            NODAL_DISPLACEMENT_STIFFNESS=[0, 750e6, 0],
                            NODAL_ROTATIONAL_STIFFNESS=[0, 0, 0],
                            NODAL_DAMPING_COEFFICIENT=[0, 750e3, 0],
                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[0, 0, 0]
                            )

The sleeper can be either modelled as a concentrated mass or as a volume element.
The following example shows how to model the sleeper as a concentrated mass:

.. code-block:: python

    from stem.structural_material import NodalConcentrated

    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                           NODAL_MASS=140,
                                           NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

To model the sleeper as a volume element the sleeper material is defined as  the soil material (see :ref:`soil_material`).

The railway track is build by assembling the rail, railpad and sleepers:

.. code-block:: python

    origin_point = [0.75, 3.0, 0.0]
    direction_vector = [0, 0, 1]
    number_of_sleepers = 101
    sleeper_spacing = 0.6
    rail_pad_thickness = 0.025

    model.generate_straight_track(sleeper_spacing,
                                  number_of_sleepers,
                                  rail_parameters,
                                  sleeper_parameters,
                                  rail_pad_parameters,
                                  rail_pad_thickness,
                                  origin_point,
                                  direction_vector,
                                  "rail_track")


Practical tips
--------------
- Units: SI (N, m, kg, s, Pa).
- Names: keep material names unique—used for mapping to physical groups and IO.
- Drainage assumptions: In STEM it is assumed that all the layers are one-phase drained.
  This means that to simulate saturared layers the Poisson ratio should be set to 0.495.
