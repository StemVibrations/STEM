Materials
=========
This page outlines how to define soil and structural materials in STEM and highlights
practical tips for choosing parameters.

.. _soil_material:
Soil materials
--------------
Soil materials are defined by combining a soil formulation, a soil law, and a saturation law.
Hereby an example of how to define two soil material layers and how to add them to the model.
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

   model.add_soil_layer_by_coordinates([(x1, y1, z1), (x2, y2, z2),(x3, y3, z3), (x4, y4, z4)], material_1, "soil_layer_1")
   model.add_soil_layer_by_coordinates([(x5, y5, z5), (x6, y6, z6),(x7, y7, z7), (x8, y8, z8), (x9, y9, z9)], material_2, "soil_layer_2")


.. _interface_material:
Interface materials
-------------------
Interface materials are zero-thickness soil materials that are defined by combining a soil formulation, a constitutive
law, and a saturation law. Hereby an example of how to define and apply an interface material. Further details regarding the
interface formulation can be found in :ref:`interface_formulation`.

.. code-block:: python

   from stem.soil_material import OnePhaseSoil, LinearElasticSoil, InterfaceMaterial, SaturatedBelowPhreaticLevelLaw

    interface_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
    interface_const_law = LinearElasticSoil(YOUNG_MODULUS=15e5, POISSON_RATIO=0.2)

    interface_material = InterfaceMaterial(name="interface_concrete_soil",
                                           constitutive_law=interface_const_law,
                                           soil_formulation=interface_formulation,
                                           retention_parameters=SaturatedBelowPhreaticLevelLaw())

    model.set_interface_between_model_parts(["soil_layer_1"], ["soil_layer_2"],
                                        interface_material, "interface_between_two_soils")

Interfaces can be applied between more than two layers and material types. Below an example of how to apply an interface
between two soil layers and sleepers. The interface material is defined in the same way as above. Furthermore, it is
assumed a track is generated with volume sleepers as described in :ref:`railway_track`.

.. code-block:: python

    model.set_interface_between_model_parts(["sleeper_track"], ["soil_layer_1", "soil_layer_2"],
                                            interface_material, "interface_sleeper_soil")


.. Random Fields
.. -------------
.. STEM supports the definition of random fields for material parameters, which can be used to model spatial
.. variability in soil properties. This can be done by defining a random field with a specified mean,
.. standard deviation, and correlation length, and then mapping this random field to the material parameters of
.. the soil layers.


Practical tips
--------------
- Units: SI (N, m, kg, s, Pa).
- Names: keep material names unique—used for mapping to physical groups and IO.
- Drainage assumptions: In STEM for railway induced vibrations all the layers should be one-phase drained.
  This means that to simulate saturated layers the Poisson ratio should be set to 0.495. Please refer to
  :ref:`formulation` for further details.


