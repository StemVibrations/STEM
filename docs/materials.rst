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
- Drainage assumptions: In STEM it is assumed that all the layers are one-phase drained.
  This means that to simulate saturared layers the Poisson ratio should be set to 0.495.


