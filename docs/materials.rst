Materials
=========

This page outlines how to define soil and structural materials in STEM and highlights
practical tips for choosing parameters.

Soil materials (one-phase, drained)
-----------------------------------
Example: two soil layers and an embankment, linear elastic with drained one-phase formulation.

.. code-block:: python

   from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw

   ndim = 3

   soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
   mat_1 = SoilMaterial(
       "soil_1",
       soil_formulation_1,
       LinearElasticSoil(YOUNG_MODULUS=30e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

   soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2550, POROSITY=0.3)
   mat_2 = SoilMaterial(
       "soil_2",
       soil_formulation_2,
       LinearElasticSoil(YOUNG_MODULUS=30e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

   emb_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
   mat_emb = SoilMaterial(
       "embankment",
       emb_formulation,
       LinearElasticSoil(YOUNG_MODULUS=10e6, POISSON_RATIO=0.2),
       SaturatedBelowPhreaticLevelLaw(),
   )

Structural materials
--------------------
For beams and track components, see :mod:`stem.structural_material`. For example,
linear elastic beam laws can be used via the relevant classes and assigned to the
appropriate model parts.

User-defined materials (UMAT/UDSM)
----------------------------------
- STEM supports external constitutive models via UMAT and UDSM interfaces through Kratos.
- See :doc:`API_definition` for details and links to example material libraries.

Practical tips
--------------
- Units: SI (N, m, kg, s, Pa).
- Drainage assumptions: one-phase drained simplifies to no transient pore pressure; for coupled analyses,
  configure groundwater flow and water-related parameters accordingly.
- Porosity and densities: ensure consistency across layers; check effective vs total stress assumptions
  in the chosen constitutive law.
- Names: keep material names unique—used for mapping to physical groups and IO.
- Validation: start with linear elastic parameters and compare against analytical or coarser models
  before introducing advanced constitutive laws.
