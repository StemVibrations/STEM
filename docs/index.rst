STEM
====
STEM is an open-source calculation model that is developed to compute the impact of
mitigation techniques on railway induced vibrations.

In STEM the train-track interaction and the propagation of the vibrations through the subsurface
are simulated. The model is able to compute the vibration levels at the ground surface taking into account
the presence of irregularities in the track geometry, the type of train and train speed, and the spatial variability
of the track and soil properties.

.. figure:: _static/STEM_overview.png
   :alt: Example image
   :width: 400

   Scope of the STEM model

The STEM model is based on the finite element method and it is powered by
`Kratos Multiphysics <https://github.com/KratosMultiphysics/Kratos>`_.

User guide
==========
Installation
............
To install STEM you need to install the following items:

* :ref:`python_stem`

* :ref:`kratos`

* :ref:`parav`

Optionally, you can install `gmsh <https://gmsh.info/>`_ to visualise the mesh.


Tutorials
.........

* :ref:`tutorial1`

* :ref:`tutorial2`

* :ref:`tutorial3`

STEM interface definitions
==========================
STEM has interface definitions to interact with the model, and allow the extension of the model and the use
of different train and material models:

   * :ref:`uvec`

   * :ref:`umat`


How to contribute
=================
If you want contribute to STEM please follow the steps defined in :doc:`contributions`.


STEM team
=========
STEM is a research programme that results from a collaboration between the following partners:

* `ProRail <https://www.prorail.nl>`_
* `Deltares <https://www.deltares.nl>`_
* `TNO <https://www.tno.nl>`_
* `TU Delft <https://www.tudelft.nl>`_

The :doc:`authors`.

Package documentation
=====================

The :doc:`stem` documentation.
