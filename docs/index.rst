STEM
====
STEM is an open-source calculation model developed to compute the impact of
mitigation techniques on railway-induced vibrations.

In STEM the train-track interaction and the propagation of the vibrations through the subsurface
are simulated. The model can compute the vibration levels at the ground surface taking into account
the presence of irregularities in the track geometry, the type of train and train speed, and the spatial variability
of the track and soil properties.

.. figure:: _static/STEM_overview.png
   :alt: STEM overview
   :width: 400

   Scope of the STEM model

The STEM model is based on the finite element method and it is powered by
`Kratos Multiphysics <https://github.com/KratosMultiphysics/Kratos>`_.


Background & prerequisites
==========================
STEM is a numerical model based on the finite element method.
For more information on the formulation and theory behind the model, see :doc:`formulation`.
It is recommended to have a basic understanding of the Finite Element theory before using STEM.
For a more detailed introduction to the finite element method, we recommend the following resources:

* `The Finite Element Method: Its Basis and Fundamentals <https://search.worldcat.org/title/857713191>`_
* `Finite Element Procedures <https://search.worldcat.org/title/191703381>`_

STEM is built in Python. It is recommended to have a basic understanding of Python before using STEM.
For new users of Python, the following resources are recommended:

* `Python beginners guide <https://wiki.python.org/moin/BeginnersGuide>`_

Alternatively, you can follow one of the many online free courses, for example (the STEM team is not
affiliated to any of these courses):

* `Introduction to Python <https://www.udacity.com/course/introduction-to-python--ud1110>`_
* `Programming for Everybody <https://www.coursera.org/learn/python>`_

To use STEM it is convenient to use an IDE (Integrated Development Environment) for Python. The STEM team
recommends to use `PyCharm <https://www.jetbrains.com/pycharm/>`_ or `Visual Studio Code <https://code.visualstudio.com/>`_.
More information can be found in the following links:

* `VS Code in Python <https://code.visualstudio.com/docs/languages/python>`_
* `PyCharm in Python <https://www.jetbrains.com/help/pycharm/quick-start-guide.html>`_

To visualise the results STEM makes use of `ParaView <https://www.paraview.org/>`_. It is recommended to have a basic
understanding of ParaView before using STEM. The following resources are recommended:

* `ParaView tutorials <https://www.paraview.org/tutorials/>`_


.. Contents
.. ========
.. This creates the TOC for the side pane

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installation
   known_issues

.. toctree::
   :maxdepth: 3
   :caption: Formulation and theory
   :hidden:

   formulation

.. toctree::
   :maxdepth: 2
   :caption: STEM definitions
   :hidden:

   materials
   boundary_conditions
   loads
   track
   outputs
   solver_settings

.. toctree::
   :maxdepth: 2
   :caption: Interface definitions
   :hidden:

   API_definition


.. toctree::
   :maxdepth: 2
   :caption: Examples and benchmarks
   :hidden:

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Developer reference
   :hidden:

   api

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   contributions
   authors

.. References
.. ==========

.. :doc:`bibliography` in STEM.