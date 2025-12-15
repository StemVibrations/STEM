STEM
====
STEM is an open-source calculation model developed to compute the impact of
mitigation techniques on railway-induced vibrations.

In STEM the train–track interaction and the propagation of the vibrations through the subsurface
are simulated. The model can compute the vibration levels at the ground surface taking into account
the presence of irregularities in the track geometry, the type of train and train speed, and the spatial variability
of the track and soil properties.

.. figure:: _static/STEM_overview.png
   :alt: Example image
   :width: 400

   Scope of the STEM model

The STEM model is based on the finite element method and it is powered by
`Kratos Multiphysics <https://github.com/KratosMultiphysics/Kratos>`_.

Background & prerequisites
==========================
STEM is a finite-element railway vibration solver. A quick refresher on the
following topics will help you get the most out of the platform:

* **Numerics:** introductory texts on the finite element method.
* **Programming:** STEM is written in Python. Good starting points are the
   `Python beginners guide <https://wiki.python.org/moin/BeginnersGuide>`_,
   `Introduction to Python <https://www.udacity.com/course/introduction-to-python--ud1110>`_, and
   `Programming for Everybody <https://www.coursera.org/learn/python>`_.
* **Tooling:** we recommend `PyCharm <https://www.jetbrains.com/pycharm/>`_ or
   `Visual Studio Code <https://code.visualstudio.com/>`_. The quick-start pages for
   `VS Code <https://code.visualstudio.com/docs/languages/python>`_ and
   `PyCharm <https://www.jetbrains.com/help/pycharm/quick-start-guide.html>`_ walk through
   STEM-style workflows.
* **Visualisation:** post-processing relies on `ParaView <https://www.paraview.org/>`_;
   the official `ParaView tutorials <https://www.paraview.org/tutorials/>`_ cover the basics.


Installation overview
=====================
STEM ships with a Python interface (:ref:`python_stem`) and ParaView utilities (:ref:`parav`).
Follow the :doc:`installation` guide to set up both components and optionally add
`gmsh <https://gmsh.info/>`_ for mesh inspection.


Tutorial track
==============
Work through the tutorials in order to get hands-on experience:

* :ref:`tutorial1`
* :ref:`tutorial2`
* :ref:`tutorial3`
* :ref:`tutorial4`
* :ref:`tutorial5`


Interface contracts
===================
STEM exposes several extension points so you can plug in custom train and material
models. The interface definitions are documented in :doc:`API_definition`, including:

* :ref:`uvec` – user-defined vector evolution components.
* :ref:`umat` – user-defined material models for soil or structure behavior.

For a full per-module reference, see :doc:`api` which aggregates every public
function discovered by ``sphinx-apidoc``.


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

See :doc:`authors` for the full list of maintainers and contributors.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   known_issues

.. toctree::
   :maxdepth: 2
   :caption: Formulation and theory

   formulation
   bibliography

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial1
   tutorial2
   tutorial3
   tutorial4
   tutorial5

.. toctree::
   :maxdepth: 2
   :caption: Simulation concepts

   materials
   boundary_conditions
   loads
   outputs
   solver_settings

.. toctree::
   :maxdepth: 2
   :caption: Interface definitions

   API_definition

.. toctree::
   :maxdepth: 1
   :caption: Developer reference

   stem
   api

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributions
   authors

.. References
.. ==========

.. :doc:`bibliography` in STEM.