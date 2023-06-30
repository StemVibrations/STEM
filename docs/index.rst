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
It is recommended to install STEM in a Python virtual environment.
The main purpose of Python virtual environments is to create an isolated environment for Python projects.
This means that each project can have its own dependencies, regardless of what dependencies every other project has.
This avoids issues with packages dependencies.

The virtual environment should be installed and activated before the installation of STEM.
To create a virtual environment with python/pip follow this `link <https://docs.python.org/3/library/venv.html>`_.
To create a virtual environment with conda follow this `link <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_.


Installing STEM as a user
.........................
To install STEM, run the following code:

.. code-block::

   pip install git+https://github.com/StemVibrations/STEM.git


Installing STEM as a developer
..............................
To install the package as a developer, you need first to check out the repository.

.. code-block::

   git clone https://github.com/StemVibrations/STEM.git

To install the package in editable mode with the following command:

.. code-block::

   pip install -e .[testing]

This will install the package in editable mode, so that any changes you make to the code will be reflected in the installed package.
The [testing] flag will also install the dependencies needed for running the tests.

Alternatively, you can install the requirements manually with the following command:

.. code-block::

   pip install -r requirements.txt


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
