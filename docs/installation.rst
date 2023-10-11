STEM installation
=================

.. _python_stem:

STEM python package
-------------------
It is recommended to install STEM in a Python virtual environment.
The main purpose of Python virtual environments is to create an isolated environment for Python projects.
This means that each project can have its own dependencies, regardless of what dependencies every other project has.
This avoids issues with packages dependencies.

The virtual environment should be installed and activated before the installation of STEM.
To create a virtual environment with pip follow this `link <https://docs.python.org/3/library/venv.html>`_.
To create a virtual environment with conda follow this `link <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_.

Currently, STEM supports Python 3.10, therefore, it is recommended to create the virtual environment with Python 3.10.

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

   pip install -e ."[testing]"

This will install the package in editable mode, so that any changes you make to the code will be reflected in the installed package.
The [testing] flag will also install the dependencies needed for running the tests.

Alternatively, you can install the requirements manually with the following command:

.. code-block::

   pip install -r requirements_dev.txt


.. _kratos:

Kratos
------
To install Kratos Multiphysics download the Kratos repository from `here <https://github.com/StemVibrations/StemKratos>`_.
Once the repository is downloaded, place the *KratosGeomechanics* folder in a directory of your choice.
This directory will be necessary for the tutorials.


.. _paraview:

ParaView
--------
To visualise the results, STEM makes use of `ParaView <https://www.paraview.org/>`_.
ParaView is an open-source multiple-platform application for interactive, scientific visualisation. It is recommended to
download ParaView to visualise the results of the tutorials.