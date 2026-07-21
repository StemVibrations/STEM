```
     _______..___________. _______ .___  ___.
    /       ||           ||   ____||   \/   |
   |   (----``---|  |----`|  |__   |  \  /  |
    \   \        |  |     |   __|  |  |\/|  |
.----)   |       |  |     |  |____ |  |  |  |
|_______/        |__|     |_______||__|  |__|

```

# STEM: Soil and Track System Modelling Tool
![Tests](https://github.com/StemVibrations/STEM/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/stemvibrations/badge/?version=latest)](https://stemvibrations.readthedocs.io/)
[![codecov](https://codecov.io/gh/stemvibrations/stem/graph/badge.svg?token=0DMYCZF4BU)](https://codecov.io/gh/stemvibrations/stem)
[![PyPI version](https://img.shields.io/pypi/v/STEM-Vibrations.svg)](https://pypi.org/project/STEM-Vibrations/)
[![Python versions](https://img.shields.io/pypi/pyversions/STEM-Vibrations.svg)](https://pypi.org/project/STEM-Vibrations/)
[![License](https://img.shields.io/pypi/l/STEM-Vibrations.svg)](https://pypi.org/project/STEM-Vibrations/)

STEM is an open-source finite element model for computing railway-induced vibrations and evaluating the effectiveness of mitigation measures.
It simulates the train–track interaction and the propagation of vibrations through the subsurface, taking into account track irregularities, the type of train and train speed, and the spatial variability of the track and soil properties.
STEM is powered by [Kratos Multiphysics](https://github.com/KratosMultiphysics/Kratos).

The tool provides a set of commands for creating the geometry of the model, defining the soil and track properties, setting the boundary conditions and loads, generating the mesh, and performing post-processing on the results.

## Features
- Parametric geometry, meshing, and model generation for 2D (plane strain) and 3D soil–track systems
- Beam, triangular, and tetrahedral element support (linear and quadratic order)
- Train–track interaction via a user-defined vehicle model (UVEC), including rail irregularities and dipped rail joints
- Linear elastic soil models, with non-linear models available through the UMAT API definition
- Point, line, surface, and moving loads
- Dirichlet, Neumann, and absorbing (Lysmer) boundary conditions
- Implicit and explicit time integration, dynamic analysis, quasi-static analyses, and multi-stage simulations
- Post-processing and output to JSON and VTK for visualisation in ParaView

## Requirements
- Python 3.10, 3.11, or 3.12
- [ParaView](https://www.paraview.org/) (to visualise results)
- [Git](https://git-scm.com/)

## Installation
STEM can be downloaded and installed on any system that supports Python.
It is recommended to install STEM inside a Python virtual environment. To install the latest stable release, run:

```bash
pip install STEM-Vibrations
```

Further details, including the development and editable (developer) installs, can be found in the [installation guide](https://stemvibrations.readthedocs.io/main/installation.html#).

## Usage
To get started, refer to the [tutorials](https://stemvibrations.readthedocs.io/main/#tutorials), which build up from basic usage to more advanced features.
For the theory and numerical methods behind the model, see the [formulation documentation](https://stemvibrations.readthedocs.io/main/formulation.html).

## Benchmarks
STEM is validated against analytical solutions. The automatically generated benchmark report is available as a PDF [here](https://github.com/StemVibrations/STEM/releases/download/pdf-latest/benchmark_report.pdf).
In addition to these analytical benchmarks, STEM contains a set of benchmarks and unit tests that run automatically
on every commit to the repository.

## Contributing
Contributions are welcome. Please refer to the [contribution guidelines](https://stemvibrations.readthedocs.io/main/contributions.html).

## Authors
See the full list of [authors and contributors](https://stemvibrations.readthedocs.io/main/authors.html).

## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
