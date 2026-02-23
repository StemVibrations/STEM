---
title: 'STEM: Railway induced vibrations model'
tags:
  - Python
  - Railways
  - Vibrations
  - Finite element method
  - Train-track interaction
authors:
  - name: A. Noordam
    orcid: 0000-0002-2559-9465
    affiliation: 1
  - name: B. Zuada Coelho
    orcid: 0000-0002-9896-3248
    corresponding: true
    affiliation: 1
  - name: J. Nuttall
    orcid:
    affiliation: 2
  - name: M. Fathian
    orcid:
    affiliation: 2
  - name: D. Moretti
    orcid: 0009-0005-4042-7062
    affiliation: 3
affiliations:
 - name: Deltares, Safe and Resilient Infrastructure, the Netherlands
   index: 1
 - name: Deltares, Digital Technology Centre, the Netherlands
   index: 2
 - name: HHLA PLT, Italy
   index: 3
date: 23-02-2026
bibliography: references.bib
---


# Summary

STEM is an open-source software package for the numerical simulation of railway-induced ground vibrations.
It provides a finite element–based framework to model the coupled train–track–soil system.
STEM accounts for train dynamics, wheel–rail interaction, track irregularities, soil stratigraphy,
and spatially variability of the subsurface parameters.
STEM is developed for the evaluation of mitigation measures in railway engineering applications.
STEM is implemented in Python and is built on top of Kratos Multiphysics (@Dadvand_2010),
providing a flexible and extensible platform for advanced dynamic analyses relevant to railway engineering
and environmental vibration assessment.


# Statement of need

Railway-induced vibrations pose significant challenges in densely populated areas.
These vibrations, produced by the interaction between train and track, propagate through the subsoil and may
affect nearby buildings, infrastructure, and human comfort.
Reliable assessment of railway induced vibrations and the effectiveness of mitigation techniques requires numerical
tools capable of representing the dynamic interaction between vehicles, track components, and the subsurface.

Existing commercial software solutions are closed-source and lack the possibility to model
custom vehicle models, custom train-track interaction models and subsurface spatial variability.
STEM addresses this gap by providing an open-source, customisable, research-oriented and practice-friendly package.
STEM is designed specifically for railway-induced vibration analysis, and its transparent implementation supports
reproducibility, methodological development, and direct integration into research and consultancy workflows.


# Software description

STEM performs time-domain analyses of railway-induced vibrations using the Finite Element Method (FEM).

STEM follows a layered, modular architecture. STEM consists of the following main components:

- [STEM](https://github.com/StemVibrations/STEM/): The core module responsible for model definition, input generation, and results processing.

- [gmsh utils](https://github.com/StemVibrations/gmsh_utils/): A utility module for mesh generation using Gmsh, including geometry definition and meshing strategies.

- [vehicle models](https://github.com/StemVibrations/vehicle_models/): A module containing predefined vehicle models, such as the 2D mass-spring-damper system for train modeling, and interfaces for user-defined vehicle formulations.

- [random fields](https://github.com/StemVibrations/RandomFields/): A package for generating subsurface parameter fields with spatial variability, supporting various correlation structures and statistical properties.

STEM main features are summarized as follows:

- 2D and 3D quasi-static and dynamic analysis

  - Multi layering and complex geometry generation

  - Meshing

  - I/O and visualization

- Coupled train–track model

  - User-defined vehicle models (UVEC): 2 and 10 degrees of freedom (DOF) mass-spring-damper systems

  - Nonlinear wheel–rail contact based on Hertzian contact theory

  - Track irregularity and discontinuity generation

- Railway track

  - Rail, railpad, and sleeper

- UMAT interface for user-defined material models

- Comprehensive documentation, tutorials, and API reference

- Open-source distribution via pip and GitHub

STEM solves the dynamic equilibrium equation, following a Total Lagrangian formulation with small strains:

$$\mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}\mathbf{u} = \mathbf{F_{ext}}\left( t \right)$$

where $\mathbf{M}$ is the mass matrix, $\mathbf{C}$ is the damping matrix, and $\mathbf{K}$ is the stiffness matrix of the entire system, $\mathbf{F_{ext}}$ denotes the vector of the external forces and $\mathbf{a}$, $\mathbf{v}$, $\mathbf{u}$ are, respectively, the acceleration, the velocity and the displacement in the nodes.

The train-track interaction is modelled using a loosely coupled formulation, where the train is represented as a multi-degree-of-freedom mass–spring–damper system and the rail is discretized using Euler–Bernoulli beam elements. The wheel–rail contact forces are computed based on nonlinear Hertzian contact theory, which accounts for the local deformation at the contact interface. Track irregularities are incorporated through stochastic and deterministic models, allowing for the simulation of realistic track conditions, including rail joints and surface roughness.


# Getting STEM

STEM is distributed as an open-source Python package and supports Python versions 3.10–3.12.
STEM can be downloaded from the Python Package Index (PyPI) and installed using pip:

.. code-block:: bash

    pip install STEM-Vibrations

or cloned from the GitHub repository for access to the latest development version.

STEM has a detailed documentation [website](https://stemvibrations.readthedocs.io/), which includes installation instructions, background on the formulation and theory behind STEM, STEM definitions, and API references. The documentation also provides tutorials and example cases to help users get started with the software.

STEM also includes a [benchmark report](https://github.com/StemVibrations/STEM/releases/download/pdf-latest/benchmark_report.pdf), which presents a comprehensive set of benchmark cases comparing STEM results with analytical solutions. This report serves as a validation resource for users and developers, demonstrating the accuracy and reliability of STEM in various scenarios relevant to railway-induced vibrations.


# Research impact statement

STEM enables reproducible and extensible research on railway-induced vibrations and the effect of mitigation techniques.

STEM's open-source nature supports method development, benchmarking, and comparison with analytical and experimental results.
The software has been designed to accommodate future extensions, such as advanced constitutive soil models or alternative vehicle formulations, making it suitable for both academic research and consulting engineering studies.
By integrating detailed train–track interaction with full-field soil response, STEM provides a unified framework that is often fragmented across multiple tools in existing workflows.


# AI usage disclosure

AI-based tools were used in STEM.

Copilot has been used in STEM to accelerate the coding process, particularly for routine coding tasks and boilerplate code generation. The authors have reviewed and edited all AI-generated content to ensure technical accuracy and consistency with the software's design principles. No AI-generated content was used without human review, and all final decisions regarding the software's implementation were made by the authors.

Chat-GPT was used to assist in drafting and editing the manuscript. In particular to improve clarity, grammar, and overall readability, while ensuring that all technical content, software descriptions, and methodological details were accurately represented. The authors have carefully reviewed and edited all AI-generated text to ensure that it accurately reflects the software's capabilities and the research context.

# Acknowledgements

The authors acknowledge ProRail for funding and supporting the development of STEM.

# References