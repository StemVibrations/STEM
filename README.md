# Soil and Track System Modeling Tool

![Tests](https://github.com/StemVibrations/STEM/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/stemvibrations/badge/?version=latest)](https://stemvibrations.readthedocs.io/en/latest/?badge=latest)

This software tool is designed to create and analyze models of soil and track systems. The tool provides a set of commands for creating the geometry of the model, defining the soil and track properties, setting the boundary conditions and loads, generating the mesh, and performing post-processing on the results.

## Installation

The software tool can be downloaded and installed on any system that supports Python. To install the tool, follow these steps:

1. Download the software tool from the GitHub repository.
2. Install the required Python libraries using pip.
3. Install the Gmsh mesh generator.

## Commands

The software tool provides the following commands:

### Geometry Commands

- `add soil layer [name]`: adds a soil layer to the geometry.
- `add embankment [name]`: adds an embankment to the geometry.
- `add ballast [name]`: adds a ballast layer to the geometry.
- `add track [name]`: adds a track to the geometry.
- `import gmsh .geo file [filename]`: imports a Gmsh .geo file.
- `geometry settings`: sets the geometry settings for the model.

### Soil Commands

- `material parameters`: sets the material parameters for the soil.
- `spatial variability`: sets the spatial variability of the soil.
- `soil settings`: sets the soil settings for the model.

### Track Commands

- `material parameters rail`: sets the material parameters for the rail.
- `material parameters sleeper`: sets the material parameters for the sleeper.
- `material parameters rail pad`: sets the material parameters for the rail pad.
- `track settings`: sets the track settings for the model.

### Boundary Condition Commands

- `boundary condition parameters`: sets the boundary condition parameters.
- `use gmsh input`: uses the Gmsh input for the boundary conditions.
- `define plane / line`: defines a plane or line for the boundary conditions.
- `settings`: sets the boundary condition settings.

### Load Commands

- `link to train model`: links the model to a train model.
- `moving load`: sets a moving load on the model.
- `point load`: sets a point load on the model.
- `settings`: sets the load settings for the model.

### Mesh Commands

- `set mesh size volume`: sets the mesh size for the volume.
- `set mesh size surface`: sets the mesh size for the surface.
- `set mesh size lines`: sets the mesh size for the lines.


## Settings

The following commands can be used to set the settings for the train track:

- integration scheme
- damping parameters
- time parameters
- etc

## stages

- possibility for multistages
- per stage, change calculation settings
- per stage change material parameters
- boundary conditions and loads
## Post Processing

The following commands can be used to post process the results of the train track simulation:

- local data
    -- coordinate
    -- parameter type
- field data
    -- parameter type

## Create Model

The following commands can be used to create the model for the train track:

- import stem
- create geometry
- create and assign soil layers
- create and assign boundary conditions
- create and assign loads

- Set calculation settings
- Set output settings

- generate mesh

- dump input

- initialisation
- run
- post process

## Rules

- max 1 level of inheritance
- only use license compatible packages