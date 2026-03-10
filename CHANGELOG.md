# Change Log
All notable changes to STEM are documented in this file.

##  [Version 1.3] - 2026-03-05

### New Features
- Support for second order elements.
- Sleepers can now be modelled as volume elements.
- Added multiple linear solver options.
- Podman configuration files have been added to support more general deployment and usage.
- Introduced a logging system.
- A verification document is now automatically generated for every commit to the main branch.

### Improvements
- Default solver settings have been updated to better support large-scale 3D dynamic simulations.
- JSON output performance has been improved.
- The documentation has been expanded and reorganized for improved clarity.
- Improved convergence in the 10-DOF UVEC.

### Changes
- Python 3.10 to Python 3.12 is now supported.
- The "direction" key in "MovingLoad" and "UvecLoad" has been renamed to "direction_signs".
- The "active" key has been removed from "DisplacementConstraint" and "RotationConstraint" boundary conditions.

### Bug Fixes
- Fixed an error in the damping matrix that occurred when Rayleigh damping was not defined.
- Fixed an error in the 3D "MovingLoad" and "UvecLoad" where rotation was wrongly calculated.
- Fixed an issue in the 3D "EulerBeam"  where terms were added to the mass matrix, which only should have been added for Timoshenko beams.


## [Version 1.2.3] - 2025-04-11

### New Features
- Possibility to add a rail joint to the track.


## [Version 1.2] - 2025-01-17

### New Features
- New linear solver.
- Train initialization outside the model domain.
- Integration of UVEC as a package within STEM.
- Support for cloud computing environments.
- Multistage construction capabilities.
- New definition of boundary conditions (by definition of a plane).
- Ability to append geometries to existing models.
- Support for spatial variability during layer generation.
- CPT-based soil model generation using conditional random fields.

### Bug Fixes
- Fixed a bug affecting simulations with multiple train wagons.


## [Version 1.1] - 2024-06-31

### Bug Fixes
- Fixed bug in the mass matrix

## [Version 1.0] - 2024-01-29

### New Features
- 2D and 3D static & dynamic FEM (powered by Kratos Multiphysics)
- Multilayering and complex geometry generation
- Meshing
- UVEC (10 dof and 2 dof)
- Random Fields to model spatial variability
- Train track interaction model


