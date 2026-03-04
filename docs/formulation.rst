Formulation
===========

This page summarizes the mathematical and numerical methods used by STEM to simulate
railway-induced ground vibrations.
STEM performs the numerical analysis of the train-track-soil system using the finite element method (FEM)
and is powered by `Kratos Multiphysics <https://github.com/KratosMultiphysics/Kratos>`_.

Governing equation
------------------
STEM solves the dynamic equilibrium equation, following a Total Lagrangian formulation with small strains.
The governing finite element equation to be solved is:

  .. math::
          \mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}\mathbf{u} = \mathbf{F_{ext}}\left( t \right)

where :math:`\mathbf{M}` is the mass matrix, :math:`\mathbf{C}` is the damping matrix, and :math:`\mathbf{K}` is the
stiffness matrix of the entire system, :math:`\mathbf{F_{ext}}` denotes the vector of the external forces and
:math:`\mathbf{a}`, :math:`\mathbf{v}`, :math:`\mathbf{u}` are, respectively, the acceleration, the velocity and
the displacement in the nodes.

In STEM it is also possible to perform quasi-static analyses, in which inertial and damping effects are neglected.
In this case, the governing equation is reduced to:

  .. math::
          \mathbf{K}\mathbf{u} = \mathbf{F_{ext}}\left( t \right).

Spatial discretisation
----------------------
The spatial discretisation of the problem is performed using the finite element method (FEM).
The geometry of the problem is built using `gmsh <http://gmsh.info/>`_ and the mesh is generated
and exported to Kratos mdpa format.

STEM supports several types of elements:

- Beam elements (for 2D and 3D problems):

  * Euler-Bernoulli beam elements (2 nodes).

- 2D plane strain elements:

  * First order triangle elements (3 nodes per element)
  * Second order triangle elements (6 nodes per element)

- 3D solid elements:

  * First order tetrahedral elements (4 nodes per element)
  * Second order tetrahedral elements (10 nodes per element)


Time integration and solution strategies
----------------------------------------
The system of equations is solved in the time domain.
A range of solution strategies are supported, including:

* Implicit and explicit time integration schemes.
* Nonlinear solution strategies (Newton-Raphson, line search, arc length).

For linear systems, it is recommended to use a custom explicit integration scheme based on the
Newmark method, `LinearNewtonRaphsonStrategy`.
This integration scheme takes advantage of the linear nature of the problem, and it is formulated explicitly.
Following the Newmark method, the acceleration and velocity at time step :math:`n+1` are calculated as:

.. math::

    \begin{split}
        \mathbf{v}_{t+\Delta t} & = \mathbf{v}_{t} + \left(1 - \gamma \right)  \Delta t \mathbf{a}_{t} + \gamma \Delta t \mathbf{a}_{t+\Delta t} \\
        \mathbf{u}_{t+\Delta t} &= \mathbf{u}_{t} + \mathbf{v}_{t} \Delta t + \left(\frac{1}{2} - \beta \right)  \Delta t^{2} \mathbf{a}_{t} + \beta \Delta t^{2} \mathbf{a}_{t+\Delta t} \\
    \end{split}

where :math:`\Delta t` is the time step for the numerical integration, the subscript
:math:`\sqcup_{t + \Delta t}` denotes kinematic variables at time :math:`t + \Delta t`, and
:math:`\gamma` and :math:`\beta` are the parameters that control the integration accuracy and stability
(the default values are :math:`\gamma = 0.5` and :math:`\beta=0.25`).

Substituting the Newmark equations into the governing equation, and taking advantage of the linearity of the problem,
the system of equations yields:

.. math::

  \mathbf{\widetilde{K}}  \Delta \mathbf{u}_{t+\Delta t} = \mathbf{F_{ext}}\left( t + \Delta t \right) - \mathbf{M}\mathbf{\widetilde{a}} - \mathbf{C}\mathbf{\widetilde{v}}

where :math:`\mathbf{\widetilde{K}}`, :math:`\mathbf{\widetilde{a}}` and :math:`\mathbf{\widetilde{v}}` are
the effective stiffness, acceleration and velocity, respectively, defined as:

.. math::

  \begin{split}
    \mathbf{\widetilde{K}} & = \mathbf{K} + \frac{\gamma}{\beta \Delta t} \mathbf{C} + \frac{1}{\beta \Delta t^2} \mathbf{M} \\
    \mathbf{\widetilde{a}} & = \frac{1}{\beta \Delta t^2} u_{t} + \frac{1}{\beta \Delta t} \mathbf{v}_{t}  + \left( \frac{1}{2\beta} - 1 \right) \mathbf{a}_{t}\\
    \mathbf{\widetilde{v}} & =  \frac{\gamma}{\beta \Delta t} u_{t} - \left(1 - \frac{\gamma}{\beta} \right) \mathbf{v}_{t} - \left(1 - \frac{\gamma}{2\beta} \right) \Delta t \mathbf{a}_{t}
  \end{split}

The main advantage of using the custom Newmark explicit integration scheme is that it only requires once
the assembly and factorisation of the entire system of matrices, making it computationally efficient for linear problems.

In STEM it is also possible to perform quasi-static analyses.
In the quasi-static case, the governing equation is reduced to:

  .. math::
          \mathbf{K}\mathbf{u} = \mathbf{F_{ext}}\left( t \right)

In this case, the system of equations is solved at each time step.

To solve the linear system of equations of type:

  .. math::
          \mathbf{A}\mathbf{x} = \mathbf{b},

STEM has several linear solvers available, including:

- LU decomposition
- Conjugate Gradient
- Algebraic Multigrid Iterative

The LU decomposition is a direct solver, which is suitable for small problems.
The Conjugate Gradient and Algebraic Multigrid Iterative solvers are iterative solvers, which are suitable for
large problems.
The iterative solvers can use a preconditioner to improve the convergence rate, and they are recommended for
large problems.
In STEM the default preconditioner is the Jacobi preconditioner, which is a simple diagonal preconditioner
that can be used for symmetric positive definite systems.

Boundary conditions
-------------------
STEM support different types of boundary conditions, including:

- Dirichlet boundary conditions (prescribed displacement or rotation)
- Neumann boundary conditions (force or traction boundary conditions)
- Absorbing boundaries :cite:`Lysmer_Kuhlemeyer_1969`

Loads
-----
STEM supports different types of loads, including:

- Point, line, surface and moving loads

All the loads can be defined as time-dependent, and they can be applied to structural elements or soil.

For STEM it is important to be able to model the train-track interaction, which is the main source of the
dynamic loading in the problem of railway-induced vibrations.
To model the train-track interaction force STEM uses point moving loads, in combination with the UVEC
(see :ref:`uvec_formulation`).


Constitutive models
===================
STEM supports linear elastic material models.
Non-linear material models can be implemented by the user and integrated into STEM through the
UMAT interface (see :doc:`API_definition`).

.. _linear_elastic_material:

Linear elastic material
-----------------------
STEM supports linear elastic material models.
In STEM a material is defined by specifying the material formulation and the linear elastic material parameters.

Currently STEM supports the `OnePhaseSoil` material formulation. This formulation is used to model the soil
as a single phase material, and it is defined by the following parameters:

- Density of the solid phase :math:`\rho_s`
- Porosity :math:`n`
- Is_drained (boolean flag to indicate if the soil is drained or not)

The linear elastic material requires the following parameters for the `OnePhaseSoil` formulation:

- Young's modulus :math:`E`
- Poisson's ratio :math:`\nu`

In STEM the soil should be defined as `OnePhaseSoil` and the solution should be 'drained'.
This means that the soil is modelled as a single phase material and that the pore water
pressure is not considered in the analysis.
This is an assumption that is commonly used in the analysis of dynamic problems, such as railway-induced vibrations,
and it is justified by the fact that the time scale of the problem is much shorter than the time scale of the pore
water pressure dissipation.

Typically, the density of the bulk material is available from site investigations.
The bulk density consists of the density of the solid phase and the density of the fluid phase,
and it can be calculated as:

.. math::
    \rho = (1-n)\rho_s + n\rho_f

where :math:`\rho_f` is the density of the fluid phase (water).
Because in STEM the soil is modelled as a drained single phase material this equation simplifies to:

.. math::
    \rho = (1-n)\rho_s

If the user has the bulk density and wants to use it as input, the porosity can be set to
zero and the density of the solid phase can be set as the bulk density.

To model soil layers below ground water table, since the soil is modelled as a drained single phase material,
the Poisson ratio of the soil layer should be set to 0.495. This means that the material is nearly incompressible,
i.e. cannot experience volumetric deformation and this accurately mimics the saturated behaviour of soil, subjected
to dynamic loading, in the short term (before pore water pressure dissipation occurs).

.. _uvec_formulation:

Train-track model
=================
The train and the train-track interaction model can be modelled by a user-defined vehicle model (UVEC)
as described in :doc:`API_definition`.
STEM provides a default train and train-track interaction model.


Vehicle model
--------------
The train is modelled as a 2D mass-spring-damper system, with 10 degrees of freedom, that consists of a car body,
bogies, suspension systems and wheels :cite:`Zhai_Sun_1994`.
This model takes advantage of the system's symmetry by simulating only half of the train, with symmetry boundary
conditions applied to the central plane.
The figure shows the schematic of the train model.

.. figure:: _static/STEM_track.png
   :alt: Train model
   :width: 500

   STEM train model

The railway track consists of rail, railpad and sleeper.
The rail is modelled as an Euler-Bernoulli beam element.
The railpad is modelled as a spring-damper element, which provides the stiffness and damping of the railpad.
The sleeper can either be modelled as a concentrated mass, or a volume element, depending on the level of
detail required by the user.

Interaction model
-----------------

The vehicle and track systems are coupled through contact forces that occur at the interface between
the wheels and the rails. These forces are determined using the non-linear Hertzian contact theory
for metals :cite:`Timoshenko_1951`:

.. math::
    F_{j}=k_\text{c} \ \delta_j^{3/2},

where :math:`k_\text{c}` is the Hertzian stiffness coefficient and :math:`\delta_j` is the indentation for wheel
:math:`j`. The indentation :math:`\delta_j` is calculated as:

.. math::
    \delta_j = u_{\text{w},j}-u_{\text{r},j} - z_j

where :math:`u_{\text{w},j}` is the displacement of wheel :math:`j`, :math:`u_{\text{r},j}` is the displacement
of the rail at the position of wheel :math:`j` and :math:`z_j` is the irregularity of the rail at the position of wheel :math:`j`.

.. _irr_formulation:

Irregularities model
--------------------
The vertical irregularities presented in the rail are expressed following the methodology presented
in :cite:`Zhang_2001`. The power spectral density (PSD) of the rail irregularities :math:`S(\Omega)`
is used to produce the samples of the irregularities.

.. math::
    S(\Omega) = \frac{2 \pi A_v \Omega^{2}_{c}}{(\Omega^2 + \Omega_c^2)\Omega^2}


where :math:`\Omega` the wavenumber, :math:`A_v` the rail irregularity parameter and :math:`\Omega_c`
is the critical wavenumber. These two parameters define the quality of the railway track :cite:`Lei_Noda_2002`.

The sample of rail irregularities can be produced by inverse Fourier transform shown as follows:

.. math::
    r(x) = \sum^{N}_{n=1} \sqrt{4 S(\omega_n) \Delta \omega} \cos(\omega_n x - \theta_n)


where :math:`\omega_n`, is a circular frequency within the interval in which the PSD function is defined,
:math:`\theta_n`, a random phase angle uniformly distributed from 0 to :math:`2\pi`, and
:math:`\Delta \omega` is defined as the total number of frequency increments :math:`N` in the range of the
circular frequency.

The parameter :math:`A_v` can be estimated based on the track quality :cite:`Lei_Noda_2002`.

.. table:: Track quality classification and corresponding Av values
    :widths: 50 50

    +-----------------+------------------------+
    | Line grade      | Av Value (m² rad / m)  |
    +=================+========================+
    | 1 (very poor)   | 1.2107e-4              |
    +-----------------+------------------------+
    | 2               | 1.0181e-4              |
    +-----------------+------------------------+
    | 3               | 0.6816e-4              |
    +-----------------+------------------------+
    | 4               | 0.5376e-4              |
    +-----------------+------------------------+
    | 5               | 0.2095e-4              |
    +-----------------+------------------------+
    | 6 (very good)   | 0.0339e-4              |
    +-----------------+------------------------+


.. _rail_joint_formulation:

Rail joint model
----------------
The rail joint is modelled as a hinge along a beam element, with a rotational stiffness defined based on the properties
of the rail and the joint.
The rail joint might have a local geometric irregularity, commonly referred to as a
dipped joint, following the approach proposed by :cite:`Kabo_2006`.
The dipped joint represents the permanent plastic deformation that develops at rail ends due to repeated
wheel-rail impact loads at insulated joints.

The dip is introduced as a prescribed vertical rail surface irregularity with a smooth cosine shape,
defined by its maximum depth :math:`d` and total length :math:`l`.
The vertical irregularity profile :math:`r(x)` along the rail longitudinal coordinate :math:`x` is given by:

.. math::
  r(x) =
    \begin{cases}
    d \left(1 - \cos\left(\dfrac{\pi x}{l}\right)\right), & 0 < x \le \dfrac{l}{2}, \\
    d \left(1 + \cos\left(\dfrac{\pi x}{l}\right)\right), & \dfrac{l}{2} < x < l,
    \end{cases}

where :math:`x = 0` corresponds to the start of the dipped region.
This formulation ensures a continuous and smooth rail profile with zero slope at the beginning and end of the joint,
avoiding artificial high-frequency excitation.

A schematic of the dipped joint is shown in the figure below.

.. figure:: _static/figure_joint.png
   :alt: Dipped joint model

   Schematic of the dipped joint model