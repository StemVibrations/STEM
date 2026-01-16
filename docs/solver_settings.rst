Solver settings and modeling choices
===================================

This guide provides practical rules of thumb for mesh size, time step selection,
and Rayleigh damping parameters when configuring :mod:`stem.solver`.

Element size (mesh resolution)
------------------------------
For wave-dominated problems, resolve the shortest wavelength of interest
with sufficient elements. Use the minimum wave speed in your domain (often shear)

.. math::

   h \;\lesssim\; \frac{c_{\min}}{n_\lambda\, f_{\max}} \quad \text{with} \quad c_{\min} = \min(V_S, V_P, V_R)

- :math:`f_{\max}`: highest frequency you want to resolve in the response.
- :math:`n_\lambda`: elements per wavelength; typical values are 10–15 (linear) or 6–10 (quadratic).
- :math:`c_{\min}`: smallest relevant wave speed (shear :math:`V_S` often governs in soils).

Example: :math:`V_S=150\,\text{m/s}`, :math:`f_{\max}=40\,\text{Hz}`, with 12 el/λ ⇒
:math:`h \lesssim 150/(12\cdot 40) \approx 0.31\,\text{m}`.

Time step selection
-------------------
STEM's default dynamic scheme (:class:`stem.solver.NewmarkScheme` with :math:`\beta=0.25,\,\gamma=0.5`) is
unconditionally stable for linear problems, but accuracy still requires a small enough :math:`\Delta t`.
Use one of the following simple rules:

- Frequency-based:

  .. math:: \quad \Delta t \;\lesssim\; \frac{1}{n_t\, f_{\max}}\,, \quad n_t\in[20,50]

- Courant-like (wave travel per step):

  .. math:: \quad \Delta t \;\lesssim\; \frac{h}{n_c\, c_{\min}}\,, \quad n_c\in[5,10]

Pick the more restrictive. For quasi-static analyses, accuracy—not stability—governs; you may use larger steps,
but ensure convergence and path accuracy.

Rayleigh damping parameters
---------------------------
Rayleigh damping in STEM uses mass and stiffness coefficients (``rayleigh_m``, ``rayleigh_k``) in
:class:`stem.solver.SolverSettings`.
The target damping ratio :math:`\zeta(\omega)` at circular frequency :math:`\omega` is

.. math:: \qquad \zeta(\omega) = \tfrac{\alpha}{2\,\omega} + \tfrac{\beta\,\omega}{2}

Given two target damping ratios :math:`\zeta_1, \zeta_2` at frequencies :math:`f_1, f_2`
(:math:`\omega_i=2\pi f_i`), solve for :math:`\alpha` (mass) and :math:`\beta` (stiffness):

.. math::
   \beta = \frac{2\,(\omega_2\,\zeta_2 - \omega_1\,\zeta_1)}{\omega_2^2 - \omega_1^2}\,,\qquad
   \alpha = 2\,\omega_1\,\zeta_1 - \beta\,\omega_1^2

Notes and tips:
- Choose :math:`f_1` near the lowest significant mode and :math:`f_2` near the highest frequency to damp.
- Typical soil/structure modal damping ratios: 1–5% in the band of interest.
- Excessive damping suppresses wave content; verify against measurements or literature.

Putting it together (example)
-----------------------------
.. code-block:: python

   from stem.solver import (
       AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,
       NewmarkScheme, Amgcl, SolverSettings, StressInitialisationType
   )

   # Mesh choice from Vs, f_max, target elements per wavelength
   element_size = 0.3  # m (per guideline above)
   model.set_mesh_size(element_size=element_size)

   # Time step from frequency-based rule
   dt = 1.0 / (30 * 40.0)  # n_t=30, f_max=40 Hz ⇒ ~0.00083 s

   time = TimeIntegration(start_time=0.0, end_time=0.2, delta_time=dt, reduction_factor=1.0, increase_factor=1.0)

   conv = DisplacementConvergenceCriteria(1e-4, 1e-9)

   settings = SolverSettings(
       analysis_type=AnalysisType.MECHANICAL,
       solution_type=SolutionType.DYNAMIC,
       stress_initialisation_type=StressInitialisationType.NONE,
       time_integration=time,
       is_stiffness_matrix_constant=True,
       are_mass_and_damping_constant=True,
       convergence_criteria=conv,
       scheme=NewmarkScheme(),
       linear_solver_settings=Amgcl(),
       rayleigh_m=0.6,           # example values; compute from two target ζ and frequencies
       rayleigh_k=2.0e-4,
   )

See also
--------
- :mod:`stem.solver` for available schemes, strategies and linear solvers.
- :doc:`formulation` for the governing equations and context.
- :doc:`materials`, :doc:`boundary_conditions` for related modeling choices.

Helper scripts
--------------
Below are small, ready-to-run snippets to estimate mesh size, time step and Rayleigh damping.
They follow the guidelines above and can be adapted to your problem.

Wave speeds, mesh size and time step
....................................
.. code-block:: python

   import math

   # Material properties
   E = 50e6        # Young's modulus [Pa]
   nu = 0.49       # Poisson's ratio [-]
   rho = 1500.0    # density [kg/m^3]

   # Target band and resolution
   f_max = 100.0   # highest frequency to resolve [Hz]
   n_lambda = 12   # elements per wavelength (10–15 linear; 6–10 quadratic)
   n_c = 6         # Courant-like divisor for accuracy (5–10)

   # Elastic moduli
   M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))  # P-wave modulus = K + 4/3 G
   G = E / (2.0 * (1.0 + nu))                             # shear modulus

   # Wave speeds
   c_p = math.sqrt(M / rho)
   c_s = math.sqrt(G / rho)

   # Guideline mesh size and time step (use shear speed for worst case in soils)
   h = c_s / (n_lambda * f_max)          # element size [m]
   dt = h / (n_c * c_s)                  # time step [s]

   print(f"Compression-wave speed c_p: {c_p:.3f} m/s")
   print(f"Shear-wave speed      c_s: {c_s:.3f} m/s")
   print(f"Recommended element size h: {h:.4f} m (n_lambda={n_lambda}, f_max={f_max} Hz)")
   print(f"Recommended time step  dt: {dt:.6f} s (n_c={n_c})")

Notes:
- The earlier rule of taking only two elements per wavelength (h = λ/2) is too coarse for FEM wave problems.
  Prefer 10–15 elements per wavelength with linear elements to limit dispersion.
- Newmark is unconditionally stable for linear systems, but accuracy still requires small enough dt.

Rayleigh damping (compute α, β and plot ζ(f))
.............................................
.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   def damping_Rayleigh(f1, d1, f2, d2):
      """
      Compute Rayleigh coefficients (alpha, beta) from two target damping ratios
      d1, d2 at frequencies f1, f2 [Hz].
      """
      if f1 == f2:
         raise ValueError("Frequencies for the Rayleigh damping are the same.")

      # Matrix for: zeta = 0.5 * (alpha / omega) + 0.5 * (beta * omega)
      A = 0.5 * np.array([[1.0 / (2.0 * np.pi * f1), 2.0 * np.pi * f1],
                     [1.0 / (2.0 * np.pi * f2), 2.0 * np.pi * f2]])
      b = np.array([d1, d2])
      alpha, beta = np.linalg.solve(A, b)
      return alpha, beta

   if __name__ == "__main__":
      f1, d1 = 1.0, 0.01
      f2, d2 = 60.0, 0.01

      alpha, beta = damping_Rayleigh(f1, d1, f2, d2)
      print(f"Rayleigh coefficients: alpha={alpha:.6e}, beta={beta:.6e}")

      f = np.linspace(0.1, 100.0, 1001)
      omega = 2.0 * np.pi * f
      zeta_alpha = 0.5 * (alpha / omega)
      zeta_beta  = 0.5 * (beta * omega)
      zeta = zeta_alpha + zeta_beta

      plt.plot(f, zeta_alpha, label=r"$\alpha$ term")
      plt.plot(f, zeta_beta,  label=r"$\beta$ term")
      plt.plot(f, zeta,       label="total")
      plt.plot([f1, f2], [d1, d2], "ro", label="targets")
      plt.xlabel("Frequency [Hz]")
      plt.ylabel("Damping ratio [-]")
      plt.grid(True)
      plt.legend()
      plt.tight_layout()
      plt.show()
