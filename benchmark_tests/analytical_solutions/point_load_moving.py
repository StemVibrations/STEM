import numpy as np
import numpy.typing as Npt
from scipy.optimize import brentq
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


class MovingLoadElasticHalfSpace:
    """
    Implementation of the steady-state response of an elastic half-space
    to a moving point load (Vertical Displacement uz).
    Based on Liao et al. (2005).
    """

    def __init__(self, E: float, nu: float, rho: float, force: float, speed: float):
        self.E = E
        self.nu = nu
        self.rho = rho
        self.force = force
        self.speed = speed

        # Lamé parameters
        self.G = E / (2 * (1 + nu))  # Shear modulus (mu)
        self.M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

        # Wave speeds [cite: 71]
        self.cp = np.sqrt(self.M / self.rho)
        self.cs = np.sqrt(self.G / self.rho)
        self.cr = self._calculate_rayleigh_speed()

        # Check subsonic condition [cite: 128]
        if self.speed >= self.cr:
            raise ValueError(f"Speed {self.speed:.2f} m/s must be sub-Rayleigh (< {self.cr:.2f} m/s)")

        # Mach numbers and Beta parameters [cite: 120]
        self.Mp = self.speed / self.cp
        self.Ms = self.speed / self.cs
        self.Mr = self.speed / self.cr

        self.beta_p = np.sqrt(1 - self.Mp**2)
        self.beta_s = np.sqrt(1 - self.Ms**2)
        self.beta_r = np.sqrt(1 - self.Mr**2)

        # variables
        self.vertical_displacement = None

    def _calculate_rayleigh_speed(self):
        """Solves the Rayleigh equation for cr."""
        eta = self.cs / self.cp

        def rayleigh_func(xi):
            # Eq: xi^3 - 8xi^2 + 8xi(3-2eta^2) - 16(1-eta^2) = 0 where xi = (cr/cs)^2
            return xi**3 - 8 * xi**2 + 8 * xi * (3 - 2 * eta**2) - 16 * (1 - eta**2)

        xi_sol = brentq(rayleigh_func, 0.1, 0.99)  # Search for (cr/cs)^2
        return np.sqrt(xi_sol) * self.cs

    def compute_vertical_displacement(self,
                                      x: float,
                                      y: float,
                                      z: float,
                                      t: float,
                                      ky_max: float = 10.0,
                                      n_ky: int = 200,
                                      n_tau: int = 100):
        """
        Computes u_z at position (x,y,z) and time t.
        Integration corresponds to Eq. (10c) [cite: 119] and Eq. (13) [cite: 138].
        """
        # Cylindrical coordinates for the field point [cite: 160]
        r = np.sqrt(x**2 + z**2)
        if r < 1e-9: r = 1e-9  # Avoid singularity
        theta = np.arctan2(z, x)  # theta in [0, pi] for z>=0

        # Moving coordinate y' [cite: 116]
        y_prime = y - self.speed * t

        # Integration range for ky
        # Note: Integrand is symmetric/conjugate; we integrate -ky_max to ky_max
        kys = np.linspace(-ky_max, ky_max, n_ky)

        # Result accumulator
        integral_ky = np.zeros_like(kys, dtype=complex)

        # Region Classification angles [cite: 232, 233]
        theta_r = np.arccos(self.beta_r / self.beta_p)
        theta_s = np.arccos(self.beta_s / self.beta_p)

        # Determine Region [cite: 230]
        # Note: Using absolute theta for symmetry if x < 0
        ang = np.abs(theta)
        region_p = 'I'
        if theta_r < ang < np.pi - theta_r:
            region_p = 'I'
        elif (theta_s < ang <= theta_r) or (np.pi - theta_r <= ang < np.pi - theta_s):
            region_p = 'II'
        else:
            region_p = 'III'

        # Pre-compute Gauss-Hermite quadrature nodes for tau if desired,
        # or use linear trapezoid for simplicity as per user request.
        # Paper suggests tau limits based on decay exp(-r * ky * tau^2).

        for i, ky in enumerate(kys):
            if np.abs(ky) < 1e-6: continue  # Skip zero wavenumber singularity

            # 1. Compute Gp (P-wave contribution) [cite: 140]
            gp_val = self._compute_G_component(ky, r, theta, z, x, n_tau, 'P', region_p)

            # 2. Compute Gs (S-wave contribution) [cite: 141]
            # Gs region logic is simpler (Fig 5): Region I or II [cite: 324]
            theta_r_star = np.arccos(self.beta_r / self.beta_s)
            region_s = 'I' if (theta_r_star < ang < np.pi - theta_r_star) else 'II'
            gs_val = self._compute_G_component(ky, r, theta, z, x, n_tau, 'S', region_s)

            integral_ky[i] = (gp_val + gs_val) * np.exp(-1j * ky * y_prime)

        # Final Integration over ky
        # The factor is -Q / (4 pi^2 mu)
        total_integral = trapezoid(integral_ky, kys)
        u_z = (-self.force / (4 * np.pi**2 * self.G)) * total_integral

        self.vertical_displacement = np.real(u_z)
        return np.real(u_z)

    def _compute_G_component(self, ky, r, theta, z, x, n_tau, wave_type, region):
        """
        Computes Gp or Gs inner integral using SDP.
        """
        beta = self.beta_p if wave_type == 'P' else self.beta_s

        # SDP Integration limits
        # Decay factor is exp(-|ky| * r * tau^2) [cite: 206]
        # We need |ky|*r*tau^2 approx 10 for convergence
        tau_limit = np.sqrt(12.0 / (np.abs(ky) * r))
        taus = np.linspace(-tau_limit, tau_limit, n_tau)

        # 1. Evaluate Integral along SDP [cite: 27, 327]
        # kx_tilde parametrization Eq (24)
        # Using abs(ky) for stability, sign handled by kx reconstruction
        kx_tilde, dkx_dtau = self._sdp_path(taus, theta, beta)

        # Compute E/F integrand terms
        integrand_vals = self._integrand_EF(kx_tilde, ky, wave_type, z)

        # Full integrand for SDP
        # G = ky * exp(-ky*r*beta) * int( E/F * exp(-ky*r*tau^2) * dkx/dtau )
        decay_sdp = np.exp(-np.abs(ky) * r * taus**2)
        integral_sdp = trapezoid(integrand_vals * decay_sdp * dkx_dtau, taus)

        G_val = np.abs(ky) * np.exp(-np.abs(ky) * r * beta) * integral_sdp

        # 2. Pole Correction (Regions II and III) [cite: 243, 330]
        if region in ['II', 'III']:
            # Pole location: -i * beta_R * sgn(cos_theta)
            # Note: The paper uses a specific sgn convention.
            # If x>0 (cos>0), pole is at -i*beta_R.
            sign_cos = np.sign(np.cos(theta))
            if sign_cos == 0: sign_cos = 1
            kx_pole = -1j * self.beta_r * sign_cos

            # Residue Term
            E_pole, F_prime_pole = self._residue_terms(kx_pole, ky, wave_type, z)

            # Eq 28: -sgn(cos) * 2*pi*i * (E/F') * exp(...)
            # Exponent: -ky * (v_R*z + beta_R*x*sgn)
            # v_R for P: sqrt(beta_p^2 - beta_R^2) -- likely imaginary?
            # Actually beta_R < beta_s < beta_p. So beta_p^2 - beta_R^2 > 0. Real.

            if wave_type == 'P':
                v_pole = np.sqrt(self.beta_p**2 - self.beta_r**2)
            else:
                v_pole = np.sqrt(self.beta_s**2 - self.beta_r**2)

            exponent = -np.abs(ky) * (v_pole * z + self.beta_r * x * sign_cos)
            residue = -sign_cos * 2j * np.pi * (E_pole / F_prime_pole) * np.exp(exponent)

            G_val += residue

        # 3. Branch Cut Integral (Region III for Gp only)
        # Only Gp involves v' branch cut crossing. Gs does not cross its own cut.
        if wave_type == 'P' and region == 'III':
            # Path: Along imaginary axis from -i*beta_s to -i*beta_p*cos(theta)
            # Variables: kx_tilde = -i * xi
            # Range: xi from beta_s to beta_p * cos(theta) (approx)
            # The saddle is at -i * beta_p * cos(theta).
            # If Region III, beta_p * cos(theta) > beta_s.

            xi_start = self.beta_s
            xi_end = self.beta_p * np.cos(theta)

            # Integrate if the range is valid
            if xi_end > xi_start:
                xis = np.linspace(xi_start, xi_end, 50)
                kx_bc = -1j * xis

                # Integrand difference across cut
                # On the cut, v' is imaginary. v' = sqrt(kx^2 + beta_s^2)
                # = sqrt(-xi^2 + beta_s^2) = i * sqrt(xi^2 - beta_s^2)
                # The sign of v' flips across the cut.

                integrand_bc = self._branch_cut_integrand(kx_bc, ky, z)

                # Term: ky * int( Integrand * exp(...) )
                integral_bc = integrate.trapezoid(integrand_bc, x=kx_bc)  # dx is complex (-1j dxi)
                G_val += np.abs(ky) * integral_bc

        return G_val

    def _sdp_path(self, tau, theta, beta):
        """Eq. (24) and derivative."""
        # kx = -i*cos(theta)*(tau^2 + beta) + tau*sin(theta)*sqrt(tau^2 + 2*beta)
        sqrt_term = np.sqrt(tau**2 + 2 * beta)
        kx = -1j * np.cos(theta) * (tau**2 + beta) + tau * np.sin(theta) * sqrt_term

        # Derivative dkx/dtau
        term2 = np.sin(theta) * (sqrt_term + tau**2 / sqrt_term)
        dkx_dtau = -1j * np.cos(theta) * 2 * tau + term2
        return kx, dkx_dtau

    def _integrand_EF(self, kx_tilde, ky, wave_type, z):
        """Calculates E/F for Eq 10c."""
        # Un-normalized variables relations
        # kx = kx_tilde * ky
        # We work with normalized variables where possible, but E/F formulas in paper
        # are mixed. Let's compute normalized F and E/k^2 terms.

        # Normalized radicals
        v_tilde = np.sqrt(kx_tilde**2 + self.beta_p**2)
        vp_tilde = np.sqrt(kx_tilde**2 + self.beta_s**2)

        # Normalized F [cite: 109] divided by ky^4
        # F(k) = (2k^2 - ks^2)^2 - 4k^2 v v'
        # Norm: F_norm = (2*kx^2 + 2 - Ms^2)^2 - 4*(kx^2+1)*v*vp
        term1 = 2 * kx_tilde**2 + (2 - self.Ms**2)
        k_sq_norm = kx_tilde**2 + 1

        F_norm = term1**2 - 4 * k_sq_norm * v_tilde * vp_tilde

        # Coefficients [cite: 104]
        # For uz (alpha=z):
        # Az = (2k^2 - ks^2) / F  -> Normalized: term1 / F_norm
        # Cz = 2v / F             -> Normalized: 2*v_tilde / F_norm

        if wave_type == 'P':
            # Term: -v * Az * exp(-vz)
            # E_p_norm = -v_tilde * (term1 / F_norm) * exp(...)
            val = -v_tilde * (term1 / F_norm) * np.exp(-np.abs(ky) * z * v_tilde)
        else:
            # Term: k^2 * Cz * exp(-v'z)
            # E_s_norm = k_sq_norm * (2*v_tilde / F_norm) * exp(...)
            val = k_sq_norm * (2 * v_tilde / F_norm) * np.exp(-np.abs(ky) * z * vp_tilde)

        return val

    def _residue_terms(self, kx_tilde, ky, wave_type, z):
        """Calculates E and F' at the pole."""
        v_tilde = np.sqrt(kx_tilde**2 + self.beta_p**2)
        vp_tilde = np.sqrt(kx_tilde**2 + self.beta_s**2)

        term1 = 2 * kx_tilde**2 + (2 - self.Ms**2)
        k_sq_norm = kx_tilde**2 + 1

        # F_prime (Analytical Derivative)
        # d/dk [ (2k^2 + A)^2 - 4(k^2+1)v v' ]
        # A = 2 - Ms^2
        # d(term1^2) = 2 * term1 * 4k
        # d(4(k^2+1)vv') = 4 [ 2k vv' + (k^2+1)(k/v v' + v k/v') ]

        d_term1 = 2 * term1 * (4 * kx_tilde)
        d_radicals = 4 * (2 * kx_tilde * v_tilde * vp_tilde + k_sq_norm *
                          (kx_tilde * vp_tilde / v_tilde + kx_tilde * v_tilde / vp_tilde))

        F_prime_norm = d_term1 - d_radicals

        if wave_type == 'P':
            E_norm = -v_tilde * term1
        else:
            E_norm = k_sq_norm * 2 * v_tilde

        return E_norm, F_prime_norm

    def _branch_cut_integrand(self, kx_tilde, ky, z):
        """
        Calculates the difference of the integrand across the branch cut.
        Here v' changes sign.
        """
        # On the cut, let v' = i * |v'| vs -i * |v'|
        # We calculate the integrand for both signs and take difference.

        v_tilde = np.sqrt(kx_tilde**2 + self.beta_p**2)
        # Force standard branch first
        vp_tilde_main = np.sqrt(kx_tilde**2 + self.beta_s**2 + 0j)

        # Construct F and E with vp_tilde
        def get_EF(vp):
            term1 = 2 * kx_tilde**2 + (2 - self.Ms**2)
            k_sq_norm = kx_tilde**2 + 1
            F = term1**2 - 4 * k_sq_norm * v_tilde * vp
            # For Gp (Region III): E = -v * Az = -v * (2k^2 - ks^2)/F
            # E_numerator = -v_tilde * term1
            # We want E/F
            return (-v_tilde * term1) / F

        val_plus = get_EF(vp_tilde_main)
        val_minus = get_EF(-vp_tilde_main)

        # The integral formula includes exp(-ky(v z + i kx x))
        # This part is common.
        common_exp = np.exp(-np.abs(ky) *
                            (v_tilde * z)) * np.exp(-1j * np.abs(ky) * kx_tilde * 0)  # x=0 approx in exp for cut?
        # Actually Eq 31 says exp(-ky(v z + i kx x)).
        # But x depends on integration variable?
        # In Region III branch cut integral[cite: 310], the exponential is exp(-ky(v z + i kx x)).

        # Wait, the x coordinate is in the exponential.
        # Since we integrate over kx_tilde (which is purely imaginary), i*kx is Real.

        return (val_plus - val_minus) * np.exp(-np.abs(ky) * (v_tilde * z + 1j * kx_tilde * x))


# class MovingLoadElasticHalfSpace:
#     """
#     Implementation of the analytical solution for a moving load on an elastic half-space.
#     Based on :cite:`Fryba_2013` (Chapter 18).
#     The solution is only valid for subsonic load speeds (c < c2), and only computes vertical displacements.

#     """

#     def __init__(self, E: float, nu: float, rho: float, force: float, speed: float):
#         """
#         Initialize the elastic half-space model.

#         Args:
#             - E (float): Young's modulus (Pa)
#             - nu (float): Poisson's ratio (dimensionless)
#             - rho (float): Density (kg/m³)
#             - force (float): Magnitude of the concentrated force (N)
#             - speed (float): Speed of the moving load (m/s)
#         """
#         self.E = E
#         self.nu = nu
#         self.rho = rho
#         self.force = force
#         self.speed = speed

#         # Lamé parameters
#         self.lame_lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
#         self.G = E / (2 * (1 + nu))  # Shear modulus
#         self.M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))  # P-wave modulus

#         # Wave speeds
#         self.cp = np.sqrt(self.M / self.rho)  # P-wave speed
#         self.cs = np.sqrt(self.G / self.rho)  # S-wave speed
#         self.cr = self.Rayleigh_wave_speed()  # Rayleigh wave speed

#         if self.speed > self.cr:
#             raise ValueError("Error: The current implementation only supports subsonic speeds (c < cr).")

#         # Parameters
#         self.Gp = None  # Green's function for P-waves
#         self.Gs = None  # Green's function for S-waves
#         self.vertical_displacement = None  # Vertical displacement uz

#         # wave Mach number
#         self.Mp = self.speed / self.cp
#         self.Ms = self.speed / self.cs
#         # wave decay parameters
#         self.beta_p = np.sqrt(1 - self.Mp**2)
#         self.beta_s = np.sqrt(1 - self.Ms**2)
#         self.beta_r = np.sqrt(1 - (self.speed / self.cr)**2)

#         # Gaussian quadrature points and weights
#         self.n_gh = 128
#         self.gh_x, self.gh_w = hermgauss(self.n_gh)

#     @staticmethod
#     def __rayleigh_func(xi: float, eta: float) -> float:
#         """
#         Rayleigh function for computing the Rayleigh wave speed.

#         Args:
#             - xi (float): Dimensionless Rayleigh speed (c_r / c_s)
#             - eta (float): Ratio of shear wave speed to compressional wave speed (c_s / c_p)

#         Returns:
#             - (float): Value of the Rayleigh function
#         """
#         return xi**3 - 8 * xi**2 + 8 * xi * (3 - 2 * eta) - 16 * (1 - eta)

#     def Rayleigh_wave_speed(self):
#         """
#         Compute the Rayleigh wave speed c_r based on the material properties.

#         It solves Rayleigh equation: ξ³ - 8ξ² + 8ξ(3 - 2n) - 16(1 - η) = 0
#         See: https://en.wikipedia.org/wiki/Rayleigh_wave
#         """

#         eta = self.cs**2 / self.cp**2

#         # solve cubic equation for qsi
#         xi = brentq(self.__rayleigh_func, 0, 0.99, args=(eta, ))
#         return np.sqrt(xi) * self.cs

#     def compute_vertical_displacement(self,
#                                       x: float,
#                                       y: float,
#                                       z: float,
#                                       t: float,
#                                       ky_max: float = 10.,
#                                       n_tau=100,
#                                       n_ky=400) -> float:
#         """
#         Compute the vertical displacement at a given point and time.

#         It assumes that the poin load moves along the y-axis.

#         Args:
#             - x (float): x-coordinate of the observation point (m)
#             - y (float): y-coordinate of the observation point (m)
#             - z (float): z-coordinate of the observation point (m)
#             - t (float): time (s)
#         Returns:
#             - uz (float): vertical displacement at the observation point (m)
#         """

#         # Cylindrical coordinates
#         r = np.sqrt(x**2 + z**2)
#         theta = np.arctan2(z, x)

#         # Moving coordinate
#         y_prime = y - self.speed * t

#         # ky limits for integration
#         ky = np.linspace(-ky_max, ky_max, n_ky)
#         # tau limits for integration
#         # Decay factor is exp(-|ky| * r * tau**2)
#         # We need |ky|*r*tau**2 approx 10 for convergence
#         tau_max = np.sqrt(12 / (ky_max * r))
#         tau = np.linspace(-tau_max, tau_max, n_tau)

#         self.Gp = np.zeros_like(ky, dtype=complex)
#         self.Gs = np.zeros_like(ky, dtype=complex)

#         for i, ky_i in enumerate(ky):
#             self.Gp[i] = self.__Gp(theta, tau, ky_i, r, z)
#             self.Gs[i] = self.__Gs(theta, tau, ky_i, r, z)

#         # Synthesis of Fourier components Equation 13
#         integrand = (self.Gp + self.Gs) * np.exp(-1j * ky * y_prime)
#         I_ky = trapezoid(integrand, x=ky)

#         # scale factor Equation 10c
#         self.vertical_displacement = -self.force / (4 * np.pi**2 * self.G) * I_ky

#         return np.real(self.vertical_displacement)

#     def __Gp(self, theta: float, tau: Npt.NDArray, ky: float, r: float, z: float):

#         # compute angles for region classification (Fig 3)
#         theta_r = np.arccos(np.real(self.beta_r / self.beta_p))
#         theta_s = np.arccos(np.real(self.beta_s / self.beta_p))

#         if theta_r < theta < np.pi - theta_r:
#             region = 'I'
#         elif (0 <= theta < theta_s) or (np.pi - theta_s < theta <= np.pi):
#             region = 'III'
#         else:
#             region = 'II'

#         # Equation 24
#         kx_bar = -1j * np.cos(theta) * (tau**2 + self.beta_p) + tau * np.sin(theta) * np.sqrt(tau**2 + 2 * self.beta_p)

#         # Equation 25
#         diff_kx_bar = -1j * np.cos(theta) * 2 * tau + \
#             np.sin(theta) * (np.sqrt(tau**2 + 2*self.beta_p + 0j) +
#                              tau**2 * np.sin(theta) / np.sqrt(tau**2 + 2*self.beta_p))

#         # variables between Equation 5 and 6
#         kx = ky * kx_bar
#         k_sq = kx**2 + ky**2
#         kp = self.Mp * ky
#         ks = self.Ms * ky
#         v = np.sqrt(k_sq - kp**2)
#         v_prime = np.sqrt(k_sq - ks**2)

#         # compute Rayleigh denominator
#         F, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar, ks, v, v_prime)

#         # Equation 8
#         A = (2 * k_sq - ks**2) / F
#         # Equation 10c
#         E = -v * A * np.exp(-v * z)

#         # Equation 14
#         integral = (E / F) * diff_kx_bar * np.exp(-ky * r * tau**2)
#         # Integrate using trapezoidal rule
#         integral_sdp = trapezoid(integral, x=tau)
#         # Using Gaussian quadrature
#         # integral_sdp = np.sum(weight * (E / F) * diff_kx_bar)

#         # Equation 25
#         Gp = ky * np.exp(-ky * r * self.beta_p) * integral_sdp

#         # perform pole correction
#         correction = self.__pole_correction_p(region, theta, ky, E, F, F_prime, kx_bar, diff_kx_bar, v, z)

#         Gp += correction

#         return Gp

#     def __Gs(self, theta: float, tau: Npt.NDArray, ky: float, r: float, z: float):

#         # compute angles for region classification (Fig 5)
#         theta_r_star = np.arccos(np.real(self.beta_r / self.beta_s))

#         if theta_r_star < theta < np.pi - theta_r_star:
#             region = 'I'
#         else:
#             region = 'II'

#         if ky <= 0:
#             return 0

#         # a = r * ky
#         # t = self.gh_x / np.sqrt(a)
#         # weight = self.gh_w / np.sqrt(a)

#         # Equation 34
#         kx_bar = -1j * np.cos(theta) * (tau**2 + self.beta_s) + tau * np.sin(theta) * np.sqrt(tau**2 + 2 * self.beta_s)

#         # Equation 35
#         diff_kx_bar = -1j * np.cos(theta) * 2 * tau + \
#             np.sin(theta) * (np.sqrt(tau**2 + 2*self.beta_s) +
#                              tau**2 * np.sin(theta) / np.sqrt(tau**2 + 2*self.beta_s))

#         # variables between Equation 5 and 6
#         kx = ky * kx_bar
#         k_sq = kx**2 + ky**2
#         kp = self.Mp * ky
#         ks = self.Ms * ky
#         v = np.sqrt(k_sq - kp**2)
#         v_prime = np.sqrt(k_sq - ks**2)

#         # compute Rayleigh denominator
#         F, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar, ks, v, v_prime)

#         # Equation 7
#         C = 2 * v / F
#         # Equation 10c
#         E = k_sq * C * np.exp(-v_prime * z)

#         # Equation 14
#         integral = (E / F) * diff_kx_bar * np.exp(-ky * r * tau**2)

#         # Integrate using trapezoidal rule
#         integral_sdp = trapezoid(integral, x=tau)
#         # Using Gaussian quadrature
#         # integral_sdp = np.sum(weight * (E / F) * diff_kx_bar)

#         # Equation 25
#         Gs = ky * np.exp(-ky * r * self.beta_s) * integral_sdp

#         # perform pole correction
#         correction = self.__pole_correction_s(region, theta, ky, E, F_prime)

#         Gs += correction
#         return Gs

#     def __pole_correction_p(self, region: str, theta: float, ky: float, E: Npt.NDArray, F: Npt.NDArray,
#                             F_prime: Npt.NDArray, kx_bar: Npt.NDArray, diff_kx_bar: Npt.NDArray, v: Npt.NDArray,
#                             z: float) -> Npt.NDArray:
#         """
#         Perform pole correction for the Green's function.

#         Args:
#             - region (str): Region classification ('I', 'II', 'III')
#             - theta (float): Angle in cylindrical coordinates
#             - ky (Npt.NDArray): Array of ky values
#             - E (Npt.NDArray): Array of E values from Equation 8
#             - F (Npt.NDArray): Array of Rayleigh function values
#             - F_prime (Npt.NDArray): Array of derivatives of the Rayleigh function
#             - kx_bar (Npt.NDArray): Array of kx_bar values
#             - diff_kx_bar (Npt.NDArray): Array of derivatives of kx_bar
#             - v (Npt.NDArray): Array of v values
#             - x (float): x-coordinate of the observation point (m)
#             - z (float): z-coordinate of the observation point (m)

#         Returns:
#             - correction (Npt.NDArray): Pole correction values
#         """
#         if region == 'I':
#             return 0.0

#         # Pole location in normalized plane
#         kx_bar_pole = -1j * self.beta_r * np.cos(theta)

#         kx = ky * kx_bar_pole
#         k_sq = kx**2 + ky**2
#         kp = self.Mp * ky
#         ks = self.Ms * ky
#         v = np.sqrt(k_sq - kp**2)
#         v_prime = np.sqrt(k_sq - ks**2)

#         _, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar_pole, ks, v, v_prime)

#         # Equation 28
#         v_r_bar = np.sqrt(self.beta_p**2 - self.beta_r**2)
#         x = r * np.cos(theta)
#         exponent = -ky * (v_r_bar * z + self.beta_r * x * np.sign(np.cos(theta)))

#         correction = (-np.sign(np.cos(theta)) * 2 * np.pi * 1j * (E / F_prime) * np.exp(exponent))

#         if region == 'II':
#             return correction

#         if region == 'III':
#             # Equation 31
#             integral = E / F * np.exp(-ky * v * z + 1j * diff_kx_bar * x)
#             correction += trapezoid(integral, dx=kx_bar)
#             return correction

#     def __pole_correction_s(self, region: str, theta: float, ky: float, E: Npt.NDArray,
#                             F_prime: Npt.NDArray) -> Npt.NDArray:
#         """
#         Perform pole correction for the Green's function.

#         Args:
#             - region (str): Region classification ('I', 'II')
#             - theta (float): Angle in cylindrical coordinates
#             - ky (Npt.NDArray): Array of ky values
#             - E (Npt.NDArray): Array of E values from Equation 8
#             - F_prime (Npt.NDArray): Array of derivatives of the Rayleigh function

#         Returns:
#             - correction (Npt.NDArray): Pole correction values
#         """
#         if region == 'I':
#             return 0.0

#         # Pole location in normalized plane
#         kx_bar_pole = -1j * self.beta_s * np.cos(theta)

#         kx = ky * kx_bar_pole
#         k_sq = kx**2 + ky**2
#         kp = self.Mp * ky
#         ks = self.Ms * ky
#         v = np.sqrt(k_sq - kp**2)
#         v_prime = np.sqrt(k_sq - ks**2)

#         _, F_prime = self.__rayleigh_denominator(k_sq, kx, kx_bar_pole, ks, v, v_prime)

#         v_r_bar = np.sqrt(self.beta_s**2 - self.beta_r**2)
#         exponent = -ky * (v_r_bar * z + self.beta_r * x * np.sign(np.cos(theta)))
#         correction = (-np.sign(np.cos(theta)) * 2 * np.pi * 1j * (E / F_prime) * np.exp(exponent))
#         return correction

#     def __rayleigh_denominator(self, k_sq: Npt.NDArray, kx: Npt.NDArray, kx_bar: Npt.NDArray, ks: Npt.NDArray,
#                                v: Npt.NDArray, v_prime: Npt.NDArray) -> Npt.NDArray:
#         """
#         Compute Rayleigh function F(k) and their derivative F'(k).

#         Args:
#             - k_sq (Npt.NDArray): Array of k squared values
#             - ky (Npt.NDArray): Array of ky values
#             - ks (Npt.NDArray): Array of ks values
#             - kx_bar (Npt.NDArray): Array of kx_bar values
#             - v (Npt.NDArray): Array of v values
#             - v_prime (Npt.NDArray): Array of v_prime values

#         Returns:
#             - F (Npt.NDArray): Array of Rayleigh function values
#         """
#         # Equation 9
#         F = (2 * k_sq - ks**2)**2 - 4 * k_sq * v * v_prime

#         # Derivative of F with respect to kx_bar (Equation 29) - NOTE: this equation is wrong the the paper
#         # I fixed it with Maxima
#         F_prime = 2*(4*kx_bar+(2*self.Ms**2*kx**2)/kx_bar**3)*(2*kx_bar**2-(self.Ms**2*kx**2)/kx_bar**2) - \
#             8 * kx_bar * v * v_prime + 2 * self.Mp * kx * v_prime / v + 2 * self.Ms * kx * v / v_prime

#         return F, F_prime


def main():
    # Example usage
    E = 30e6  # Pa
    nu = 0.1  # dimensionless
    rho = 2000  # kg/m³
    force = -1e3  # N
    speed = 700  # m/s

    # 1. Defined Constants from Paper
    c_s_target = 1000.0  # m/s
    rho = 2000.0  # kg/m^3 (Arbitrary scaling factor, cancels out in dimensionless results)
    nu = 0.25  # Poisson's ratio

    # 2. Back-calculate Young's Modulus E to match cs = 1000 m/s
    # Formula: cs = sqrt( G / rho ) and G = E / (2*(1+nu))
    # Therefore: E = rho * cs^2 * 2 * (1 + nu)
    E = rho * (c_s_target**2) * 2 * (1 + nu)  # Result: 5e9 Pa

    x, y, z = 0.0, 0.0, 10
    time = np.linspace(-0.1, 0.1, num=100)

    model = MovingLoadElasticHalfSpace(E, nu, rho, force, speed)

    uz = []
    for t in time:
        model.compute_vertical_displacement(x, y, z, t, ky_max=10.0, n_ky=1000)
        uz.append(np.real(model.vertical_displacement))

    radius = np.sqrt(x**2 + z**2)
    shear_modulus = model.G

    tau = model.cs * time / radius
    dimensionless_displacement = np.array(uz) * shear_modulus * radius / model.force

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))

    ax[0].plot(time, uz)
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("Vertical displacement uz (m)")
    ax[0].grid()

    ax[1].plot(tau, dimensionless_displacement)
    ax[1].set_xlabel("Dimensionless time τ = c_s t / r")
    ax[1].set_ylabel("Dimensionless displacement U_z = uz G r / F")
    ax[1].grid()
    plt.savefig("moving.png")
    plt.show()


if __name__ == "__main__":
    main()
