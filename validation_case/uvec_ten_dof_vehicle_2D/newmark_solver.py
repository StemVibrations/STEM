from typing import Tuple
import numpy as np


class NewmarkExplicit():
    """
    Newmark integration scheme for explicit time integration

    Attributes:
        - beta (float): beta parameter
        - gamma (float): gamma parameter
    """

    def __init__(self, beta: float = 0.25, gamma: float = 0.5):
        """
        Constructor of the NewmarkExplicit class

        Args:
            - beta (float): beta parameter
            - gamma (float): gamma parameter
        """
        self.beta = beta
        self.gamma = gamma

    def calculate(self, M: np.ndarray, C: np.ndarray, K: np.ndarray, F: np.ndarray, dt: float, t_index: int,
                  u_ini: np.ndarray, v_ini: np.ndarray, a_ini: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Newmark integration step.

        Args:
            - M (np.ndarray): mass matrix
            - C (np.ndarray): damping matrix
            - K (np.ndarray): stiffness matrix
            - F (np.ndarray): force vector
            - dt (float): time step
            - t_index (int): time index
            - u_ini (np.ndarray): initial displacement
            - v_ini (np.ndarray): initial velocity
            - a_ini (np.ndarray): initial acceleration

        Returns:
            - Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple containing the new displacement, velocity and acceleration
        """

        # calculate time step size

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial acceleration

        if t_index == 0:
            a_ini = np.linalg.solve(M, F - C.dot(v_ini) - K.dot(u_ini))

        #calculate newmark constants
        a0 = 1 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1 / (beta * dt)
        a3 = 1 / (2 * beta) - 1
        a4 = gamma / beta - 1
        a5 = dt / 2 * (gamma / beta - 2)
        # a5 = gamma/beta - 2*dt/2
        a6 = dt * (1 - gamma)
        a7 = gamma * dt

        # calculate effective stiffness matrix
        K_eff = K + a0 * M + a1 * C

        # calculate effective force vector
        m_part = M @ (a0 * u_ini + a2 * v_ini + a3 * a_ini)
        c_part = C @ (a1 * u_ini + a4 * v_ini + a5 * a_ini)
        force_eff = F + m_part + c_part

        # solve for displacement
        u = np.linalg.solve(K_eff, force_eff)

        # solve for acceleration and velocity
        a = a0 * (u - u_ini) - a2 * v_ini - a3 * a_ini
        v = v_ini + a6 * a_ini + a7 * a

        return u, v, a
