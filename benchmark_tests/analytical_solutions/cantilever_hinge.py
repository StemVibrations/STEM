import numpy as np


def cantilever_hinge(x: np.ndarray, L: float, E: float, I: float, k: float, V: float) -> np.ndarray:
    r"""
    Calculate the displacement of a cantilever beam with a hinge at the end, following the deflection curve.


    /|                   ! V
    /|-------------------o
    /|

     |--> x

    Args:
        - x: np.ndarray: The discretisation of the beam
        - L: float: The length of the beam
        - E: float: The Young's modulus
        - I: float: The moment of inertia
        - k: float: The stiffness of the hinge
        - V: float: The vertical force applied at the hinge

    Returns:
        - np.ndarray: The displacement at each x value
    """


    # Calculate the constants
    B=-((V * k * L**2) / (4 * E * I * k * L +4 * E * I**2)) - (2 * E * I * V * L)/(4 * E * I * k * L + 4 * E * I**2)
    A = V / (6 * E * I)

    # displacement
    u = A * x**3 + B * x**2
    return u


if __name__ == "__main__":
    L = 20
    E = 200
    I = 1
    k = 20000
    V = 20
    x = np.linspace(0, L, 100)
    disp = cantilever_hinge(x, L, E, I, k, V)
    import matplotlib.pyplot as plt
    plt.plot(x, disp)
    plt.show()


