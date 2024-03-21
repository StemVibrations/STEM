import json
import numpy as np
from typing import Tuple

from uvec_ten_dof_vehicle_2D.base_model import TrainModel
from uvec_ten_dof_vehicle_2D.hertzian_contact import HertzianContact
from uvec_ten_dof_vehicle_2D.newmark_solver import NewmarkExplicit


def uvec(json_string: str) -> str:
    """
    Args:
        - json_string (str): json string containing the uvec data

    Returns:
        - str: json string containing the load data
    """

    # Get the uvec data
    uvec_data = json.loads(json_string)

    # load the data
    u = uvec_data["u"]
    theta = uvec_data["theta"]
    time_index = uvec_data["time_index"]
    time_step = uvec_data["dt"]
    state = uvec_data["state"]

    # Get the uvec parameters
    parameters = uvec_data["parameters"]

    # initialise the train system
    (M, C, K, F_train), train = initialise(time_index, parameters, state)

    # calculate norm of u vector, gravity is downwards
    gravity_axis = parameters["gravity_axis"]

    u_vertical = [u[uw][gravity_axis] for uw in u.keys()]

    # calculate static displacement
    u_static = train.calculate_initial_displacement(K, F_train, u_vertical)

    if time_index <= 0:
        state["u"] = u_static
        state["v"] = np.zeros_like(u_static)
        state["a"] = np.zeros_like(u_static)

    state["u"] = np.array(state["u"])
    state["v"] = np.array(state["v"])
    state["a"] = np.array(state["a"])

    # calculate contact forces
    F_contact = calculate_contact_forces(u_vertical, train.calculate_static_contact_force(), state, parameters, train,
                                         time_index)

    # calculate force vector
    F = F_train
    F[train.contact_dofs] = F[train.contact_dofs] + F_contact

    # calculate new state
    u_train, v_train, a_train = calculate(state, (M, C, K, F), time_step, time_index)

    state["u"] = u_train.tolist()
    state["v"] = v_train.tolist()
    state["a"] = a_train.tolist()

    # calculate unit vector
    aux = {}
    for i, val in enumerate(F_contact):
        aux[i + 1] = [0., (-val).tolist(), 0.]
    uvec_data["loads"] = aux

    return json.dumps(uvec_data)


def initialise(time_index: int, parameters: dict, state: dict) -> \
                                    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TrainModel]:
    """
    Initialise the train system

    Args:
        - time_index (int): time index
        - parameters (dict): dictionary containing the parameters
        - state (dict): dictionary containing the state

    Returns:
        - Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TrainModel]: tuple containing the global matrices (M, C, K, F) and the train model
    """

    train = TrainModel()

    train.n_carts = parameters["n_carts"]
    train.cart_inertia = parameters["cart_inertia"]
    train.cart_mass = parameters["cart_mass"]
    train.cart_stiffness = parameters["cart_stiffness"]
    train.cart_damping = parameters["cart_damping"]

    train.bogie_distances = parameters["bogie_distances"]

    train.bogie_inertia = parameters["bogie_inertia"]
    train.bogie_mass = parameters["bogie_mass"]
    train.wheel_distances = parameters["wheel_distances"]

    train.wheel_mass = parameters["wheel_mass"]
    train.wheel_stiffness = parameters["wheel_stiffness"]
    train.wheel_damping = parameters["wheel_damping"]

    train.initialise()

    # set global matrices
    K, C, M, F = train.generate_global_matrices()

    return (M, C, K, F), train


def calculate_contact_forces(u: np.ndarray, F_static: np.ndarray, state: dict, parameters: dict, train: TrainModel,
                             time_index: int) -> np.ndarray:
    """
    Calculate the contact forces

    Args:
        - u (np.ndarray): vertical displacement of the wheels
        - F_static (np.ndarray): static contact force
        - state (dict): dictionary containing the state
        - parameters (dict): dictionary containing the parameters
        - train (TrainModel): train model
        - time_index (int): time index

    Returns:
        - np.ndarray: array containing the contact forces
    """

    contact_method = HertzianContact()
    contact_method.contact_coeff = parameters["contact_coefficient"]
    contact_method.contact_power = parameters["contact_power"]

    u_wheel = np.array(state["u"])[train.contact_dofs]

    static_contact_u = contact_method.calculate_contact_deformation(F_static)

    du = u_wheel + static_contact_u - u

    return contact_method.calculate_contact_force(du)


def calculate(state: dict, matrices: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], time_step, t) -> \
                                                                        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the new state

    Args:
        - state (dict): dictionary containing the state
        - matrices (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): tuple containing the global matrices
        - time_step (float): time step
        - t (int): time index

    Returns:
        - tuple: tuple containing the new state
    """

    (M, C, K, F) = matrices
    (u, v, a) = state["u"], state["v"], state["a"]

    solver = NewmarkExplicit()
    return solver.calculate(M, C, K, F, time_step, t, u, v, a)
