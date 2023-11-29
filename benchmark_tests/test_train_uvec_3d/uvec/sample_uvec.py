import numpy as np
import json
from typing import Dict, Tuple

from uvec.base_model import TrainModel
from uvec.hertzian_contact import HertzianContact
from uvec.newmark_solver import NewmarkExplicit


def uvec(uvec_data: Dict) -> Dict:
    """
    Calculate the force vector of the train.

    Args:
        - uvec_data (Dict): json string containing the data to calculate the unit vector

    Returns:
        - Dict: json string containing the force vector
    """

    # load json
    uvec_data = json.loads(uvec_data)

    # load data
    u = uvec_data["u"]
    theta = uvec_data["theta"]
    time_step = uvec_data["dt"]
    t = uvec_data["t"]
    time_index = uvec_data["time_index"]
    loads = uvec_data["loads"]
    parameters = uvec_data["parameters"]
    state = uvec_data["state"]

    # initialise the train system
    (M, C, K, F_train), train = initialise(time_index, parameters)

    # calculate norm of u vector, gravity is downwards
    gravity_axis = parameters["gravity_axis"]

    u_vertical_wheel = [u[uw][gravity_axis] for uw in u.keys()]

    # calculate static displacement
    u_static = train.calculate_initial_displacement(K, F_train, u_vertical_wheel)

    if time_index <= 0:
        state["u"] = u_static
        state["v"] = np.zeros_like(u_static)
        state["a"] = np.zeros_like(u_static)

    # update state
    state["u"] = np.array(state["u"])
    state["v"] = np.array(state["v"])
    state["a"] = np.array(state["a"])

    # calculate contact forces
    F_contact = calculate_contact_forces(u_vertical_wheel, train.calculate_static_contact_force(),
                                         state, parameters, train)

    # calculate force vector
    F = F_train
    F[train.contact_dofs] = F[train.contact_dofs] + F_contact

    # calculate new state
    u_train, v_train, a_train = calculate(state, (M, C, K, F), time_step, time_index)

    # update state
    state["u"] = u_train.tolist()
    state["v"] = v_train.tolist()
    state["a"] = a_train.tolist()
    uvec_data["state"] = state

    # calculate unit vector
    aux = {}
    for i in range(len(F_contact)):
        aux[i + 1] = [0., (-F_contact[i]).tolist(), 0.]
    uvec_data["loads"] = aux

    return json.dumps(uvec_data)


def initialise(time_index: int, parameters: Dict):
    """
    Initialise the train system

    Args:
        - time_index (int): current time index
        - parameters (Dict): json dictionary containing the parameters to initialise the train system

    Returns:
        - Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TrainModel]: tuple containing the global matrices and the train model
    """

    # initialise gravity loading on the train in loading steps
    if time_index <= parameters["initialisation_steps"]:
        gravity = 9.81  * time_index / parameters["initialisation_steps"]
    else:
        gravity = 9.81

    # initialise & build train model
    train = TrainModel(gravity)
    train.n_carts = parameters["n_carts"]
    train.cart_intertia = parameters["cart_inertia"]
    train.cart_mass = parameters["cart_mass"]
    train.cart_stiffness = parameters["cart_stiffness"]
    train.cart_damping = parameters["cart_damping"]

    train.bogie_distances = parameters["bogie_distances"]

    train.bogie_intertia = parameters["bogie_inertia"]
    train.bogie_mass = parameters["bogie_mass"]
    train.wheel_distances = parameters["wheel_distances"]

    train.wheel_mass = parameters["wheel_mass"]
    train.wheel_stiffness = parameters["wheel_stiffness"]
    train.wheel_damping = parameters["wheel_damping"]

    train.initialise()

    # set global matrices
    K, C, M, F = train.generate_global_matrices()

    return (M, C, K, F), train


def calculate_contact_forces(u: np.ndarray, F_static: np.ndarray, state: Dict, parameters: Dict, train: TrainModel) -> np.ndarray:
    """
    Calculate the contact forces

    Args:
        - u (np.ndarray): vertical displacement of the wheels
        - F_static (np.ndarray): static contact force
        - state (Dict): json dictionary containing the state of the train system
        - parameters (Dict): json dictionary containing the parameters to calculate the contact forces
        - train (TrainModel): train model

    Returns:
        - np.ndarray: contact forces
    """


    # initialise contact method: Hertzian contact
    contact_method = HertzianContact()
    contact_method.contact_coeff = parameters["contact_coefficient"]
    contact_method.contact_power = parameters["contact_power"]

    u_wheel = state["u"][train.contact_dofs]

    static_contact_u = contact_method.calculate_contact_deformation(F_static)

    # compute contact forces
    du = u_wheel + static_contact_u - u

    return contact_method.calculate_contact_force(du)


def calculate(state: Dict, matrices: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              time_step: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the new state of the train system

    Args:
        - state (Dict): json dictionary containing the state of the train system
        - matrices (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): tuple containing the global matrices
        - time_step (float): time step
        - t (float): current time

    Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple containing the new state of the train system
    """
    (M, C, K, F) = matrices
    (u, v, a) = state["u"], state["v"], state["a"]

    solver = NewmarkExplicit()
    return solver.calculate(M, C, K, F, time_step, t, u, v, a)
