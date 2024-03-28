import numpy as np

GRAVITY = 9.81


class Cart():
    """
    Cart class

    Attributes:
        - bogies (list): list of bogies
        - dofs (list): list of degrees of freedom
    """

    def __init__(self):
        """
        Initialise the cart
        """
        self.bogies = []
        self.dofs = []


class Bogie():
    """
    Bogie class

    Attributes:
        - wheels (list): list of wheels
        - dofs (list): list of degrees of freedom
    """

    def __init__(self):
        """
        Initialise the bogie
        """
        self.wheels = []
        self.dofs = []


class Wheel():
    """
    Wheel class

    Attributes:
        - dofs (list): list of degrees of freedom
    """

    def __init__(self):
        self.dofs = []


class TrainModel():

    def __init__(self):

        self.n_carts = None
        self.n_bogies = None
        self.n_wheels = None

        self.cart_intertia = None
        self.cart_mass = None
        self.cart_stiffness = None
        self.cart_damping = None
        self.bogie_distances = None
        self.bogie_intertia = None
        self.bogie_mass = None
        self.wheel_intertia = None
        self.wheel_mass = None
        self.wheel_stiffness = None
        self.wheel_damping = None
        self.wheel_distances = None

        self.carts = []

        self.ndof = None
        self.contact_dofs = []

        self.contact_coeff = None
        self.contact_power = None

    def initialise(self):
        """
        Initialise the train system

        :return:
        """
        self.generate_components()
        self.set_dofs()

    def calculate_n_elements(self):
        """
        Calculate the number of bogies and wheels in the train system

        :return:
        """

        self.n_bogies = self.n_carts * len(self.bogie_distances)
        self.n_wheels = self.n_bogies * len(self.wheel_distances)

    def generate_components(self):
        """
        Generate the train components, i.e. wheels, bogies and carts

        :return:
        """

        self.calculate_n_elements()

        for i in range(self.n_carts):
            cart = Cart()
            for j in range(self.n_bogies):
                bogie = Bogie()
                for k in range(len(self.wheel_distances)):
                    wheel = Wheel()
                    bogie.wheels.append(wheel)
                cart.bogies.append(bogie)
            self.carts.append(cart)

    def set_dofs(self):
        """
        Set the degrees of freedom for each component and calculate the total number of degrees of freedom

        :return:
        """

        i_dof = 0

        for cart in self.carts:
            # cart has displacement and rotation dof
            cart.dofs = [i_dof, i_dof + 1]
            i_dof += 2

            for bogie in cart.bogies:
                # bogie has displacement and rotation dof
                bogie.dofs = [i_dof, i_dof + 1]
                i_dof += 2

                for wheel in bogie.wheels:
                    # wheel has displacement dof
                    wheel.dofs = [i_dof]
                    self.contact_dofs.append(i_dof)
                    i_dof += 1

        self.ndof = i_dof

    def generate_mass_matrix(self):
        """
        Generate the mass matrix
        """

        mass_matrix = np.zeros((self.ndof, self.ndof))

        for cart in self.carts:
            mass_matrix[cart.dofs[0], cart.dofs[0]] += self.cart_mass
            mass_matrix[cart.dofs[1], cart.dofs[1]] += self.cart_inertia

            for bogie in cart.bogies:
                mass_matrix[bogie.dofs[0], bogie.dofs[0]] += self.bogie_mass
                mass_matrix[bogie.dofs[1], bogie.dofs[1]] += self.bogie_inertia

                for wheel in bogie.wheels:
                    mass_matrix[wheel.dofs[0], wheel.dofs[0]] += self.wheel_mass

        return mass_matrix

    def generate_stiffness_matrix(self):
        """
        Generate stiffness matrix
        """
        stiffness_matrix = np.zeros((self.ndof, self.ndof))

        for cart in self.carts:
            stiffness_matrix[cart.dofs[0], cart.dofs[0]] += self.cart_stiffness * len(cart.bogies)

            for b, bogie in enumerate(cart.bogies):
                stiffness_matrix[cart.dofs[1], cart.dofs[1]] += self.cart_stiffness * self.bogie_distances[b]**2

                stiffness_matrix[cart.dofs[0], bogie.dofs[0]] += -self.cart_stiffness
                stiffness_matrix[bogie.dofs[0], cart.dofs[0]] += -self.cart_stiffness

                stiffness_matrix[cart.dofs[1], bogie.dofs[0]] += self.cart_stiffness * self.bogie_distances[b]
                stiffness_matrix[bogie.dofs[0], cart.dofs[1]] += self.cart_stiffness * self.bogie_distances[b]

                stiffness_matrix[bogie.dofs[0], bogie.dofs[0]] += self.cart_stiffness

                stiffness_matrix[bogie.dofs[0], bogie.dofs[0]] += self.wheel_stiffness * len(bogie.wheels)

                for w, wheel in enumerate(bogie.wheels):

                    stiffness_matrix[bogie.dofs[1], bogie.dofs[1]] += self.wheel_stiffness * self.wheel_distances[w]**2
                    stiffness_matrix[bogie.dofs[0], wheel.dofs[0]] += -self.wheel_stiffness
                    stiffness_matrix[wheel.dofs[0], bogie.dofs[0]] += -self.wheel_stiffness

                    stiffness_matrix[bogie.dofs[1], wheel.dofs[0]] += self.wheel_stiffness * self.wheel_distances[w]
                    stiffness_matrix[wheel.dofs[0], bogie.dofs[1]] += self.wheel_stiffness * self.wheel_distances[w]

                    stiffness_matrix[wheel.dofs[0], wheel.dofs[0]] += self.wheel_stiffness

        return stiffness_matrix

    def generate_damping_matrix(self):
        """
        Generate damping matrix

        :return:
        """

        damping_matrix = np.zeros((self.ndof, self.ndof))

        for cart in self.carts:
            damping_matrix[cart.dofs[0], cart.dofs[0]] += self.cart_damping * len(cart.bogies)

            for b, bogie in enumerate(cart.bogies):
                damping_matrix[cart.dofs[1], cart.dofs[1]] += self.cart_damping * self.bogie_distances[b]**2

                damping_matrix[cart.dofs[0], bogie.dofs[0]] += -self.cart_damping
                damping_matrix[bogie.dofs[0], cart.dofs[0]] += -self.cart_damping

                damping_matrix[cart.dofs[1], bogie.dofs[0]] += self.cart_damping * self.bogie_distances[b]
                damping_matrix[bogie.dofs[0], cart.dofs[1]] += self.cart_damping * self.bogie_distances[b]

                damping_matrix[bogie.dofs[0], bogie.dofs[0]] += self.cart_damping

                damping_matrix[bogie.dofs[0], bogie.dofs[0]] += self.wheel_damping * len(bogie.wheels)
                for w, wheel in enumerate(bogie.wheels):
                    damping_matrix[bogie.dofs[1], bogie.dofs[1]] += self.wheel_damping * self.wheel_distances[w]**2
                    damping_matrix[bogie.dofs[0], wheel.dofs[0]] += -self.wheel_damping
                    damping_matrix[wheel.dofs[0], bogie.dofs[0]] += -self.wheel_damping

                    damping_matrix[bogie.dofs[1], wheel.dofs[0]] += self.wheel_damping * self.wheel_distances[w]
                    damping_matrix[wheel.dofs[0], bogie.dofs[1]] += self.wheel_damping * self.wheel_distances[w]

                    damping_matrix[wheel.dofs[0], wheel.dofs[0]] += self.wheel_damping

        return damping_matrix

    def generate_force_vector(self):
        """
        Generate force vector
        """
        force_vector = np.zeros(self.ndof)

        for cart in self.carts:
            force_vector[cart.dofs[0]] = -self.cart_mass * GRAVITY

            for bogie in cart.bogies:
                force_vector[bogie.dofs[0]] = -self.bogie_mass * GRAVITY

                for wheel in bogie.wheels:
                    force_vector[wheel.dofs[0]] = -self.wheel_mass * GRAVITY

        return force_vector

    def calculate_static_contact_force(self):
        """
        Calculate static contact force

        :return:
        """

        contact_forces = []

        # Calculate static contact force
        for cart in self.carts:
            distributed_load_cart = self.cart_mass * -GRAVITY / len(cart.bogies)

            for bogie in cart.bogies:

                distributed_load_bogie = (self.bogie_mass * -GRAVITY + distributed_load_cart) / len(bogie.wheels)

                for _ in bogie.wheels:
                    contact_force = self.wheel_mass * -GRAVITY + distributed_load_bogie
                    contact_forces.append(contact_force)

        return np.array(contact_forces)

    def apply_dirichlet_bc(self, A, b, bc_indices):
        """
        Apply Dirichlet boundary conditions to a stiffness matrix and force vector.

        :param A: The stiffness matrix.
        :param b: The force vector.
        :param bc_indices: The indices at which to apply the boundary conditions.

        :returns: The modified stiffness matrix and force vector.
        """

        # Replace the i-th row of A with the unit vector.
        for index in bc_indices:
            # Replace the i-th row of A with the unit vector.
            A[index, :] = 0.0
            A[:, index] = 0.0
            A[index, index] = 1.0

            b[index] = 0.0

        return A, b

    def calculate_initial_displacement(self, K, F, u_wheels):
        """
        Calculate initial displacement

        :param K: stiffness matrix
        :param F: force vector
        :param u_wheels: initial displacement of the wheels
        :return:
        """

        # apply dirichlet boundary condition
        u = np.zeros(self.ndof)
        u[self.contact_dofs] = u_wheels

        F_internal = K.dot(u)
        F_total = F - F_internal

        K_constr, F_constr, = self.apply_dirichlet_bc(np.copy(K), np.copy(F_total), self.contact_dofs)

        # Solve the linear system.
        u = np.linalg.solve(K_constr, F_constr)

        # # add dirichlet boundary condition
        u[self.contact_dofs] = u_wheels

        return u

    def generate_global_matrices(self):
        """
        Generate global matrices

        and apply dirichlet boundary condition on elements without mass or inertia term
        :return:
        """

        K = self.generate_stiffness_matrix()
        C = self.generate_damping_matrix()
        M = self.generate_mass_matrix()
        F = self.generate_force_vector()

        # dirichlet constraint on elements without mass or intertia term
        tolerance = 1e-10
        mask = np.abs(np.diag(M)) < tolerance

        np.fill_diagonal(M, np.where(mask, 1, np.diag(M)))
        np.fill_diagonal(K, np.where(mask, 1, np.diag(K)))
        np.fill_diagonal(C, np.where(mask, 1, np.diag(C)))
        F[mask] = 0

        return K, C, M, F
