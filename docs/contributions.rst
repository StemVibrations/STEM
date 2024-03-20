Code contribution
=================

Steps for submitting your code
------------------------------

When contributing code follow this checklist:

    #. Fork the repository on GitHub.
    #. Create an issue with the desired feature or bug fix.
    #. Make your modifications or additions in a feature branch.
    #. Make changes and commit your changes using a descriptive commit message.
    #. Provide tests for your changes, and ensure they all pass.
    #. Provide documentation for your changes, in accordance with the style of the rest of the project (see :ref:`style_guide`).
    #. Create a pull request to STEM main branch. The STEM team will review and discuss your Pull Request with you.

For any questions, please get in contact with one of the members of :doc:`authors`.


.. _style_guide:

Code style guide
----------------
The additional features should follow the style of the STEM project.

We follow the PEP 8 style guide for Python code, with our custom modifications as defined in the
`Yapf file <../../.style.yapf>`_ and the `flake8 file <../../.flake8>`_. These files can be ran manually by using the
following command from the root directory of the project:

.. code-block::

    pre-commit run --all-files


The class or function name should be clear and descriptive of the functionality it provides.

There should be a docstring at the beginning of the class or function describing its purpose and usage.
The docstring should be in the form of a triple-quoted string.

The class or function must have a type annotation.
The class should specify the attributes and inheritance.
The function should specify the arguments, exceptions and returns (in this order).
The return type annotation should be in the form of a comment after the closing parenthesis of the arguments.

Please, avoid inheritance, and favour composition when writing your code.

An example of a class:

.. code-block::

    class ResidualConvergenceCriteria(ConvergenceCriteriaABC):
        """
        Class containing information about the residual convergence criteria

        Inheritance:
            - :class:`ConvergenceCriteriaABC`

        Attributes:
            - residual_relative_tolerance (float): The relative tolerance for the residual. Default value is 1e-4.
            - residual_absolute_tolerance (float): The absolute tolerance for the residual. Default value is 1e-9.
        """
        def __init__(self):
            """
            Constructor of the ResidualConvergenceCriteria class.
            """
            self.residual_relative_tolerance: float = 1e-4
            self.residual_absolute_tolerance: float = 1e-9


An example of a function:

.. code-block::

    def create_solver_settings_dictionary(self, model: Model, mesh_file_name: str, materials_file_name: str) -> Dict[str, Any]:
        """
        Creates a dictionary containing the solver settings.

        Args:
            - model (:class:`stem.model.Model`): The model object containing the solver data and model parts.
            - mesh_file_name (str): The name of the mesh file.
            - materials_file_name (str): The name of the materials parameters json file.

        Raises:
            - ValueError: if solver_settings in model are not initialised.

        Returns:
            - Dict[str, Any]: dictionary containing the part of the project parameters
                dictionary related to problem data and solver settings.
        """

        if model.project_parameters is None:
            raise ValueError("Solver settings are not initialised in model.")

        return self.solver_io.create_settings_dictionary(
            model.project_parameters,
            Path(mesh_file_name).stem,
            materials_file_name,
            model.get_all_model_parts(),
        )
