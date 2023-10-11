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

The class or function name should be clear and descriptive of the functionality it provides.

There should be a docstring at the beginning of the class or function describing its purpose and usage.
The docstring should be in the form of a triple-quoted string.

The class or function arguments must have a type annotation. The type annotation should be in the form of a comment after the argument name.
The class or function should have a return type annotation. The return type annotation should be in the form of a comment after the closing parenthesis of the arguments.

Please, avoid inheritance, and favour composition when writing your code.

An example of a class:

.. code-block::

   class KratosMaterialIO:
    """
    Class containing methods to write materials to Kratos

    Attributes:
        - ndim (int): number of dimensions of the mesh
    """

    def __init__(self, ndim: int, domain:str):
        """
        Constructor of KratosMaterialIO class

        Args:
            - ndim (int): number of dimensions of the mesh
        """
        self.ndim: int = ndim
        self.domain = domain

An example of a function:

.. code-block::

       def are_2d_coordinates_clockwise(coordinates: Sequence[Sequence[float]]) -> bool:
        """
        Checks if the 2D coordinates are given in clockwise order. If the signed area is positive, the coordinates
        are given in clockwise order.

        Args:
            - coordinates (Sequence[Sequence[float]]): coordinates of the points of a surface

        Returns:
            - bool: True if the coordinates are given in clockwise order, False otherwise.
        """

        # calculate signed area of polygon
        signed_area = 0.0
        for i in range(len(coordinates) - 1):
            signed_area += (coordinates[i + 1][0] - coordinates[i][0]) * (coordinates[i + 1][1] + coordinates[i][1])

        signed_area += (coordinates[0][0] - coordinates[-1][0]) * (coordinates[0][1] + coordinates[-1][1])

        # if signed area is positive, the coordinates are given in clockwise order
        return signed_area > 0.0





