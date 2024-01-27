Interface definitions
=====================

.. _uvec:

User-defined vehicle model
--------------------------
STEM support the definition of user-defined vehicle models. To define the vehicle model the user needs to create an *Python UVEC file*.
The UVEC file is a Python script that defines the vehicle model. The UVEC file must contain a function called `uvec` that has the following structure:

.. code-block:: python

    import json

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
        parameters = uvec_data["parameters"]

        # compute the loads at the wheels

        uvec_data['loads'] = {1: [0, load_wheel_1, 0], 2: [0, load_wheel_2, 0]}

        return json.dumps(uvec_data)

In this example the uvec_data contains the structure that it is used in STEM. The structure of the uvec_data is the following:

    * *uvec_data["u"]* - the displacement of the vehicle at the location of the wheel
    * *uvec_data["theta"]* - the rotation of the vehicle at the location of the wheel
    * *uvec_data["time_index"]* - the time index of the analysis
    * *uvec_data["dt"]* - the time step of the integration step
    * *uvec_data["state"]* - the state of the vehicle model (can be any JSON serialisable data structure). This can be used to pass information regarding the degrees of freedom of the vehicle model or any other information needed to compute the loads at the wheels
    * *uvec_data["parameters"]* - the parameters of the vehicle model (can be any JSON serialisable data structure). This can be used to pass information regarding the vehicle model, such as stiffness, mass and damping parameters
    * *uvec_data["loads"]* - the loads at the wheels

An example of the usage of a UVEC file can be found in the :ref:`tutorial3`.
For further information about the UVEC please refer to `Vehicle Models <https://github.com/StemVibrations/vehicle_models>`_.


.. _umat:

User-defined soil models
------------------------
STEM supports two types of user-defined soil models:

#. User-defined mechanical material behaviour (UMAT):

    UMAT is a format defined by `Abaqus <https://www.simuleon.com/simulia-abaqus/>`_.
    Examples of UMAT material models can be found in `Soil Models <https://soilmodels.com>`_.

#. User-defined soil model (UDSM):

    UDSM is a format defined by `Plaxis <https://www.bentley.com/software/plaxis-3d/>`_.
    More information about UDSM material models can be found in `here <https://communities.bentley.com/products/geotech-analysis/w/wiki/45468/creating-user-defined-soil-models>`_.