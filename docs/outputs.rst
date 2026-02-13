Outputs
=======

STEM can write results in the VTK format and JSON.
The VTK format is compatible with ParaView for visualization, and it is useful to inspect the full field of results
and make animations.
The JSON format is meant to store time history results for specific nodes in the model.


VTK output
----------
To generate VTK output, you need to define the output settings for the model.
This includes specifying the output parameters, such as the output interval,
the nodal and gauss point results to be written, and the output control type (step or time).

.. code-block:: python

   from stem.output import NodalOutput, VtkOutputParameters

   nodal = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
   gauss = [GaussPointOutput.CAUCHY_STRESS_VECTOR]
   vtk = VtkOutputParameters(
       output_interval=1,
       nodal_results=nodal,
       gauss_point_results=gauss,
       output_control_type="step"
   )

   model.add_output_settings(
       part_name="porous_computational_model_part",
       output_name="vtk_output",
       output_dir="output",
       output_parameters=vtk,
   )


JSON output
-----------
To generate JSON output, it is required to specify the nodes to record the results.
This includes specifying the output parameters, such as the output interval and the output control type (step or time).
The JSON output will contain the time history of the specified results for the nodes of interest.


.. code-block:: python

   from stem.output import JsonOutputParameters

    desired_output_points = [
                            (0.75, 10, 10),
                            (2.5, 10, 10),
                            ]

    model.add_output_settings_by_coordinates(
        part_name="subset_outputs",
        output_dir=output_dir,
        output_name="json_output",
        coordinates=desired_output_points,
        output_parameters=JsonOutputParameters(
            output_interval=time_step,
            nodal_results=nodal_results,
            gauss_point_results=gauss_point_results
        )
    )

Practical tips
--------------
- Choose output_control_type="step" to write every N steps; or "time" to write at time intervals.
- Keep output lists minimal for performance, especially at small time steps.
- ParaView: load the VTK series from the output directory and use filters to visualize displacements and accelerations.
- Large runs: prefer sparse intervals or JSON for targeted data extraction.
