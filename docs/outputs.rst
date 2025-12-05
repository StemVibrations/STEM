Outputs
=======

STEM can write results via Kratos output processes. Common formats are VTK (for ParaView),
GiD (post files) and JSON (for data extraction). Configure outputs through :mod:`stem.output`.

VTK output (ParaView)
---------------------
.. code-block:: python

   from stem.output import NodalOutput, VtkOutputParameters

   nodal = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
   vtk = VtkOutputParameters(
       output_interval=1,          # write every step (see control type below)
       nodal_results=nodal,
       gauss_point_results=[],
       output_control_type="step"  # or "time"
   )

   model.add_output_settings(
       part_name="porous_computational_model_part",  # as used by Kratos in the analysis
       output_name="vtk_output",
       output_dir="output",                          # folder where .vtk files are written
       output_parameters=vtk,
   )

GiD output
----------
.. code-block:: python

   from stem.output import GiDOutputParameters

   gid = GiDOutputParameters(
       output_interval=0.01,            # depends on output_control_type
       result_file_configuration={
           "gidpost_flags": {
               "GiDPostMode": "GiD_PostBinary",
               "WriteDeformedMeshFlag": "WriteDeformed",
               "WriteConditionsFlag": "WriteElementsOnly",
               "MultiFileFlag": "MultipleFiles",
           }
       },
       output_control_type="time"
   )
   model.add_output_settings("porous_computational_model_part", "gid_output", "gid_output", gid)

JSON output
-----------
.. code-block:: python

   from stem.output import JsonOutputParameters

   json_params = JsonOutputParameters(output_interval=1, output_control_type="step")
   model.add_output_settings("porous_computational_model_part", "json_output", "json", json_params)

Tips
----
- Choose output_control_type="step" to write every N steps; or "time" to write at time intervals.
- Keep output lists minimal for performance, especially at small time steps.
- ParaView: load the VTK series from the output directory and use filters to visualize displacements and accelerations.
- Large runs: prefer sparse intervals or JSON for targeted data extraction.
