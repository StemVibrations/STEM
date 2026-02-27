API reference
=============

The complete STEM API is generated automatically during every documentation build.
We invoke ``sphinx-apidoc`` (see ``docs/conf.py``) to create a dedicated page for
each Python module, so every public function appears under the file it lives in
along with its signature and docstring-derived description.

Explore the package tree below to drill into any module. The generated pages live
under ``docs/api/modules`` and will always reflect the current repository state.

.. toctree::
   :maxdepth: 2
   :caption: STEM package layout

   api/modules/modules