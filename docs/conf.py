# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import stem

project = 'STEM'
copyright = '2023-2025, STEM team'
author = 'STEM team'
version = stem.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme', 'sphinx.ext.intersphinx', 'sphinxcontrib.bibtex']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

bibtex_bibfiles = ['refs.bib']

# to add __init__ documentation to the build
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# # Configure the RTD theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,  # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 2,
    'includehidden': True,
    'titles_only': True,
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False
# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True
# If true shows the source link to the rst code
html_show_sourcelink = False

# Link to the github
html_context = {
    "display_github": True,
    "github_user": "stemVibrations",
    "github_repo": "STEM",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
