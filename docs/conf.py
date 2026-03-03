# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import stem

project = 'STEM'
copyright = '2023-2025, STEM team'
author = 'STEM team'
version = stem.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

bibtex_bibfiles = ['refs.bib']

# to add __init__ documentation to the build
autoclass_content = 'both'

# generate autosummary pages (stub files)
autosummary_generate = True

# mock heavy optional dependencies during autodoc to avoid import errors
autodoc_mock_imports = [
    'KratosMultiphysics',
    'gmsh_utils',
]

# intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', {}),
    'sphinx': ('https://www.sphinx-doc.org/en/master', {}),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# # Configure the RTD theme options
html_theme_options = {
    'logo_only': False,
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


def run_apidoc(app):
    """Generate up-to-date API stubs before every Sphinx build."""
    try:
        from sphinx.ext import apidoc
    except ImportError:
        return

    package_dir = PROJECT_ROOT / 'stem'
    if not package_dir.exists():
        return

    output_dir = DOCS_DIR / 'api' / 'modules'
    os.makedirs(output_dir, exist_ok=True)

    apidoc_args = [
        '--force',
        '--module-first',
        '--separate',
        '-o',
        str(output_dir),
        str(package_dir),
    ]
    apidoc.main(apidoc_args)


def setup(app):
    app.connect('builder-inited', run_apidoc)
