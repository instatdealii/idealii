# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ideal.II'
copyright = '2024, the ideal.II authors'
author = 'the ideal.II authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.video",
    "breathe",
    "sphinx_rtd_theme",
    "sphinx_rtd_dark_mode",
    "cpp_example_directives",
    "maths_admonition_directives"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

highlight_language = 'c++'

autosectionlabel_prefix_document = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,

    # Header options
    'style_nav_header_background': '#450d54',
    'logo_only': True,

    # TOC options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
html_static_path = ['_static']
html_extra_path = ['doxygenhtml']

html_logo = "_static/idealii_logo_text.png"
github_url = 'www.github.com/instatdealii/idealii'
# pygments_style = 'github-dark'
pygments_style = 'monokai'

# -- Breathe config -------------------------------------

breathe_projects = {
    "ideal.II": "_build/doxygenxml/"
}

breathe_default_project = "ideal.II"
breathe_default_members = ('members','undoc-members')

import sys 
import os 

sys.path.append(os.path.abspath("./_ext"))