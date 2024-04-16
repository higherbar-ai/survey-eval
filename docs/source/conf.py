# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Custom code (starting from https://medium.com/@djnrrd/automatic-documentation-with-pycharm-70d37927df57)

import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'surveyeval'
copyright = '2024, Higher Bar AI, PBC'
author = 'Higher Bar AI, PBC'
release = '0.1.4'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Solution for documenting class constructors
# from https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method

def skip_member(app, what, name, obj, would_skip, options):
    if name == "__init__":
        # don't skip documentation for constructors!
        return False

    return would_skip

def setup(app):
    # give us a say in which members are skipped by Sphinx autodoc
    app.connect("autodoc-skip-member", skip_member)
