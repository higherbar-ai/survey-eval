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
copyright = '2025, Higher Bar AI, PBC'
author = 'Higher Bar AI, PBC'
release = '0.1.32'


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

_current_autodoc_module = None

def track_module(app, what, name, obj, options, lines):
    """Track which module autodoc is currently processing."""
    global _current_autodoc_module
    if what == 'module':
        _current_autodoc_module = name

def skip_member(app, what, name, obj, would_skip, options):
    if name == "__init__":
        # don't skip documentation for constructors!
        return False

    # when documenting a module's members, skip any that are imported from submodules
    # (they will be documented in their own submodule pages instead)
    if (what == "module" and _current_autodoc_module
            and hasattr(obj, "__module__") and obj.__module__
            and obj.__module__ != _current_autodoc_module
            and obj.__module__.startswith(_current_autodoc_module + ".")):
        return True

    return would_skip

def setup(app):
    # track current module so skip_member can detect re-exported submodule members
    app.connect("autodoc-process-docstring", track_module)
    # give us a say in which members are skipped by Sphinx autodoc
    app.connect("autodoc-skip-member", skip_member)
