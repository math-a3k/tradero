# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import django

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext"))
)

os.environ["DJANGO_SETTINGS_MODULE"] = "tradero.settings"

django.setup()

project = "tradero"
copyright = "2023, Rodrigo Gadea"
author = "Rodrigo Gadea"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "djangodocs",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_theme_options = {
    "github_user": "math-a3k",
    "github_repo": "tradero",
    # 'travis_button': True,
    # 'codecov_button': True,
    "show_related": True,
    "show_relbars": True,
}

html_static_path = ["_static"]

# sphinx-intl
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional.
