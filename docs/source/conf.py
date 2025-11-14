"""
Config dor sphinx documentation
"""
# -- Project information -----------------------------------------------------

project = "interFEBio"


extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_mock_imports = ["fastapi", "uvicorn", "prettytable"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
    "private-members": True,
}
