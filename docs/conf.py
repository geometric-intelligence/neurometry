"""Sphinx configuration file."""

project = "neurometry"
copyright = "2023, Geometric Intelligence Lab."
author = "GI Authors"

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
]

# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

# Configure nbsphinx for notebooks execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"

# To get a prompt similar to the Classic Notebook, use
nbsphinx_input_prompt = " In [%s]:"
nbsphinx_output_prompt = " Out [%s]:"

nbsphinx_allow_errors = True

templates_path = ["_templates"]

source_suffix = [".rst"]

master_doc = "index"

language = "en"

nbsphinx_prolog = (
    r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      <p>Notebook source code:
        <a class="reference external" href="https://github.com/geometric-intelligence/"""
    r"""neurometry/blob/main/{{ docname|e }}">{{ docname|e }}</a>
      </p>
    </div>

.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_theme = "pydata_sphinx_theme"
html_baseurl = "geometric-intelligence.github.io"
htmlhelp_basename = "geometric-intelligencedoc"
html_last_updated_fmt = "%c"

latex_elements = {}


latex_documents = [
    (
        master_doc,
        "neurometry.tex",
        "Neurometry Documentation",
        "geometric-intelligence",
        "manual",
    ),
]

man_pages = [(master_doc, "neurometry", "Neurometry Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "neurometry",
        "Neurometry Documentation",
        author,
        "neurometry",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]
