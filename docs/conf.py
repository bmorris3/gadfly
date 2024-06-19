# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime
import inspect
import sys
from os.path import relpath, dirname

# -- Project information -----------------------------------------------------

project = 'gadfly'
author = 'Brett M. Morris'
copyright = f"{datetime.datetime.now().year}, {author}"  # noqa: A001

# The full version, including alpha/beta/rc tags
import gadfly
from gadfly import __version__
release = __version__
dev = "dev" in release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx.ext.autosectionlabel',
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Treat everything in single ` as a Python reference.
default_role = 'py:obj'

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'celerite2': ('https://celerite2.readthedocs.io/en/latest/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'lightkurve': ('https://docs.lightkurve.org/', None),
    'matplotlib': ('http://matplotlib.org/stable', None),
    'tynt': ('https://tynt.readthedocs.io/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------

github_issues_url = 'https://github.com/bmorris3/gadfly/issues/'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_copy_source = False
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_logo = "assets/logo.png"
html_favicon = "assets/logo.ico"

html_theme_options = {
    "use_edit_page_button": True,
    "use_download_button": True,
    "repository_url": "https://github.com/bmorris3/gadfly",
    "github_url": "https://github.com/bmorris3/gadfly",
    "repository_branch": "main",
    "path_to_docs": "docs",
}

numpydoc_show_class_members = False
autodoc_inherit_docstrings = True

html_context = {
    "github_user": "bmorris3",
    "github_repo": "gadfly",
    "github_version": "main",
    "conf_py_path": "docs/",
    "default_mode": "light",
    "to_be_indexed": ["stable", "latest"],
    "is_development": dev,
    "doc_path": "docs",
    "display_github": True,
}

autosectionlabel_prefix_document = True
autoclass_content = 'both'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0}'.format(project)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Prefixes that are ignored for sorting the Python module index
modindex_common_prefix = ["gadfly."]

suppress_warnings = ['autosectionlabel.*']


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None
    lineno = None

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        # Ignore re-exports as their source files are not within the numpy repo
        module = inspect.getmodule(obj)
        if module is not None and not module.__name__.startswith("shone"):
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        fn = relpath(fn, start=dirname(gadfly.__file__))

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in gadfly.__version__:
        return "https://github.com/bmorris3/shone/blob/main/shone/%s%s" % (
           fn, linespec)
    else:
        return "https://github.com/bmorris3/shone/blob/v%s/shone/%s%s" % (
           gadfly.__version__, fn, linespec)


plot_formats = [('png', 250)]
