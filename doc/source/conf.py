# -*- coding: utf-8 -*-
#
# SunPy documentation build configuration file, created by
# sphinx-quickstart on Sat Apr  9 13:09:06 2011.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os, math

class Mock(object):
    __all__ = []
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return Mock()

    def __iter__(self):
        return iter([Mock()])

    __add__  = __mul__  = __getitem__ = __setitem__ = \
__delitem__ = __sub__ =  __floordiv__ = __mod__ = __divmod__ = \
__pow__ = __lshift__ = __rshift__ = __and__ = __xor__ = __or__ = \
__rmul__  = __rsub__  = __rfloordiv__ = __rmod__ = __rdivmod__ = \
__rpow__ = __rlshift__ = __rrshift__ = __rand__ = __rxor__ = __ror__ = \
__imul__  = __isub__  = __ifloordiv__ = __imod__ = __idivmod__ = \
__ipow__ = __ilshift__ = __irshift__ = __iand__ = __ixor__ = __ior__ = \
__div__ = __rdiv__ = __idiv__ = __truediv__ = __rtruediv__ = __itruediv__ = \
__neg__ = __pos__ = __abs__ = __invert__ = __call__

    def __getattr__(self, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        # This clause is commented out because it makes an assumption with
        # case convention that is not necessarily true
        #elif name[0] != '_' and name[0] == name[0].upper():
        #    return type(name, (), {})
        else:
            return Mock(**vars(self))

    def __lt__(self, *args, **kwargs):
        return True

    __nonzero__ = __le__ = __eq__ = __ne__ = __gt__ = __ge__ = __contains__ = \
__lt__


    def __repr__(self):
        # Use _mock_repr to fake the __repr__ call
        res = getattr(self, "_mock_repr")
        return res if isinstance(res, str) else "Mock"

    def __hash__(self):
        return 1

    __len__ = __int__ = __long__ = __index__ = __hash__

    def __oct__(self):
        return '01'

    def __hex__(self):
        return '0x1'

    def __float__(self):
        return 0.1

    def __complex__(self):
        return 1j


MOCK_MODULES = [
    'scipy', 'matplotlib', 'matplotlib.pyplot', 'pyfits',
    'scipy.constants.constants', 'matplotlib.cm',
    'matplotlib.image', 'matplotlib.colors',
    'pandas', 'pandas.io', 'pandas.io.parsers',
    'suds', 'matplotlib.ticker', 'matplotlib.colorbar',
    'matplotlib.dates', 'scipy.optimize', 'scipy.ndimage',
    'matplotlib.figure', 'scipy.ndimage.interpolation', 'bs4',
    'scipy.interpolate',
    'matplotlib.cbook','matplotlib.axes','matplotlib.transforms',
    'matplotlib.gridspec','matplotlib.artist','matplotlib.axis',
    'matplotlib.collections','matplotlib.contour','matplotlib.path',
    'matplotlib.patches','matplotlib.animation','matplotlib.widgets',
    'mpl_toolkits','mpl_toolkits.axes_grid1',
    'mpl_toolkits.axes_grid1.axes_size',

    # The following lines are for sunpy.gui, which is a mess
    #'PyQt4','PyQt4.QtCore','PyQt4.QtGui',
    #'matplotlib.backends.backend_qt4agg',
    'sunpy.gui.ui.mainwindow.widgets.figure_canvas',
    'sunpy.gui.ui.mainwindow.widgets.toolbars',
    'sunpy.gui.ui.mainwindow.resources',

    'scipy.constants',

    'astropy', 'astropy.units', 'astropy.io', 'astropy.constants']

if not tags.has('doctest'):
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = Mock(pi=math.pi, G=6.67364e-11)

    # We want np.dtype() to return a special Mock class because it shows up as a
    # default value for arguments (see sunpy.spectra.spectrogram)
    sys.modules['numpy'] = Mock(pi=math.pi, G=6.67364e-11,
                                ndarray=type('ndarray', (), {}),
                                dtype=lambda _: Mock(_mock_repr='np.dtype(\'float32\')'))
else:
    import matplotlib
    matplotlib.interactive(True)
    exclude_patterns = ["reference/*"]

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../sunpy/'))


# -- General configuration -----------------------------------------------------

# Note: numpydoc extension is required and can be found at
# is available with the numpy source code
# https://github.com/numpy/numpy/tree/master/doc/sphinxext

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'numpydoc', 'sphinx.ext.todo',
              'sphinx.ext.pngmath', 'sphinx.ext.viewcode', 
              'sphinx.ext.autosummary', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'SunPy'
copyright = u'2013, SunPy Community' #pylint: disable=W0622

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.3'
# The full version, including alpha/beta/rc tags.
release = '0.3.2'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['sunpy.']


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'default'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '../logo/sunpy_logo_compact_192x239.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'SunPydoc'

# mapping from unique project names to (target, inventory) tuples to use the
# sphinx extension sphinx.ext.intersphinx to link to 3rdparty Sphinx projects
intersphinx_mapping = {'sqlalchemy': ('http://docs.sqlalchemy.org/en/rel_0_8/', None)}

# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'SunPy.tex', u'SunPy Documentation',
   u'SunPy Community', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

# Autosummary
import glob
autosummary_generate = (glob.glob("reference/*.rst") +
                        glob.glob("reference/*/*.rst"))

# Uncomment this to stop numpydoc from autolisting class members, which
# generates a ridiculous number of warnings.
#numpydoc_show_class_members = False
