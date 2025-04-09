import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'figaroh'
copyright = '2021-2025, Thanh Nguyen'
author = 'Thanh Nguyen'

# Extensions configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages'
]

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = 'https://thanhndv212.github.io/figaroh/'

# Disable Jekyll
html_extra_path = ['.nojekyll']

# Theme options
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
}

# Other Sphinx settings
nitpicky = True
nitpick_ignore = []

# GitHub Pages settings
html_context = {
    'display_github': True,
    'github_user': 'thanhndv212',
    'github_repo': 'figaroh',
    'github_version': 'main',
    'conf_py_path': '/docs/source/'
}
