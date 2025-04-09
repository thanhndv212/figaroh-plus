import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'figaroh'
copyright = '2021-2025, Thanh Nguyen'
author = 'Thanh Nguyen'
version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = []

# GitHub Pages config
html_baseurl = 'https://thanhndv212.github.io/figaroh/'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
}

html_static_path = ['_static']
