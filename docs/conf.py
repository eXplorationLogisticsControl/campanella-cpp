# Configuration file for the Sphinx documentation builder.
# Layout modelled on heyoka's documentation (sphinx-book-theme), with MyST Markdown sources.

project = "campanella"
copyright = "2026, Yuri Shimane"
author = "Yuri Shimane"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

source_suffix = {".md": "markdown"}
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
]
# auto-generate anchors for headings down to level 4, so that [text](page.md#heading) links resolve
myst_heading_anchors = 4

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "sphinx_book_theme"
html_title = "campanella"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/eXplorationLogisticsControl/campanella-cpp",
    "use_repository_button": True,
    "use_issues_button": True,
    "navigation_with_keys": False,
}
