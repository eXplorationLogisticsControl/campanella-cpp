"""pytest configuration: make the built module and helpers.py importable"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import helpers  # noqa: E402,F401  (puts the built module on sys.path)
