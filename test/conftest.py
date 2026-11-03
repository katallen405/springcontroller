import os
import sys

# Test this source tree, not a springcontroller already installed in
# install/ (a sourced workspace puts that on PYTHONPATH ahead of the CWD).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "springcontroller_ui"))
sys.path.insert(0, os.path.join(_ROOT, "springcontroller"))
