import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
