"""
face_sorter.py
Backward-compatible entry point; equivalent to the ``facesorter`` console script.
"""

import sys

from facesorter.cli import main

if __name__ == "__main__":
    sys.exit(main())
