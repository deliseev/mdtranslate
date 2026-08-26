"""Позволяет запускать пакет как `python -m mdtranslate`."""

import sys

from .translator import main

if __name__ == "__main__":
    sys.exit(main())
