#!/usr/bin/env python3
"""Deprecated shim. The canonical CLI is `ferroload.cli` (installed as the
`ferroload` console command). Kept so `python python/ferroload_cli.py …` still
works once the package is installed.
"""
import sys

from ferroload.cli import main

if __name__ == "__main__":
    sys.exit(main())
