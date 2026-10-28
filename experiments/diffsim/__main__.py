"""Stage entry point: python -m experiments.diffsim <stage>.

Stages and flags are defined in identify.main; see the package docstring for the
campaign and the run commands.
"""
from __future__ import annotations

from .identify import main

if __name__ == "__main__":
    main()
