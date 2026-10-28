"""Stage entry point: python -m experiments.fe_ls <stage>.

Stages and flags are defined in baseline.main; see the package docstring for the
campaign and the run commands. stage_rollout re-invokes this same entry point
once per leg, so the module path here and the one it builds must agree.
"""
from __future__ import annotations

from .baseline import main

if __name__ == "__main__":
    main()
