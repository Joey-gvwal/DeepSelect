"""CLI entry point: ``python -m deep_select.compat``.

Prints the runtime/build compatibility report for prebuilt `deepselect-musa`
wheels (see `deep_select._compat`).
"""
from ._compat import main

if __name__ == "__main__":
    raise SystemExit(main())
