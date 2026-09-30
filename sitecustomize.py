"""Project-local Python startup hook.

Runtime source overlays are intentionally forbidden. The checked-in v10 modules
are the single source of truth and importing Python must never mutate them.
"""
