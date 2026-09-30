#!/usr/bin/env python3
"""Convenience entry point for the auditable three-condition paired runner."""

from scripts.repro_core.run_llm_srbench_paired_v64 import main


if __name__ == "__main__":
    raise SystemExit(main())
