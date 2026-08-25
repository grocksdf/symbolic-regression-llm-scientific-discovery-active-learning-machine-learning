"""Run the frozen P3G.3 dimension-stable function-prior real-data gate."""

from pathlib import Path

from scripts.run_pcpi_p3g1_structurewise_discrepancy_gate import main


if __name__ == "__main__":
    raise SystemExit(
        main(Path("configs/p3g_3_dimension_stable_function_prior_gate.json"))
    )
