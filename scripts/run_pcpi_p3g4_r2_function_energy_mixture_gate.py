"""Run the frozen P3G.4 R-squared function-energy mixture real-data gate."""

from pathlib import Path

from scripts.run_pcpi_p3g1_structurewise_discrepancy_gate import main


if __name__ == "__main__":
    raise SystemExit(main(Path("configs/p3g_4_r2_function_energy_mixture_gate.json")))
