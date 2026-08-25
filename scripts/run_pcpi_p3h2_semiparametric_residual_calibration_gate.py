"""Run the frozen P3H.2 semiparametric residual real calibration-only Gate."""

from pathlib import Path

from scripts.run_pcpi_p3g1_structurewise_discrepancy_gate import main


if __name__ == "__main__":
    raise SystemExit(
        main(Path("configs/p3h_2_semiparametric_residual_calibration_gate.json"))
    )
