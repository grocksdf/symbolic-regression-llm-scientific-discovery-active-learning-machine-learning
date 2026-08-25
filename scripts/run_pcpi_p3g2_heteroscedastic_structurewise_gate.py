"""Run the frozen P3G.2 heteroscedastic structure-wise real-data gate."""

from pathlib import Path

from scripts.run_pcpi_p3g1_structurewise_discrepancy_gate import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            Path(
                "configs/p3g_2_heteroscedastic_structurewise_calibration_gate.json"
            )
        )
    )
