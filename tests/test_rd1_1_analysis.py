from scripts.analyze_rd1_1_ablation import analyze
from scripts.run_rd1_1_real_development import _contract
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preregistered_ablation_analysis_uses_complete_pairs() -> None:
    runs = []
    for dataset in ("a", "b", "c"):
        for seed in (42, 31415, 27182, 16180):
            runs.append({
                "dataset": dataset, "seed": seed, "variant": "full",
                "best_val_nmse": 0.4,
            })
            for variant, value in (
                ("no_llm", 0.5), ("single_engine", 0.52), ("no_acquisition", 0.48)
            ):
                runs.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "best_val_nmse": value,
                })
    summary = {
        "runs": runs, "heldout_opened": False, "selection_used_heldout": False,
    }
    contract = _contract(ROOT / "contracts" / "rd1_1_real_development_contract.json")
    report = analyze(summary, contract)
    assert report["heldout_opened"] is False
    for variant in ("no_llm", "single_engine", "no_acquisition"):
        row = report["statistics"][variant]
        assert row["complete_pairs"] == 12
        assert row["median_effect"] > 0.0
        assert row["supported"] is True
