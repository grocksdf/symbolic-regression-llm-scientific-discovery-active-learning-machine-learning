from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v212_is_evaluator_only_and_bounded():
 r=(ROOT/"scripts/run_expanded_formula_recovery_v212.py").read_text()
 b=(ROOT/"scripts/build_expanded_formula_recovery_v212.py").read_text()
 assert "generation_or_admission_rerun_authorized\":False" in b
 assert '"global_simplify_forbidden":True' in b
 assert "_evaluate(admissions,metadata" in r
 assert "provider_called\":False" in r
