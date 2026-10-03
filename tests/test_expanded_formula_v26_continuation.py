from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v26_official_provider_contract():
    c=(ROOT/"configs/aistats_three_arm_formula_prospective.yaml").read_text()
    b=(ROOT/"scripts/build_expanded_formula_v26_continuation.py").read_text()
    r=(ROOT/"scripts/run_expanded_formula_discovery_prospective.py").read_text()
    assert "https://open.bigmodel.cn/api/paas/v4" in c
    assert "llm_reasoning_effort: low" in c
    assert '\"reasoning_effort\": \"low\"' in b
    assert "scientific-expanded-formula-discovery-freeze-v2.6" in r
