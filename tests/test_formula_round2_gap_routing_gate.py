import scripts.run_formula_round2_gap_routing_gate as gate


def test_gap_routing_gate_is_response_free(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gate, "verify_clean_git_source",
        lambda path: {"source_identity_kind": "fixture",
                      "source_git_dirty": False})
    output = tmp_path / "gate"
    assert gate.main(["--output-dir", str(output)]) == 0
    assert (output / "GATE.json").is_file()
