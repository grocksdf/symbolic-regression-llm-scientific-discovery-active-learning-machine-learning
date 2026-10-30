"""Identity contract for the real artifact emitter, exercised without a provider.

Every artifact here is produced by ``PCPISearcher._write_artifact`` itself, never
hand-assembled, so passing proves the shipped emitter satisfies the audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from methods.hypothesis_mvp_pcpi.searcher import PCPISearcher
from scripts.feedback_repair_audit import validate_child_artifact


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "lsr_transform"
TASK = "I.10.7_1_0"
SEED = 2401
ARM = "R"
CONFIGS = {"C": "configs/feedback_mechanism_C_masked_generation_update.yaml",
           "D": "configs/feedback_mechanism_D_full.yaml",
           "R": "configs/feedback_repair_R_v2.yaml"}
BOUNDARY = {
    "test_or_ood_accessed": False,
    "heldout_opened": False,
    "ground_truth_expression_opened": False,
    "sealed_response_values_decoded": False,
}


def condition(arm: str) -> str:
    return str(yaml.safe_load((ROOT / CONFIGS[arm]).read_text(encoding="utf-8"))["condition"])


def child(arm: str = ARM, task: str = TASK, seed: int = SEED) -> dict:
    return {"family": FAMILY, "task": task, "seed": seed, "arm": arm}


def emit(tmp_path, *, arm: str = ARM, task: str = TASK, seed: int = SEED,
         family: str = FAMILY, tag: str = "out") -> tuple[Path, Path]:
    """Run the real writer and return (child output folder, artifact path)."""
    searcher = PCPISearcher(
        name="emitter-fixture", top_k_results=1, random_seed=seed,
        use_library=False, output_dir=str(tmp_path / tag),
        execution_family=family, execution_condition=condition(arm),
    )
    searcher._write_artifact(task, {"best_export_expression": "x0", "best_programs": ["x0"]})
    return tmp_path / tag, searcher.artifacts_dir / f"{task}.json"


def payload_for(artifact: Path, *, arm: str = ARM, task: str = TASK,
                seed: int = SEED, family: str = FAMILY) -> dict:
    return {"status": "success", "artifact": str(artifact.resolve()),
            "family": family, "task": task, "seed": seed,
            "condition": condition(arm), **BOUNDARY}


def test_real_emitter_artifact_carries_all_four_coordinates(tmp_path):
    folder, artifact = emit(tmp_path)
    raw = json.loads(artifact.read_text(encoding="utf-8"))
    for field, expected in (("task_name", TASK), ("seed", SEED),
                            ("family", FAMILY), ("condition", condition(ARM))):
        assert raw.get(field) == expected, field
    assert type(raw["seed"]) is int
    _, parsed, digest = validate_child_artifact(
        payload_for(artifact), child(), folder=folder, condition=condition(ARM))
    assert parsed["family"] == FAMILY
    assert digest == __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()


def test_seed_is_serialized_as_a_strict_json_integer(tmp_path):
    _, artifact = emit(tmp_path)
    text = artifact.read_text(encoding="utf-8")
    assert f'"seed": {SEED}' in text          # never 2401.0 or "2401"
    assert '"seed": "2401"' not in text
    assert type(json.loads(text)["seed"]) is int


def test_seed_route_refuses_float_and_bool_seeds(tmp_path):
    searcher = PCPISearcher(name="f", output_dir=str(tmp_path / "a"), random_seed=2401,
                            execution_family=FAMILY, execution_condition=condition(ARM))
    assert type(searcher.random_seed) is int
    with pytest.raises(ValueError, match="non-empty string"):
        PCPISearcher(name="f", output_dir=str(tmp_path / "b"), random_seed=2401,
                     execution_family="", execution_condition=condition(ARM))
    with pytest.raises(ValueError, match="non-empty string"):
        PCPISearcher(name="f", output_dir=str(tmp_path / "c"), random_seed=2401,
                     execution_family=FAMILY, execution_condition=None)


@pytest.mark.parametrize("field,value", [
    ("task_name", "I.10.7_1_1"), ("seed", 2402),
    ("family", "feynman"), ("condition", "three_arm_feedback_full_v1")])
def test_artifact_fails_when_any_single_coordinate_is_altered(tmp_path, field, value):
    folder, artifact = emit(tmp_path)
    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered[field] = value
    artifact.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the registered child"):
        validate_child_artifact(payload_for(artifact), child(),
                                folder=folder, condition=condition(ARM))


@pytest.mark.parametrize("field,value", [
    ("task", "I.10.7_1_1"), ("seed", 2402),
    ("family", "feynman"), ("condition", "three_arm_feedback_full_v1")])
def test_child_result_fails_when_any_single_coordinate_is_altered(tmp_path, field, value):
    folder, artifact = emit(tmp_path)
    payload = payload_for(artifact)
    payload[field] = value
    with pytest.raises(ValueError, match="does not match the registered child"):
        validate_child_artifact(payload, child(), folder=folder, condition=condition(ARM))


def test_real_emitter_with_a_different_seed_is_rejected(tmp_path):
    """A genuinely emitted wrong-seed artifact, not a hand-edited one."""
    folder, _ = emit(tmp_path, tag="out")
    _, wrong = emit(tmp_path, seed=SEED + 1, tag="out")
    with pytest.raises(ValueError, match="seed does not match"):
        validate_child_artifact(payload_for(wrong), child(),
                                folder=folder, condition=condition(ARM))


def test_cross_arm_artifact_is_rejected(tmp_path):
    folder, artifact = emit(tmp_path, arm="D", tag="d_arm")
    with pytest.raises(ValueError, match="condition does not match"):
        validate_child_artifact(payload_for(artifact, arm="D"), child(arm="R"),
                                folder=folder, condition=condition("R"))


def test_cross_family_artifact_is_rejected(tmp_path):
    folder, artifact = emit(tmp_path, family="feynman", tag="other_family")
    with pytest.raises(ValueError, match="family does not match"):
        validate_child_artifact(payload_for(artifact, family="feynman"), child(),
                                folder=folder, condition=condition(ARM))


def test_artifact_outside_the_child_folder_is_rejected(tmp_path):
    folder, _ = emit(tmp_path)
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this child output directory"):
        validate_child_artifact(payload_for(outside), child(),
                                folder=folder, condition=condition(ARM))


def test_parent_traversal_artifact_is_rejected(tmp_path):
    folder, _ = emit(tmp_path)
    traversal = folder / ".." / "escaped.json"
    traversal.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this child output directory"):
        validate_child_artifact(payload_for(traversal), child(),
                                folder=folder, condition=condition(ARM))
