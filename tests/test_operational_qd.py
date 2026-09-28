"""Response-free operational quality-diversity correctness fixtures."""

from hypothesis_mvp.discovery.operational_qd import (
    EmitterTaskEvidence, OperationalQDArchive, OperationalQDElite,
    fit_emitter_credit, operational_descriptor,
)


def _elite(identity, scores, support, quality, *, safe=True):
    return OperationalQDElite(
        identity, identity, "fixture",
        operational_descriptor(scores, support),
        quality, quality / 2.0, float(support), safe)


def test_descriptor_is_deterministic_and_action_sensitive():
    first = operational_descriptor([0.0, 1.0, 0.0, 0.0], 2)
    repeated = operational_descriptor([0.0, 1.0, 0.0, 0.0], 2)
    shifted = operational_descriptor([0.0, 0.0, 0.0, 1.0], 2)
    assert first == repeated and first != shifted


def test_archive_keeps_one_safe_quality_elite_per_cell():
    archive = OperationalQDArchive()
    weak = _elite("weak", [0.0, 1.0, 0.0], 2, .1)
    strong = _elite("strong", [0.0, 1.0, 0.0], 2, .2)
    unsafe = _elite(
        "unsafe", [0.0, 0.0, 1.0], 2, 1.0, safe=False)
    assert archive.add(weak)["accepted"]
    assert archive.add(strong)["reason"] == (
        "accepted-quality-improvement")
    assert archive.add(unsafe)["reason"] == "rejected-verification-failed"
    assert archive.elites == (strong,)
    assert archive.coverage > 0.0 and archive.qd_score == .2
    assert archive.certificate()["candidate_response_accessed"] is False


def test_emitter_credit_counts_tasks_not_candidates():
    evidence = (
        EmitterTaskEvidence("a", "llm", True),
        EmitterTaskEvidence("b", "llm", False),
        EmitterTaskEvidence("a", "engine", True),
    )
    credit = fit_emitter_credit(evidence)
    assert credit["llm"]["task_count"] == 2
    assert credit["llm"]["successes"] == 1
