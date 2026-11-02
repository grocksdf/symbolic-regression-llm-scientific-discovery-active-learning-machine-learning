"""Replay three admission policies on identical frozen historical candidates."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY,
    support_parser_for_policy,
)
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.discovery.regional_candidate_admission import (
    _fixed_predictive_logpdf,
    admit_global_protected_candidates,
    global_protected_evidence_decision,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from run_gap_candidate_marginal_value import (
    _generation_roles,
    _gap_admission_role,
    _posterior_update_role,
    _reporting_response,
    _score,
    identity,
)

ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path):
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_new(path: Path, value):
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _source_identity(root: Path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True).strip()
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(git("status", "--short", "--untracked-files=no")),
    }


def _bootstrap(task_values, seed, count):
    values = np.asarray(list(task_values.values()), dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(count, dtype=float)
    for index in range(count):
        samples[index] = float(np.mean(
            values[rng.integers(0, len(values), len(values))]))
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5))]


def _same_data_admission(
        core, optional, fit, gap_audit, domain, regions,
        eligible_regions):
    """Deliberately reuse gap-audit responses for a leaky control arm."""
    if not optional:
        return [], {"attempts": 0, "candidates": []}
    prior = NormalInverseGammaPrior()
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    core_supports = {
        parser(row["expression"], fit.X.shape[1]) for row in core}
    core_score, core_identity = _fixed_predictive_logpdf(
        core, fit, gap_audit, domain, prior, 2,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    assignments = regions.assign(gap_audit.X)
    threshold = math.log(len(optional) / .025)
    retained, candidates, admitted_supports = [], [], set()
    for candidate in optional:
        support = parser(candidate["expression"], fit.X.shape[1])
        common = {
            "candidate_identity": sha256(json.dumps(
                candidate, sort_keys=True, default=str).encode()).hexdigest(),
            "composed_expression": candidate["expression"],
            "support": list(support)}
        if support in core_supports:
            candidates.append({
                **common, "admitted": False,
                "reason": "duplicate-engine-support"})
            continue
        optional_score, law_identity = _fixed_predictive_logpdf(
            [candidate], fit, gap_audit, domain, prior, 2,
            coefficient_policy=EXPANDED_FORMULA_POLICY)
        decision = global_protected_evidence_decision(
            optional_score - core_score, assignments,
            eligible_regions, threshold)
        duplicate = decision["passed"] and support in admitted_supports
        admitted = bool(decision["passed"] and not duplicate)
        if admitted:
            retained.append(dict(candidate))
            admitted_supports.add(support)
        candidates.append({
            **common, **decision, "expert_identity": law_identity,
            "admitted": admitted,
            "reason": (
                "duplicate-admitted-support" if duplicate
                else decision["reason"])})
    return retained, {
        "schema": "candidate-same-data-global-admission-v1",
        "core_identity": core_identity,
        "attempts": len(optional), "candidates": candidates,
        "same_gap_audit_responses_reused": True,
        "candidate_response_accessed": False,
        "heldout_opened": False}


def _accept_all(core, optional, n_features):
    parser = support_parser_for_policy(EXPANDED_FORMULA_POLICY)
    seen = {parser(row["expression"], n_features) for row in core}
    retained = []
    for candidate in optional:
        support = parser(candidate["expression"], n_features)
        if support not in seen:
            retained.append(dict(candidate))
            seen.add(support)
    return retained, {
        "schema": "candidate-accept-all-admission-v1",
        "attempts": len(optional), "admitted": len(retained),
        "response_accessed": False,
        "candidate_response_accessed": False,
        "heldout_opened": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("admission ablation output root already exists")
    freeze = _read(args.freeze)
    if (freeze.get("schema") != "protected-admission-ablation-freeze-v1"
            or freeze.get("execution_authorized") is not True
            or freeze.get("provider_calls_authorized") is not False):
        raise ValueError("invalid admission ablation freeze")
    for root, key in ((ROOT, "benchmark"), (MAINLINE, "mainline")):
        actual = _source_identity(root)
        expected = freeze["source"][key]
        if (actual["tracked_dirty"]
                or actual["commit"] != expected["commit"]
                or actual["tree"] != expected["tree"]):
            raise ValueError(f"{key} source identity mismatch")
    for path, digest in freeze["file_hashes"].items():
        if _digest(Path(path)) != digest:
            raise ValueError(f"frozen code identity mismatch: {path}")
    for row in freeze["artifact_manifest"]:
        if _digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen artifact identity mismatch: {row['path']}")
    hdf5 = Path(freeze["hdf5"])
    if _digest(hdf5) != freeze["hdf5_sha256"]:
        raise ValueError("HDF5 identity mismatch")
    args.output_root.mkdir(parents=True, exist_ok=False)
    _write_new(args.output_root / "RUN_FREEZE.json", freeze)
    result_rows = []
    arms = tuple(freeze["arms"])
    for index, coordinate in enumerate(freeze["coordinates"], start=1):
        family = coordinate["family"]
        task = coordinate["task"]
        seed = int(coordinate["seed"])
        print(f"[{index}/{freeze['coordinate_count']}] "
              f"admission replay {family}/{task} seed={seed}", flush=True)
        if coordinate["coordinate_failure"]:
            for arm in arms:
                result_rows.append({
                    "family": family, "task": task, "seed": seed,
                    "arm": arm, "gain_vs_E0": 0.0,
                    "candidate_count": 0, "coordinate_failure": True})
            continue
        shared = _read(Path(coordinate["shared_block"]))
        generation = _read(Path(coordinate["generation_result"]))
        E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
        optional = [dict(row) for row in generation["optional_rows"]]
        roles = _generation_roles(hdf5, family, task, seed)
        selection = roles["selection"]
        gap_admission = _gap_admission_role(
            hdf5, family, task, roles["indices"])
        posterior_update = _posterior_update_role(
            hdf5, family, task, roles["indices"])
        y_report = _reporting_response(
            hdf5, family, task, roles["indices"])
        baseline, _ = _score(
            E0, selection, roles["reporting_X"], y_report,
            identity({"task": task, "seed": seed, "arm": "E0"}),
            posterior_update=posterior_update)
        # The source runner freezes one split at the median of acquisition
        # covariate zero. Reconstruct it deterministically and verify its
        # identity against the immutable source admission artifact.
        gap_record = generation["admission"]
        cuts = (float(np.median(
            selection.acquisition_pool.X[:, 0])),)
        regions = FrozenAxisRegions(
            n_features=selection.development.X.shape[1],
            axis=0, cuts=cuts)
        if (gap_record.get("region_identity")
                and gap_record["region_identity"] != regions.identity):
            raise ValueError("reconstructed region identity mismatch")
        eligible_regions = tuple(range(len(cuts) + 1))
        independent, independent_audit = (
            admit_global_protected_candidates(
                E0, optional, selection.development,
                roles["gap_audit"], gap_admission,
                selection.acquisition_pool.X, regions,
                eligible_regions, NormalInverseGammaPrior(), 2,
                alpha=.025,
                coefficient_policy=EXPANDED_FORMULA_POLICY)
            if optional else ([], {"attempts": 0, "candidates": []}))
        same_data, same_data_audit = _same_data_admission(
            E0, optional, selection.development, roles["gap_audit"],
            selection.acquisition_pool.X, regions, eligible_regions)
        accept_all, accept_all_audit = _accept_all(
            E0, optional, selection.development.X.shape[1])
        projected = {
            "independent_protected": independent,
            "same_data": same_data,
            "accept_all": accept_all}
        projection_audits = {
            "independent_protected": independent_audit,
            "same_data": same_data_audit,
            "accept_all": accept_all_audit}
        candidate_bank_identity = identity({
            "E0": E0, "optional": optional,
            "source_response": generation.get("raw_response_sha256")})
        for arm in arms:
            bank = [*E0, *projected[arm]]
            score, audit = _score(
                bank, selection, roles["reporting_X"], y_report,
                identity({"task": task, "seed": seed, "arm": arm}),
                posterior_update=posterior_update)
            optional_count = max(0, len(bank) - len(E0))
            result_rows.append({
                "family": family, "task": task, "seed": seed,
                "arm": arm, "baseline_log_score": baseline,
                "arm_log_score": score, "gain_vs_E0": score - baseline,
                "candidate_count": optional_count,
                "candidate_bank_identity": candidate_bank_identity,
                "admission_audit": projection_audits[arm],
                "score_audit": audit, "coordinate_failure": False})
    task_arm = {}
    for row in result_rows:
        task_arm.setdefault(row["task"], {}).setdefault(
            row["arm"], []).append(float(row["gain_vs_E0"]))
    task_means = {
        task: {
            arm: float(np.mean(values))
            for arm, values in by_arm.items()}
        for task, by_arm in task_arm.items()}
    arm_summaries = {}
    for arm in arms:
        values = {
            task: by_arm[arm] for task, by_arm in task_means.items()}
        arm_summaries[arm] = {
            "task_mean_gain_vs_E0": float(np.mean(list(values.values()))),
            "ci95": _bootstrap(
                values, int(freeze["analysis"]["bootstrap_seed"]),
                int(freeze["analysis"]["resamples"])),
            "positive_task_count": sum(value > 0 for value in values.values()),
            "negative_task_count": sum(value < 0 for value in values.values()),
            "nonzero_candidate_coordinates": sum(
                row["candidate_count"] > 0
                for row in result_rows if row["arm"] == arm)}
    contrasts = {}
    for left, right in (
            ("independent_protected", "same_data"),
            ("independent_protected", "accept_all")):
        values = {
            task: by_arm[left] - by_arm[right]
            for task, by_arm in task_means.items()}
        contrasts[f"{left}_minus_{right}"] = {
            "task_mean": float(np.mean(list(values.values()))),
            "ci95": _bootstrap(
                values, int(freeze["analysis"]["bootstrap_seed"]) + 1,
                int(freeze["analysis"]["resamples"]))}
    result = {
        "schema": "protected-admission-ablation-result-v1",
        "protocol_complete": True,
        "claim_boundary": freeze["claim_boundary"],
        "provider_calls": 0,
        "candidate_banks_reused": True,
        "coordinate_count": freeze["coordinate_count"],
        "task_count": freeze["task_count"],
        "row_count": len(result_rows),
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "task_arm_means": task_means,
        "rows": result_rows,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
    }
    _write_new(args.output_root / "RESULT.json", result)
    print(json.dumps({
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "protocol_complete": True,
        "provider_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
