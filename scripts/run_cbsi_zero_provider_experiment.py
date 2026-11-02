"""Run the frozen CB-SI provider-free development mechanism experiment."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAINLINE = ROOT.parent / "hypothesis_mvp"
sys.path[:0] = [str(ROOT), str(MAINLINE), str(ROOT / "scripts")]

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
    BayesianOperatorPolicy,
    StructuralEditingEngine,
    counterfactual_edit_credit,
)
from hypothesis_mvp.discovery.joint_evidence_gated_expansion import (
    CandidateArchive,
    extract_subexpression_materials,
)
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY,
    freeze_discovery_target,
    freeze_expanded_formula_model,
)
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from run_gap_candidate_marginal_value import (
    _gap_admission_role,
    _generation_roles,
    identity,
)
from run_jegpe_frozen_pool_diagnostic import role
from stacking_diagnostics import fit_stacking_weights, mixture_log_density


ARM_MODES = {
    "blind_random": "random",
    "single_fitness_proxy": "fitness",
    "residual_proxy": "residual",
    "influence_proxy": "influence",
    "bank_greedy": "greedy",
    "standard_thompson": "thompson_raw",
    "cbsi_full": "thompson_matched",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    if path.exists():
        raise ValueError(f"CB-SI output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def density(rows, selection, X, y, namespace):
    model = freeze_expanded_formula_model(
        rows, n_features=selection.development.X.shape[1],
        prior=NormalInverseGammaPrior(),
        exploration_identity=identity(namespace),
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    target = freeze_discovery_target(
        model, selection.development, selection.acquisition_pool.X,
        measurement_budget=2,
        expected_model_identity=model.stable_hash)
    values = np.asarray(
        model.engine(model.stable_hash).predictive_logpdf(
            target.initial_posterior, X, y), dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("CB-SI predictive density is nonfinite")
    return values


class CrossfitBankScorer:
    def __init__(self, E0, selection, left: RoleDataset,
                 right: RoleDataset, evaluation: RoleDataset,
                 namespace: str, seed: int):
        self.E0 = [dict(row) for row in E0]
        self.selection = selection
        self.left = left
        self.right = right
        self.evaluation = evaluation
        self.namespace = namespace
        self.seed = int(seed)
        self.cache = {}
        self.score_calls = 0
        self.e0_supports = {
            tuple(compile_fixed_formula_support(
                str(row["expression"]),
                selection.development.X.shape[1]))
            for row in self.E0}
        self._base = {
            "left": density(
                self.E0, selection, left.X, left.y,
                {"namespace": namespace, "bank": "E0", "fold": "left"}),
            "right": density(
                self.E0, selection, right.X, right.y,
                {"namespace": namespace, "bank": "E0", "fold": "right"}),
            "evaluation": density(
                self.E0, selection, evaluation.X, evaluation.y,
                {"namespace": namespace, "bank": "E0", "fold": "evaluation"}),
        }

    def _candidate_density(self, expression: str, fold: str):
        key = (expression, fold)
        if key in self.cache:
            return self.cache[key]
        support = tuple(compile_fixed_formula_support(
            expression, self.selection.development.X.shape[1]))
        if support in self.e0_supports:
            values = self._base[fold]
            self.cache[key] = values
            return values
        role_value = {
            "left": self.left,
            "right": self.right,
            "evaluation": self.evaluation,
        }[fold]
        row = {
            "expression": expression,
            "source": "cbsi:structural_edit",
            "origin": "archive-edit",
            "lineage_id": "cbsi-" + sha256(
                expression.encode()).hexdigest()[:24],
        }
        values = density(
            [*self.E0, row], self.selection,
            role_value.X, role_value.y,
            {"namespace": self.namespace,
             "candidate": sha256(expression.encode()).hexdigest(),
             "fold": fold})
        self.score_calls += 1
        self.cache[key] = values
        return values

    @staticmethod
    def _mixture_gain(cal_base, cal_candidate,
                      score_base, score_candidate, seed):
        cal = np.column_stack((cal_base, cal_candidate))
        score = np.column_stack((score_base, score_candidate))
        weights, audit = fit_stacking_weights(
            cal, seed=int(seed), random_starts=2)
        mixed = mixture_log_density(score, weights)
        gain = float(np.mean(mixed - score_base))
        return gain, weights, audit

    def crossfit_gain(self, expression: str):
        left_candidate = self._candidate_density(expression, "left")
        right_candidate = self._candidate_density(expression, "right")
        left_to_right, _, _ = self._mixture_gain(
            self._base["left"], left_candidate,
            self._base["right"], right_candidate, self.seed + 1)
        right_to_left, _, _ = self._mixture_gain(
            self._base["right"], right_candidate,
            self._base["left"], left_candidate, self.seed + 2)
        return (left_to_right, right_to_left)

    def evaluation_gain(self, expression: str):
        left_candidate = self._candidate_density(expression, "left")
        right_candidate = self._candidate_density(expression, "right")
        eval_candidate = self._candidate_density(expression, "evaluation")
        calibration_base = np.concatenate((
            self._base["left"], self._base["right"]))
        calibration_candidate = np.concatenate((
            left_candidate, right_candidate))
        gain, weights, audit = self._mixture_gain(
            calibration_base, calibration_candidate,
            self._base["evaluation"], eval_candidate, self.seed + 3)
        return gain, weights.tolist(), audit


def finite_prediction(runtime, expression, X):
    try:
        return runtime.predict(expression, X)
    except Exception:
        return None


def proxy_context(E0, archive, selection, runtime):
    development = selection.development
    # Parent selection is search-only.  Never use the validation fold that
    # subsequently supplies counterfactual-credit evaluations.
    editor = StructuralEditingEngine(
        selection.development.X.shape[1])
    rows = []
    for candidate in archive.candidates:
        try:
            if not editor.editable_paths(candidate.expression):
                continue
        except Exception:
            continue
        pred_dev = finite_prediction(
            runtime, candidate.expression, development.X)
        if pred_dev is None:
            continue
        rows.append({
            "candidate": candidate,
            "development_mse": float(np.mean(
                (development.y - pred_dev) ** 2)),
        })
    if not rows:
        raise ValueError("CB-SI archive has no executable parent")
    best = min(rows, key=lambda row: (
        row["development_mse"],
        row["candidate"].structural_hash))
    baseline_pred = finite_prediction(
        runtime, best["candidate"].expression, development.X)
    residual = development.y - baseline_pred

    materials = extract_subexpression_materials(archive)
    material_rows = []
    for material in materials:
        pred = finite_prediction(
            runtime, material["expression"], development.X)
        if pred is None or float(np.std(pred)) <= 1e-12:
            continue
        corr = float(abs(np.corrcoef(residual, pred)[0, 1]))
        if not math.isfinite(corr):
            corr = 0.0
        material_rows.append({**material, "residual_correlation": corr})
    if not material_rows:
        material_rows = [{
            "expression": f"x{index}",
            "structural_hash": f"variable-{index}",
            "residual_correlation": 0.0,
        } for index in range(development.X.shape[1])]

    engine_rows = []
    for row in E0:
        pred = finite_prediction(
            runtime, str(row["expression"]), development.X)
        if pred is not None:
            engine_rows.append((row, pred))
    if not engine_rows:
        raise ValueError("E0 has no executable engine formula")
    full = np.mean(
        np.column_stack([pred for _, pred in engine_rows]), axis=1)
    full_mse = float(np.mean((development.y - full) ** 2))
    influence = {}
    for index, (row, _) in enumerate(engine_rows):
        if len(engine_rows) == 1:
            value = full_mse
        else:
            without = np.mean(np.column_stack([
                pred for other_index, (_, pred) in enumerate(engine_rows)
                if other_index != index]), axis=1)
            value = float(np.mean(
                (development.y - without) ** 2) - full_mse)
        influence[str(row["expression"])] = value
    return {
        "parents": rows,
        "best_parent": best["candidate"],
        "materials": material_rows,
        "best_residual_material": max(
            material_rows,
            key=lambda row: (
                row["residual_correlation"], row["structural_hash"])),
        "influence": influence,
    }


def choose_parent_and_donor(mode, context, rng, step):
    parents = context["parents"]
    materials = context["materials"]
    if mode == "fitness":
        parent = context["best_parent"]
    elif mode == "influence":
        parent = max(
            (row["candidate"] for row in parents),
            key=lambda candidate: (
                context["influence"].get(candidate.expression, -math.inf),
                candidate.structural_hash))
    else:
        parent = parents[int(rng.integers(0, len(parents)))]["candidate"]
    if mode == "residual":
        donor = context["best_residual_material"]["expression"]
    else:
        donor = materials[int(rng.integers(0, len(materials)))]["expression"]
    return parent, donor


def fixed_operator(mode, operators, step, rng):
    if mode == "random":
        return operators[int(rng.integers(0, len(operators)))]
    if mode == "fitness":
        return operators[step % len(operators)]
    if mode == "residual":
        cycle = (
            "correction_term", "subtree_composition",
            "interaction_insertion")
        return cycle[step % len(cycle)]
    if mode == "influence":
        cycle = (
            "operator_substitution", "subtree_replacement",
            "subtree_composition")
        return cycle[step % len(cycle)]
    raise ValueError("mode does not use fixed operator")


def build_shared_opportunities(freeze, coordinate, context, n_features):
    """Freeze exogenous edit opportunities before any arm learns rewards.

    These opportunities are shared by random, bank-greedy, standard
    Thompson and CB-SI.  Fitness/residual/influence proxy arms intentionally
    use their own parent/donor heuristics and MUST be reported separately.
    """
    seed = int(coordinate["seed"]) + int(sha256(
        f"{coordinate['family']}:{coordinate['task']}:shared-edit".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    editor = StructuralEditingEngine(n_features)
    opportunities = []
    evaluations = int(freeze["budget"]["edit_evaluations_per_arm"])
    if evaluations <= 0 or evaluations % 2:
        raise ValueError("CB-SI edit budget must be positive and paired")
    for block in range(evaluations // 2):
        parent, donor = choose_parent_and_donor(
            "random", context, rng, block)
        sites = editor.editable_paths(parent.expression)
        # Same parent, donor and edit site across the core policy arms.
        # Prefer genuinely substitutable nodes so all operators can act.
        candidates = [
            path for path in sites
            if editor._operator_supported_at_site(
                "operator_substitution", _site_node(
                    parent.expression, path))]
        usable = candidates if candidates else sites
        site = usable[int(rng.integers(0, len(usable)))]
        opportunities.append({
            "parent": parent,
            "donor": donor,
            "site_path": tuple(site),
            "opportunity_seed": int(seed + block),
            "opportunity_id": sha256(
                f"{coordinate['task']}:{coordinate['seed']}:{block}:"
                f"{parent.structural_hash}:{donor}:{site}".encode()
            ).hexdigest(),
        })
    return tuple(opportunities)


def _site_node(expression, site_path):
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        _node_at_path, _parse_body)
    return _node_at_path(_parse_body(expression), site_path)


def run_arm(arm, freeze, coordinate, archive, E0, selection,
            left, right, evaluation, context, domains,
            shared_opportunities):
    mode = ARM_MODES[arm]
    operators = tuple(freeze["operators"])
    seed = int(coordinate["seed"]) + int(
        sha256(arm.encode()).hexdigest()[:8], 16)
    scorer_seed = int(coordinate["seed"]) + int(sha256(
        f"{coordinate['family']}:{coordinate['task']}:shared-scorer".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    runtime = StructuralEditingEngine(
        selection.development.X.shape[1])
    scorer = CrossfitBankScorer(
        E0, selection, left, right, evaluation,
        f"{coordinate['task']}/{coordinate['seed']}/shared-scorer",
        scorer_seed)
    policy = BayesianOperatorPolicy(operators=operators, seed=seed)
    initial_hashes = {
        candidate.structural_hash for candidate in archive.candidates}
    seen = set(initial_hashes)
    evaluations = int(
        freeze["budget"]["edit_evaluations_per_arm"])
    if evaluations % 2:
        raise ValueError("CB-SI edit budget must be paired")
    records = []
    eligible = []
    started = time.perf_counter()
    shared_modes = {"random", "greedy", "thompson_raw",
                    "thompson_matched"}
    for block in range(evaluations // 2):
        shared = shared_opportunities[block]
        if mode in shared_modes:
            parent, donor = shared["parent"], shared["donor"]
            site_path = shared["site_path"]
        else:
            parent, donor = choose_parent_and_donor(
                mode, context, rng, block)
            site_path = None
        if mode in {"greedy", "thompson_raw", "thompson_matched"}:
            operator = policy.choose(
                block, mode="greedy" if mode == "greedy" else "thompson")
        else:
            operator = fixed_operator(
                mode, operators, block, rng)
        primary_request = runtime.deterministic_request(
            operator, parent.expression, donor,
            seed=(shared["opportunity_seed"] if mode in shared_modes
                  else seed + block),
            task_identity=(
                f"{coordinate['task']}:{coordinate['seed']}:"
                f"{shared['opportunity_id'] if mode in shared_modes else arm}:"
                f"{block}"),
            site_path=site_path)
        control_request = runtime.matched_control_request(primary_request)
        primary = runtime.apply(
            primary_request, registered_domains=domains,
            known_structural_hashes=seen)
        control = runtime.apply(
            control_request, registered_domains=domains,
            known_structural_hashes=seen)
        # Missing/invalid edits are NOT zero-gain observations.  Only valid
        # novel edits can be scored, and only a valid matched pair can inform
        # the counterfactual policy.  Keep ineligible attempts in the audit.
        primary_usable = bool(primary.valid and primary.novel)
        control_usable = bool(control.valid and control.novel)
        parent_folds = None
        primary_folds = None
        control_folds = None
        credit = None
        if primary_usable:
            primary_folds = scorer.crossfit_gain(str(primary.expression))
            seen.add(str(primary.structural_hash))
            eligible.append({
                "result": primary,
                "search_gain": float(np.mean(primary_folds)),
                "credit": None,
            })
            if mode in {"greedy", "thompson_raw"}:
                policy.update(
                    operator, float(np.mean(primary_folds)),
                    reward_kind="raw_bank_gain",
                    audit={"paired_control_valid": control_usable})
        if primary_usable and control_usable:
            parent_folds = scorer.crossfit_gain(parent.expression)
            control_folds = scorer.crossfit_gain(str(control.expression))
            parent_complexity = float(
                runtime.runtime.dag(parent.expression).complexity)
            credit = counterfactual_edit_credit(
                operator, primary_folds, parent_folds,
                (control_folds,),
                primary_cost=float(primary.complexity) - parent_complexity,
                control_costs=(
                    float(control.complexity) - parent_complexity,),
                cost_weight=float(freeze["budget"]["complexity_cost_weight"]))
            eligible[-1]["credit"] = credit.credit
            if mode == "thompson_matched":
                policy.update(
                    operator, credit.credit,
                    reward_kind="matched_counterfactual_credit",
                    audit={"credit": credit.row()})
        records.append({
            "block": block,
            "opportunity_id": shared["opportunity_id"] if
                mode in shared_modes else None,
            "shared_opportunity": mode in shared_modes,
            "primary": primary.row(),
            "control": control.row(),
            "parent_structural_hash": parent.structural_hash,
            "parent_fold_gains": (
                list(parent_folds) if parent_folds is not None else None),
            "primary_fold_gains": (
                list(primary_folds) if primary_folds is not None else None),
            "control_fold_gains": (
                list(control_folds) if control_folds is not None else None),
            "counterfactual_pair_valid": credit is not None,
            "counterfactual_pair_exclusion_reason": (
                None if credit is not None else
                "primary_invalid_or_non_novel" if not primary_usable else
                "control_invalid_or_non_novel"),
            "credit": credit.row() if credit is not None else None,
        })
    selected = (
        max(eligible, key=lambda row: (
            row["search_gain"],
            row["credit"] if row["credit"] is not None else -math.inf,
            str(row["result"].structural_hash)))
        if eligible else None)
    if selected is not None and selected["search_gain"] > 0:
        evaluation_gain, weights, optimizer = scorer.evaluation_gain(
            str(selected["result"].expression))
        selected_row = selected["result"].row()
    else:
        evaluation_gain, weights, optimizer = (
            0.0, [1.0], {"component_count": 1})
        selected_row = None
    return {
        "arm": arm,
        "mode": mode,
        "shared_scorer_seed": scorer_seed,
        "edit_evaluation_budget": evaluations,
        "primary_attempt_count": len(records),
        "matched_control_count": sum(
            row["counterfactual_pair_valid"] for row in records),
        "attempted_control_count": len(records),
        "invalid_pair_count": sum(
            not row["counterfactual_pair_valid"] for row in records),
        "shared_opportunities": mode in shared_modes,
        "valid_primary_count": sum(
            row["primary"]["valid"] for row in records),
        "novel_primary_count": sum(
            row["primary"]["novel"] for row in records),
        "valid_control_count": sum(
            row["control"]["valid"] for row in records),
        "selected": selected_row,
        "selected_search_gain": (
            selected["search_gain"] if selected else 0.0),
        "selected_counterfactual_credit": (
            selected["credit"] if selected else None),
        "independent_development_evaluation_gain": evaluation_gain,
        "evaluation_stacking_weights": weights,
        "evaluation_optimizer": optimizer,
        "score_calls": scorer.score_calls,
        "wall_time_seconds": time.perf_counter() - started,
        "provider_calls": 0,
        "pysr_calls": 0,
        "policy": policy.audit(),
        "records": records,
    }


def bootstrap(values, seed, count):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(count))
    for index in range(len(samples)):
        samples[index] = float(np.mean(
            data[rng.integers(0, len(data), len(data))]))
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("CB-SI output root already exists")
    freeze = read(args.freeze)
    if (freeze.get("schema")
            != "cbsi-zero-provider-development-freeze-v2"
            or freeze.get("execution_authorized") is not True
            or freeze.get("provider_calls_authorized") is not False
            or freeze.get("expensive_engine_runs_authorized") is not False
            or freeze.get("protocol_id") != "cbsi-correctness-hotfix-v2"
            or not isinstance(freeze.get("budget", {}).get(
                "complexity_cost_weight"), (int, float))
            or not math.isfinite(float(freeze["budget"][
                "complexity_cost_weight"]))
            or float(freeze["budget"]["complexity_cost_weight"]) < 0):
        raise ValueError("invalid CB-SI v2 zero-provider freeze")
    if digest(Path(freeze["hdf5"])) != freeze["hdf5_sha256"]:
        raise ValueError("CB-SI HDF5 identity mismatch")
    for path, expected in freeze["file_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"CB-SI code identity mismatch: {path}")
    for row in freeze["artifact_manifest"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(
                f"CB-SI candidate artifact mismatch: {row['path']}")
    args.output_root.mkdir(parents=True)
    write_new(args.output_root / "RUN_FREEZE.json", freeze)

    hdf5 = Path(freeze["hdf5"])
    rows = []
    task_csv = []
    for index, coordinate in enumerate(freeze["coordinates"], start=1):
        family = coordinate["family"]
        task = coordinate["task"]
        seed = int(coordinate["seed"])
        print(
            f"[{index}/{freeze['coordinate_count']}] CB-SI {task} seed={seed}",
            flush=True)
        root = Path(coordinate["coordinate_root"])
        shared = read(root / "SHARED_BLOCK.json")
        E0 = [dict(row) for row in shared["shared"]["E0_rows"]]
        candidate_rows = []
        for condition in freeze["candidate_conditions"]:
            generation = read(root / condition / "GENERATION_RESULT.json")
            candidate_rows.extend(generation["optional_rows"])
        roles = _generation_roles(hdf5, family, task, seed)
        selection = roles["selection"]
        left = selection.validation
        right = roles["gap_audit"]
        evaluation = _gap_admission_role(
            hdf5, family, task, roles["indices"])
        archive = CandidateArchive.freeze(
            [*E0, *candidate_rows],
            n_features=selection.development.X.shape[1],
            coefficient_policy=EXPANDED_FORMULA_POLICY)
        runtime = StructuralEditingEngine(
            selection.development.X.shape[1]).runtime
        context = proxy_context(
            E0, archive, selection, runtime)
        domains = (
            selection.development.X,
            left.X,
            right.X,
            evaluation.X,
            selection.acquisition_pool.X,
        )
        shared_opportunities = build_shared_opportunities(
            freeze, coordinate, context,
            selection.development.X.shape[1])
        arms = {}
        for arm in freeze["arms"]:
            arms[arm] = run_arm(
                arm, freeze, coordinate, archive, E0, selection,
                left, right, evaluation, context, domains,
                shared_opportunities)
            write_new(
                args.output_root / "coordinates" / task /
                f"seed{seed}" / f"{arm}.json",
                arms[arm])
        row = {
            "family": family, "task": task, "seed": seed,
            "initial_bank_size": len(E0),
            "archive_size": len(archive.candidates),
            "subexpression_material_count": len(
                extract_subexpression_materials(archive)),
            "arms": arms,
        }
        rows.append(row)
        for arm, result in arms.items():
            task_csv.append({
                "family": family,
                "task": task,
                "seed": seed,
                "arm": arm,
                "evaluation_gain":
                    result["independent_development_evaluation_gain"],
                "search_gain": result["selected_search_gain"],
                "counterfactual_credit":
                    result["selected_counterfactual_credit"],
                "novel_primary_count": result["novel_primary_count"],
                "valid_primary_count": result["valid_primary_count"],
                "score_calls": result["score_calls"],
                "wall_time_seconds": result["wall_time_seconds"],
                "provider_calls": 0,
                "pysr_calls": 0,
            })
    csv_path = args.output_root / "CBSI_TASK_LEVEL_RESULTS.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_csv[0]))
        writer.writeheader()
        writer.writerows(task_csv)

    summaries = {}
    for arm in freeze["arms"]:
        values = [
            row["arms"][arm][
                "independent_development_evaluation_gain"]
            for row in rows]
        summaries[arm] = {
            "task_mean_evaluation_gain": float(np.mean(values)),
            "ci95": bootstrap(
                values, freeze["analysis"]["bootstrap_seed"],
                freeze["analysis"]["bootstrap_resamples"]),
            "positive_task_count": sum(value > 1e-12 for value in values),
            "negative_task_count": sum(value < -1e-12 for value in values),
            "zero_task_count": sum(abs(value) <= 1e-12 for value in values),
            "total_novel_primary_count": sum(
                row["arms"][arm]["novel_primary_count"] for row in rows),
            "total_valid_primary_count": sum(
                row["arms"][arm]["valid_primary_count"] for row in rows),
            "total_score_calls": sum(
                row["arms"][arm]["score_calls"] for row in rows),
            "total_wall_time_seconds": sum(
                row["arms"][arm]["wall_time_seconds"] for row in rows),
            "provider_calls": 0,
            "pysr_calls": 0,
        }
    cbsi = summaries["cbsi_full"]["task_mean_evaluation_gain"]
    strongest_baseline_name = max(
        (arm for arm in freeze["arms"] if arm != "cbsi_full"),
        key=lambda arm: summaries[arm]["task_mean_evaluation_gain"])
    strongest = summaries[strongest_baseline_name][
        "task_mean_evaluation_gain"]
    mechanism_passed = bool(
        cbsi > summaries["single_fitness_proxy"][
            "task_mean_evaluation_gain"]
        and cbsi > summaries["residual_proxy"][
            "task_mean_evaluation_gain"])
    result = {
        "schema": "cbsi-zero-provider-development-result-v1",
        "protocol_complete": True,
        "matched_opportunity_policy_arms": [
            "blind_random", "bank_greedy", "standard_thompson",
            "cbsi_full"],
        "comparison_limitations": [
            "fitness/residual/influence proxies use their own parent/donor "
            "selection and are not fully opportunity-matched",
            "repeated crossfit feedback is search-adaptive, not an "
            "independent held-out confirmation",
            "the archive is static in v2; multi-generation reuse requires "
            "a separate preregistered experiment"],
        "provider_calls": 0,
        "pysr_calls": 0,
        "reporting_response_accessed": False,
        "test_or_ood_accessed": False,
        "untouched_confirmation_opened": False,
        "coordinate_count": len(rows),
        "rows": rows,
        "arm_summaries": summaries,
        "strongest_non_cbsi_arm": strongest_baseline_name,
        "cbsi_minus_strongest_non_cbsi": cbsi - strongest,
        "mechanism_gate_passed": mechanism_passed,
        "go_nogo": (
            "MECHANISM_DEVELOPMENT_GO"
            if mechanism_passed and cbsi > 0
            else "MECHANISM_NO_GO"),
        "claim_boundary": freeze["claim_boundary"],
    }
    write_new(
        args.output_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json", result)
    print(json.dumps({
        "go_nogo": result["go_nogo"],
        "strongest_non_cbsi_arm": strongest_baseline_name,
        "cbsi_minus_strongest_non_cbsi":
            result["cbsi_minus_strongest_non_cbsi"],
        "arm_means": {
            arm: value["task_mean_evaluation_gain"]
            for arm, value in summaries.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

