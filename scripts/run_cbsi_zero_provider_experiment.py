"""Run the frozen CB-SI provider-free development mechanism experiment."""
from __future__ import annotations

import argparse
import csv
from typing import NamedTuple
from hashlib import sha256
import json
import math
import subprocess
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
    FeasibilityAwareContextualPolicy,
    INVENTION_OPERATORS,
    StructuralEditingEngine,
    counterfactual_edit_credit,
    validate_matched_edit_pair,
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
# Only the actual HDF5 experiment imports data readers.  Importing this
# module for response-free correctness tests must not require pyarrow, PySR,
# provider clients or archived experiment runtime dependencies.
from gap_minimal_harness_contract import identity
from stacking_diagnostics import fit_stacking_weights, mixture_log_density


ARM_MODES = {
    "blind_random": "random",
    "single_fitness_proxy": "fitness",
    "residual_proxy": "residual",
    "influence_proxy": "influence",
    "bank_greedy": "greedy",
    "standard_thompson": "thompson_raw",
    "cbsi_full": "thompson_matched",
    "cbsi_feasible": "thompson_feasible",
    "cbsi_frontier": "thompson_frontier",
}

# Schema v3 describes the artifact layout, not permission to run changed
# search semantics. Old frozen protocols must never silently pick up repairs.
IMPLEMENTATION_CONTRACT = "cbsi-parent-marginal-search-v1"


def verify_implementation_contract(freeze):
    if freeze.get("implementation_contract") != IMPLEMENTATION_CONTRACT:
        raise ValueError(
            "CB-SI implementation changed; historical freezes require their "
            "original source, and new execution requires a reviewed new freeze")


def parent_relative_gain(primary_folds, parent_folds):
    """Search increment attributable to this edit, not inherited parent gain."""
    primary = np.asarray(primary_folds, dtype=float)
    parent = np.asarray(parent_folds, dtype=float)
    if (primary.ndim != 1 or len(primary) < 2 or parent.shape != primary.shape
            or not np.all(np.isfinite(primary))
            or not np.all(np.isfinite(parent))):
        raise ValueError("parent marginal gain needs finite aligned search folds")
    with np.errstate(over="raise", invalid="raise"):
        increments = primary - parent
        gain = float(np.mean(increments))
    if not math.isfinite(gain):
        raise ValueError("nonfinite parent marginal gain")
    return gain, tuple(float(value) for value in increments)


def frontier_promotion_reason(raw_gain, marginal_gain, depth, max_depth):
    if marginal_gain is None or not math.isfinite(marginal_gain):
        return "parent_increment_unavailable"
    if not math.isfinite(raw_gain) or raw_gain <= 0:
        return "no_gain_over_initial_bank"
    if marginal_gain <= 0:
        return "no_gain_over_parent"
    if depth >= max_depth:
        return "depth_budget_exhausted"
    return "promoted"


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
    """Search-only folds plus a lazily opened, separately cached evaluation.

    Cache keys remain exact expressions within this scorer's fixed bank,
    roles, namespace and seed. Equal supports of two new candidates are not
    assumed to imply interchangeable model provenance or priors.
    """

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
        self._crossfit_cache = {}
        self._evaluation_cache = {}
        self._support_cache = {}
        self._selection_frozen = False
        self._selected_expression = None
        self.selection_identity = None
        # Preserve the historical successful-candidate counter while exposing
        # all attempted density/optimizer work, including the baseline bank.
        self.score_calls = 0
        self.density_evaluations = 0
        self.baseline_density_evaluations = 0
        self.candidate_density_evaluations = 0
        self.optimizer_fits = 0
        self.cache_hits = 0
        self.density_cache_hits = 0
        self.crossfit_cache_hits = 0
        self.evaluation_cache_hits = 0
        self.e0_supports = {
            tuple(compile_fixed_formula_support(
                str(row["expression"]),
                selection.development.X.shape[1]))
            for row in self.E0}
        self._base = {}
        self._baseline_density("left")
        self._baseline_density("right")

    def _role(self, fold):
        return {"left": self.left, "right": self.right,
                "evaluation": self.evaluation}[fold]

    def _baseline_density(self, fold):
        if fold not in self._base:
            role_value = self._role(fold)
            self.density_evaluations += 1
            self.baseline_density_evaluations += 1
            self._base[fold] = density(
                self.E0, self.selection, role_value.X, role_value.y,
                {"namespace": self.namespace, "bank": "E0", "fold": fold})
        return self._base[fold]

    def _candidate_density(self, expression: str, fold: str):
        key = (expression, fold)
        if key in self.cache:
            self.cache_hits += 1
            self.density_cache_hits += 1
            return self.cache[key]
        if expression not in self._support_cache:
            self._support_cache[expression] = tuple(compile_fixed_formula_support(
                expression, self.selection.development.X.shape[1]))
        support = self._support_cache[expression]
        if support in self.e0_supports:
            # The bank contains unique supports. An existing support adds no
            # component and leaves the baseline mixture and prior unchanged.
            values = self._baseline_density(fold)
            self.cache[key] = values
            return values
        role_value = self._role(fold)
        row = {
            "expression": expression,
            "source": "cbsi:structural_edit",
            "origin": "archive-edit",
            "lineage_id": "cbsi-" + sha256(
                expression.encode()).hexdigest()[:24],
        }
        self.density_evaluations += 1
        self.candidate_density_evaluations += 1
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
        if self._selection_frozen:
            raise ValueError("CB-SI search is closed after selection is frozen")
        if expression in self._crossfit_cache:
            self.cache_hits += 1
            self.crossfit_cache_hits += 1
            return self._crossfit_cache[expression]
        left_candidate = self._candidate_density(expression, "left")
        right_candidate = self._candidate_density(expression, "right")
        self.optimizer_fits += 1
        left_to_right, _, _ = self._mixture_gain(
            self._base["left"], left_candidate,
            self._base["right"], right_candidate, self.seed + 1)
        self.optimizer_fits += 1
        right_to_left, _, _ = self._mixture_gain(
            self._base["right"], right_candidate,
            self._base["left"], left_candidate, self.seed + 2)
        result = (left_to_right, right_to_left)
        self._crossfit_cache[expression] = result
        return result

    def freeze_selection(self, expression: str | None) -> str:
        """Seal one searched expression (or baseline fallback) before evaluation."""
        if self._selection_frozen:
            if expression != self._selected_expression:
                raise ValueError("CB-SI selected expression is already frozen")
            return self.selection_identity
        if expression is not None and (
                not isinstance(expression, str)
                or expression not in self._crossfit_cache):
            raise ValueError("CB-SI selection requires a scored search expression")
        self.selection_identity = identity({
            "schema": "cbsi-search-selection-v1",
            "namespace": self.namespace,
            "seed": self.seed,
            "E0": self.E0,
            "selected_expression": expression,
        })
        self._selected_expression = expression
        self._selection_frozen = True
        return self.selection_identity

    def evaluation_gain(self, expression: str):
        from copy import deepcopy

        if (not self._selection_frozen or self._selected_expression is None
                or expression != self._selected_expression):
            raise ValueError("CB-SI evaluation requires the frozen selected expression")
        if expression in self._evaluation_cache:
            self.cache_hits += 1
            self.evaluation_cache_hits += 1
            return deepcopy(self._evaluation_cache[expression])
        left_candidate = self._candidate_density(expression, "left")
        right_candidate = self._candidate_density(expression, "right")
        # Evaluation covariates and responses are first read here, after the
        # caller has frozen the selected candidate. Search never opens them.
        evaluation_base = self._baseline_density("evaluation")
        eval_candidate = self._candidate_density(expression, "evaluation")
        calibration_base = np.concatenate((
            self._base["left"], self._base["right"]))
        calibration_candidate = np.concatenate((
            left_candidate, right_candidate))
        self.optimizer_fits += 1
        gain, weights, audit = self._mixture_gain(
            calibration_base, calibration_candidate,
            evaluation_base, eval_candidate, self.seed + 3)
        result = (gain, weights.tolist(), audit)
        self._evaluation_cache[expression] = deepcopy(result)
        return result


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


class FrontierParent(NamedTuple):
    expression: str
    structural_hash: str
    lineage_id: str
    generation: int


def select_frontier_parent(frontier, block, default_parent, *, max_depth=None):
    """Exploit search-only descendants on alternating blocks, no retries."""
    expandable = [item for item in frontier
                  if max_depth is None or item["parent"].generation < max_depth]
    if not expandable or block % 2 == 0:
        return default_parent, False
    best = min(expandable, key=lambda item: (
        -item["search_gain"], item["parent"].generation,
        item["parent"].structural_hash))
    return best["parent"], True


def run_arm(arm, freeze, coordinate, archive, E0, selection,
            left, right, evaluation, context, domains,
            shared_opportunities):
    started = time.perf_counter()
    mode = ARM_MODES[arm]
    operators = tuple(freeze["operators"])
    seed = int(coordinate["seed"]) + int(
        sha256(arm.encode()).hexdigest()[:8], 16)
    scorer_seed = int(coordinate["seed"]) + int(sha256(
        f"{coordinate['family']}:{coordinate['task']}:shared-scorer".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    editor = StructuralEditingEngine(selection.development.X.shape[1])
    scorer = CrossfitBankScorer(
        E0, selection, left, right, evaluation,
        f"{coordinate['task']}/{coordinate['seed']}/shared-scorer",
        scorer_seed)
    policy = (FeasibilityAwareContextualPolicy(
        operators=tuple(op for op in operators if op in INVENTION_OPERATORS),
        seed=seed, credit_weight=float(
            freeze["budget"]["contextual_credit_weight"]))
        if mode in {"thompson_feasible", "thompson_frontier"}
        else BayesianOperatorPolicy(operators=operators, seed=seed))
    # Archive hashes use a support-identity namespace, NOT EquationDAG's
    # structural hash namespace.  Recompute all three compatible identities.
    seen_structures = set()
    seen_canonicals = set()
    seen_supports = set()
    for original in archive.candidates:
        dag = editor.runtime.dag(original.expression)
        seen_structures.add(dag.structural_hash)
        seen_canonicals.add(dag.canonical_hash)
        seen_supports.add(sha256(json.dumps(tuple(
            compile_fixed_formula_support(
                original.expression, editor.n_features)),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    evaluations = int(freeze["budget"]["edit_evaluations_per_arm"])
    if evaluations <= 0 or evaluations % 2:
        raise ValueError("CB-SI edit budget must be positive and paired")
    records = []
    eligible = []
    frontier = []
    lineage_depth = {}
    shared_modes = {"random", "greedy", "thompson_raw",
                    "thompson_matched", "thompson_feasible",
                    "thompson_frontier"}
    proposal_slots = evaluations // 2
    for block in range(proposal_slots):
        shared = shared_opportunities[block]
        from_frontier = False
        if mode in shared_modes:
            parent, donor = shared["parent"], shared["donor"]
            site_path = shared["site_path"]
            if mode == "thompson_frontier":
                parent, from_frontier = select_frontier_parent(
                    frontier, block, parent,
                    max_depth=int(freeze["budget"]["frontier_max_depth"]))
                if from_frontier:
                    sites = editor.editable_paths(parent.expression)
                    # Adaptive trajectories may have a different tree shape.
                    # Use a stable path draw, not a path from the old parent.
                    token = sha256(
                        f"{shared['opportunity_id']}:frontier:"
                        f"{parent.structural_hash}".encode()).digest()
                    site_path = sites[
                        int.from_bytes(token[:8], "big") % len(sites)]
        else:
            parent, donor = choose_parent_and_donor(
                mode, context, rng, block)
            site_path = None
        parent_depth = int(getattr(parent, "generation", 0))
        opportunity_seed = (
            shared["opportunity_seed"] if mode in shared_modes
            else seed + block)
        task_identity = (
            f"{coordinate['task']}:{coordinate['seed']}:"
            f"{shared['opportunity_id'] if mode in shared_modes else arm}:"
            f"{block}")
        preflight_eligible = None
        site_context = None
        if mode in {"thompson_feasible", "thompson_frontier"}:
            site_context = policy.site_context(parent.expression, site_path)
            preflight_eligible = editor.applicable_operators(
                parent.expression, donor, site_path,
                seed=opportunity_seed, operators=policy.operators,
                known_support_hashes=seen_supports)
            # No rescue or re-draw. If preflight finds no actionable operator,
            # the nominal slot is recorded as a failed attempt.
            selection_pool = (
                preflight_eligible if preflight_eligible else policy.operators)
            operator = policy.choose(
                block, context=site_context,
                eligible_operators=selection_pool)
        elif mode in {"greedy", "thompson_raw", "thompson_matched"}:
            operator = policy.choose(
                block, mode="greedy" if mode == "greedy" else "thompson")
        else:
            operator = fixed_operator(mode, operators, block, rng)
        primary_request = editor.deterministic_request(
            operator, parent.expression, donor,
            seed=opportunity_seed,
            task_identity=task_identity, site_path=site_path)
        control_request = None
        control = None
        control_planning_failure = None
        try:
            control_request = editor.matched_control_request(primary_request)
        except ValueError as error:
            control_planning_failure = str(error)
        primary = editor.apply(
            primary_request, registered_domains=domains,
            known_structural_hashes=seen_structures,
            known_canonical_hashes=seen_canonicals,
            known_support_hashes=seen_supports)
        if control_request is not None:
            control = editor.apply(
                control_request, registered_domains=domains,
                known_structural_hashes=seen_structures,
                known_canonical_hashes=seen_canonicals,
                known_support_hashes=seen_supports)
        # A valid pair needs distinct, executable novel supports and identical
        # request lineage. Failing control is NEVER imputed as zero reward.
        primary_usable = bool(primary.valid and primary.novel)
        control_usable = bool(control is not None and control.valid
                              and control.novel)
        pair_identity_failure = None
        if primary_usable and control_usable:
            try:
                validate_matched_edit_pair(
                    primary_request, control_request, primary, control)
            except ValueError as error:
                pair_identity_failure = str(error)
                control_usable = False
        parent_folds = None
        primary_folds = None
        control_folds = None
        credit = None
        scoring_failure = None
        raw_gain = None
        marginal_gain = None
        marginal_folds = None
        parent_scoring_failure = None
        promotion_reason = "not_frontier_or_unscored"
        if primary_usable:
            try:
                primary_folds = scorer.crossfit_gain(str(primary.expression))
                raw_gain = float(np.mean(primary_folds))
                if not math.isfinite(raw_gain):
                    raise ValueError("nonfinite crossfit Bank gain")
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                scoring_failure = f"{type(error).__name__}:{error}"
                primary_usable = False
                primary_folds = None
                raw_gain = None
        if primary_usable and mode in {"thompson_feasible", "thompson_frontier"}:
            try:
                parent_folds = scorer.crossfit_gain(parent.expression)
                marginal_gain, marginal_folds = parent_relative_gain(
                    primary_folds, parent_folds)
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                parent_scoring_failure = f"{type(error).__name__}:{error}"
        if primary_usable:
            seen_structures.add(str(primary.structural_hash))
            seen_canonicals.add(str(primary.canonical_hash))
            seen_supports.add(str(primary.support_hash))
            eligible.append({
                "result": primary, "search_gain": raw_gain,
                "credit": None,
            })
            if mode in {"greedy", "thompson_raw"}:
                policy.update(
                    operator, raw_gain, reward_kind="raw_bank_gain",
                    audit={"paired_control_valid": control_usable})
            if mode == "thompson_frontier":
                promotion_reason = frontier_promotion_reason(
                    raw_gain, marginal_gain, parent_depth,
                    int(freeze["budget"]["frontier_max_depth"]))
            if promotion_reason == "promoted":
                descendant = FrontierParent(
                    expression=str(primary.expression),
                    structural_hash=str(primary.structural_hash),
                    lineage_id=str(primary.lineage_id),
                    generation=parent_depth + 1)
                frontier.append({
                    "parent": descendant,
                    "search_gain": raw_gain,
                    "parent_relative_gain": marginal_gain,
                    "progenitor": getattr(parent, "lineage_id", None),
                })
                frontier.sort(key=lambda row: (
                    -row["search_gain"], row["parent"].structural_hash))
                frontier = frontier[:int(freeze["budget"]["frontier_size"])]
                lineage_depth[descendant.lineage_id] = descendant.generation
        if primary_usable and control_usable:
            try:
                parent_folds = scorer.crossfit_gain(parent.expression)
                control_folds = scorer.crossfit_gain(str(control.expression))
                parent_complexity = float(
                    editor.runtime.dag(parent.expression).complexity)
                credit = counterfactual_edit_credit(
                    operator, primary_folds, parent_folds,
                    (control_folds,),
                    primary_cost=float(primary.complexity) - parent_complexity,
                    control_costs=(
                        float(control.complexity) - parent_complexity,),
                    cost_weight=float(
                        freeze["budget"]["complexity_cost_weight"]))
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                scoring_failure = f"{type(error).__name__}:{error}"
                credit = None
        if credit is not None:
            eligible[-1]["credit"] = credit.credit
            if mode == "thompson_matched":
                policy.update(
                    operator, credit.credit,
                    reward_kind="matched_counterfactual_credit",
                    audit={"credit": credit.row()})
        if mode in {"thompson_feasible", "thompson_frontier"}:
            # Search credit is local to this edit. The final protected-bank
            # candidate is still selected by its absolute gain against E0.
            observable = primary_usable and marginal_gain is not None
            policy.observe(
                operator=operator, context=site_context,
                novel=observable, raw_gain=marginal_gain if observable else None,
                pair_valid=observable and credit is not None,
                matched_credit=credit.credit
                if observable and credit is not None else None)
        records.append({
            "block": block,
            "opportunity_id": shared["opportunity_id"]
            if mode in shared_modes else None,
            "shared_opportunity": mode in shared_modes and not from_frontier,
            "used_frontier_parent": from_frontier,
            "parent_generation": parent_depth,
            "candidate_generation": parent_depth + 1,
            "parent_lineage_id": getattr(parent, "lineage_id", None),
            "site_context": site_context,
            "planning_eligible_operators": (
                list(preflight_eligible) if preflight_eligible is not None
                else None),
            "planning_no_eligible_action": preflight_eligible == (),
            "primary": primary.row(),
            "primary_request_identity": primary_request.identity,
            "control": control.row() if control is not None else None,
            "control_request_identity": (
                control_request.identity if control_request is not None else None),
            "control_matched_primary_identity": (
                control_request.matched_control_of
                if control_request is not None else None),
            "control_planning_failure": control_planning_failure,
            "pair_identity_failure": pair_identity_failure,
            "scoring_failure": scoring_failure,
            "parent_scoring_failure": parent_scoring_failure,
            "parent_relative_gain": marginal_gain,
            "parent_relative_fold_gains": marginal_folds,
            "frontier_promotion_reason": promotion_reason,
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
                "primary_invalid_or_non_novel_or_unscored"
                if not primary_usable else
                "control_absent_or_invalid_or_non_novel_or_unscored"),
            "credit": credit.row() if credit is not None else None,
        })
    selected = (
        max(eligible, key=lambda row: (
            row["search_gain"],
            row["credit"] if row["credit"] is not None else -math.inf,
            str(row["result"].structural_hash)))
        if eligible else None)
    selection_expression = (
        str(selected["result"].expression)
        if selected is not None and selected["search_gain"] > 0 else None)
    # Commit the one selected formula before touching evaluation responses.
    if hasattr(scorer, "freeze_selection"):
        scorer.freeze_selection(selection_expression)
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
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "mode": mode,
        "shared_scorer_seed": scorer_seed,
        "edit_evaluation_budget": evaluations,
        "primary_attempt_count": len(records),
        "matched_control_count": sum(
            row["counterfactual_pair_valid"] for row in records),
        "reserved_control_slots": len(records),
        "attempted_control_count": sum(
            row["control"] is not None for row in records),
        "materialized_control_count": sum(
            row["control"] is not None for row in records),
        "invalid_pair_count": sum(
            not row["counterfactual_pair_valid"] for row in records),
        "shared_opportunities": (mode in shared_modes
                                 and mode != "thompson_frontier"),
        "valid_primary_count": sum(
            row["primary"]["valid"] for row in records),
        "novel_primary_count": sum(
            row["primary"]["novel"] for row in records),
        "valid_control_count": sum(
            bool(row["control"] and row["control"]["valid"])
            for row in records),
        "frontier_parent_reuses": sum(
            row["used_frontier_parent"] for row in records),
        "frontier_max_observed_depth": max(
            lineage_depth.values(), default=0),
        "max_attempted_candidate_generation": max(
            (row["candidate_generation"] for row in records), default=0),
        "final_frontier_size": len(frontier),
        "selected": selected_row,
        "selected_search_gain": (
            selected["search_gain"] if selected else 0.0),
        "selected_counterfactual_credit": (
            selected["credit"] if selected else None),
        "independent_development_evaluation_gain": evaluation_gain,
        "evaluation_stacking_weights": weights,
        "evaluation_optimizer": optimizer,
        "score_calls": scorer.score_calls,
        "compute": {name: getattr(scorer, name, 0) for name in (
            "density_evaluations", "baseline_density_evaluations",
            "candidate_density_evaluations", "optimizer_fits", "cache_hits")},
        "selection_identity": getattr(scorer, "selection_identity", None),
        "operator_preflight_count": (
            len(policy.operators) * proposal_slots
            if mode in {"thompson_feasible", "thompson_frontier"} else 0),
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


def paired_task_contrast(rows, method, baseline, seed, resamples):
    """Paired *task* bootstrap; multiple seeds are not independent tasks."""
    grouped = {}
    for row in rows:
        key = str(row["task"])
        grouped.setdefault(key, []).append(row)
    diffs = []
    for task in sorted(grouped):
        paired = grouped[task]
        if len({int(row["seed"]) for row in paired}) != len(paired):
            raise ValueError("duplicate task/seed in CB-SI contrast")
        diffs.append(float(np.mean([
            float(row["arms"][method]["independent_development_evaluation_gain"])
            - float(row["arms"][baseline]["independent_development_evaluation_gain"])
            for row in paired])))
    if len(diffs) < 4:
        raise ValueError("at least four distinct tasks required for v3")
    return float(np.mean(diffs)), bootstrap(diffs, seed, resamples), len(diffs)


def verify_source_provenance(freeze):
    """Disallow stale/dirty v3 sources before any experiment data is opened."""
    sources = freeze.get("source_v3")
    if not isinstance(sources, dict) or set(sources) != {"mainline", "benchmark"}:
        raise ValueError("v3 source provenance absent")
    for label, path in (("mainline", MAINLINE), ("benchmark", ROOT)):
        frozen = sources[label]
        if Path(frozen.get("root", "")).resolve() != path.resolve():
            raise ValueError(f"wrong {label} repository root")
        def git(*arguments):
            return subprocess.run(
                ["git", *arguments], cwd=path, check=True, text=True,
                encoding="utf-8", stdout=subprocess.PIPE).stdout.strip()
        if (frozen.get("commit") != git("rev-parse", "HEAD")
                or frozen.get("tree") != git("rev-parse", "HEAD^{tree}")
                or bool(git("status", "--short", "--untracked-files=no"))):
            raise ValueError(f"stale or dirty {label} source provenance")


def ensure_development_roles_separate(*roles):
    """Check labelled role row identities before selecting edit operators."""
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            if left.row_fingerprints & right.row_fingerprints:
                raise ValueError("CB-SI development roles overlap")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("CB-SI output root already exists")
    freeze = read(args.freeze)
    verify_implementation_contract(freeze)
    if (freeze.get("schema")
            != "cbsi-zero-provider-development-freeze-v3"
            or freeze.get("execution_authorized") is not True
            or freeze.get("provider_calls_authorized") is not False
            or freeze.get("expensive_engine_runs_authorized") is not False
            or freeze.get("protocol_id") != IMPLEMENTATION_CONTRACT
            or set(freeze.get("arms", ())) != set(ARM_MODES)
            or len(freeze.get("arms", ())) != len(ARM_MODES)
            or not 0 <= float(freeze.get("budget", {}).get(
                "contextual_credit_weight", float("nan"))) <= 2
            or not 1 <= int(freeze.get("budget", {}).get(
                "frontier_max_depth", 0)) <= 4
            or not 1 <= int(freeze.get("budget", {}).get(
                "frontier_size", 0)) <= 16
            or float(freeze.get("budget", {}).get(
                "frontier_min_search_gain", float("nan"))) != 0.0
            or not isinstance(freeze.get("budget", {}).get(
                "complexity_cost_weight"), (int, float))
            or not math.isfinite(float(freeze["budget"][
                "complexity_cost_weight"]))
            or float(freeze["budget"]["complexity_cost_weight"]) < 0):
        raise ValueError("invalid CB-SI v3 zero-provider freeze")
    verify_source_provenance(freeze)
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

    # HDF5/parquet and original generation helpers are imported only after
    # the reviewed freeze has been validated.  Offline correctness tests can
    # import the pure runner without expensive optional dependencies.
    from run_gap_candidate_marginal_value import (
        _gap_admission_role, _generation_roles)
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
        ensure_development_roles_separate(
            selection.development, left, right, evaluation)
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
    cbsi = summaries["cbsi_frontier"]["task_mean_evaluation_gain"]
    strongest_baseline_name = max(
        (arm for arm in freeze["arms"]
         if arm not in {"cbsi_full", "cbsi_feasible", "cbsi_frontier"}),
        key=lambda arm: summaries[arm]["task_mean_evaluation_gain"])
    strongest = summaries[strongest_baseline_name][
        "task_mean_evaluation_gain"]
    contrasts = {}
    for method, baseline in (
            ("cbsi_feasible", "cbsi_full"),
            ("cbsi_frontier", "cbsi_feasible"),
            ("cbsi_frontier", "cbsi_full"),
            ("cbsi_frontier", "bank_greedy")):
        effect, interval, task_count = paired_task_contrast(
            rows, method, baseline,
            freeze["analysis"]["bootstrap_seed"],
            freeze["analysis"]["bootstrap_resamples"])
        contrasts[f"{method}_minus_{baseline}"] = {
            "task_mean_difference": effect,
            "task_paired_ci95": interval,
            "task_count": task_count,
            "development_ci_excludes_zero_positive": interval[0] > 0,
        }
    # This is a preregistered *development* screening rule, not proof of
    # independent efficacy. Check both mechanism components and Bank-greedy.
    mechanism_passed = all(
        contrasts[name]["development_ci_excludes_zero_positive"]
        for name in (
            "cbsi_feasible_minus_cbsi_full",
            "cbsi_frontier_minus_cbsi_feasible",
            "cbsi_frontier_minus_bank_greedy"))
    result = {
        "schema": "cbsi-zero-provider-development-result-v3",
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "protocol_complete": True,
        "matched_opportunity_policy_arms": [
            "blind_random", "bank_greedy", "standard_thompson",
            "cbsi_full", "cbsi_feasible"],
        "comparison_limitations": [
            "CB-SI frontier changes parent trajectories after a positive "
            "search-side gain, so its adaptive opportunities are NOT "
            "matched to the static-core arms",
            "response-free preflight checks are planning operations and "
            "must be included when comparing measured compute, whereas "
            "nominal edit slots alone do not prove equal CPU budgets",
            "fitness/residual/influence proxies use their own parent/donor "
            "selection and are not fully opportunity-matched",
            "repeated crossfit feedback is search-adaptive, not an "
            "independent held-out confirmation",
            "the new frontier arm is a development-only experimental "
            "mechanism and cannot support discovery or truth claims"],
        "provider_calls": 0,
        "pysr_calls": 0,
        "reporting_response_accessed": False,
        "test_or_ood_accessed": False,
        "untouched_confirmation_opened": False,
        "coordinate_count": len(rows),
        "rows": rows,
        "arm_summaries": summaries,
        "preregistered_paired_development_contrasts": contrasts,
        "new_method_arm": "cbsi_frontier",
        "novelty_claim_authorized": False,
        "formal_confirmation_authorized": False,
        "strongest_legacy_baseline_arm": strongest_baseline_name,
        "cbsi_frontier_minus_strongest_legacy_baseline": cbsi - strongest,
        "mechanism_gate_passed": mechanism_passed,
        "go_nogo": (
            "V3_DEVELOPMENT_SCREEN_GO"
            if mechanism_passed and cbsi > max(0.0, strongest)
            else "V3_DEVELOPMENT_SCREEN_NO_GO"),
        "claim_boundary": freeze["claim_boundary"],
    }
    write_new(
        args.output_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json", result)
    print(json.dumps({
        "go_nogo": result["go_nogo"],
        "strongest_legacy_baseline_arm": strongest_baseline_name,
        "cbsi_frontier_minus_strongest_legacy_baseline":
            result["cbsi_frontier_minus_strongest_legacy_baseline"],
        "v3_paired_contrasts": contrasts,
        "arm_means": {
            arm: value["task_mean_evaluation_gain"]
            for arm, value in summaries.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

