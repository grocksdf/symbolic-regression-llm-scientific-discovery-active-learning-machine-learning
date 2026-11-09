"""Opt-in local diagnostic -> joint edit -> paired learning -> admission adapter.

No provider, dataset access or real experiment on import. Static parents and
regions isolate the one-step mechanism; no Frontier recycling. The old AST
policy and historical runners are untouched. This adapter requires a caller's
frozen design and explicit independent role readers.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.discovery.bank_search_feedback import feedback_from_posterior
from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import StructuralEditingEngine, INVENTION_OPERATORS
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model
from hypothesis_mvp.discovery.predictive_bank_expansion import PredictiveExpansionSession
from hypothesis_mvp.discovery.region_site_attribution import (
    FEATURE_NAMES, AST_FEATURES, ASSOCIATION_FEATURES, build_region_site_contexts, digest,
)
from hypothesis_mvp.discovery.regional_site_operator_policy import RegionalSiteOperatorPolicy, SiteOperatorAction
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support
from scripts.bank_guided_search import _manifest, _ClosedEvaluation
from scripts.cbsi_paired_search import search_scoring_identity
from scripts.frozen_predictive_expansion import FrozenBankComponents, DIRECT_CANDIDATE_POLICY, _publish, _posterior_id
from scripts.stacking_diagnostics import fit_stacking_weights
from scripts.run_cbsi_zero_provider_experiment import CrossfitBankScorer


CONTRACT = "bank-diagnostic-joint-structural-editing-v1"
MODES = ("A_random", "B_local_random", "C_ast_joint", "D_local_joint", "E_attributed_joint", "F_shuffled_joint")


class DirectCrossfitBankScorer(CrossfitBankScorer):
    """Reuse paired fold bookkeeping, but compare E0 with standalone candidates.

    Search and final calibration therefore use the same component family.
    Neither side silently nests the candidate inside another copy of E0.
    """
    def __init__(self, *args, **kwargs):
        self._predictors = {}
        self.model_fits = 0
        super().__init__(*args, **kwargs)

    def _density(self, bank, fold):
        identity = digest(bank)
        if identity not in self._predictors:
            fit = self.selection.development
            model = freeze_expanded_formula_model(bank, n_features=fit.X.shape[1],
                prior=NormalInverseGammaPrior(), exploration_identity=identity,
                coefficient_policy=EXPANDED_FORMULA_POLICY, minimum_supports=1)
            engine = model.engine(model.stable_hash)
            posterior = engine.fit_batch(fit.X, fit.y)
            self._predictors[identity] = engine, posterior
            self.model_fits += 1
        engine, posterior = self._predictors[identity]
        data = self._role(fold)
        values = np.asarray(engine.predictive_logpdf(posterior, data.X, data.y))
        if values.shape != data.y.shape or not np.all(np.isfinite(values)):
            raise ValueError("nonfinite standalone predictive density")
        return values

    def _baseline_density(self, fold):
        if fold not in self._base:
            self.density_evaluations += 1
            self.baseline_density_evaluations += 1
            self._base[fold] = self._density(self.E0, fold)
        return self._base[fold]

    def _candidate_density(self, expression, fold):
        key = expression, fold
        if key not in self.cache:
            self.density_evaluations += 1
            self.candidate_density_evaluations += 1
            self.cache[key] = self._density([{"expression":expression, "source":"regional-joint-edit",
                                            "lineage_id":digest(expression)}], fold)
            self.score_calls += 1
        else:
            self.cache_hits += 1
            self.density_cache_hits += 1
        return self.cache[key]


def project_contexts(snapshot, region_id, mode, seed):
    rows = [dict(r, features=list(r["features"])) for r in snapshot["sites"] if r["region_id"] == region_id]
    if mode not in MODES:
        raise ValueError("unknown context intervention")
    if mode in {"A_random", "C_ast_joint"}:
        for row in rows:
            row["features"] = [v if n in AST_FEATURES else 0. for n, v in zip(FEATURE_NAMES, row["features"])]
    elif mode in {"B_local_random", "D_local_joint"}:
        for row in rows:
            row["features"] = [0. if n in ASSOCIATION_FEATURES else v for n, v in zip(FEATURE_NAMES, row["features"])]
    elif mode == "F_shuffled_joint" and len(rows) > 1:
        # A fixed derangement within parent/region, no response-dependent order.
        order = np.random.default_rng(seed).permutation(len(rows))
        sources = list(order[1:]) + [int(order[0])]
        originals = [tuple(r["features"]) for r in rows]
        for destination, source in zip(order, sources):
            rows[int(destination)]["features"] = [originals[int(source)][i] if name in ASSOCIATION_FEATURES else value
                for i, (name, value) in enumerate(zip(FEATURE_NAMES, originals[int(destination)]))]
    return rows


def _prepare(E0, archive, selection, ids, forbidden, axes):
    fit = selection.development
    if not axes or len(set(axes)) != len(axes) or any(type(a) is not int or not 0 <= a < fit.X.shape[1] for a in axes):
        raise ValueError("frozen diagnostic axes required")
    model = freeze_expanded_formula_model(E0, n_features=fit.X.shape[1], prior=NormalInverseGammaPrior(),
        exploration_identity=digest(E0), coefficient_policy=EXPANDED_FORMULA_POLICY, minimum_supports=1)
    engine = model.engine(model.stable_hash)
    posterior = engine.fit_batch(fit.X, fit.y)
    bank_id = _posterior_id(model, posterior, digest({"rows": ids, "data": fit.fingerprint}))
    feedback = feedback_from_posterior(engine, posterior, fit, search_row_ids=ids,
                                     forbidden_row_ids=forbidden, bank_identity=bank_id)
    weights = np.exp(posterior.log_probabilities)
    locations, scales = [], []
    mass = {}
    for member, weight in zip(posterior.members, weights):
        matrix = engine.design_rows(fit.X, member.structure)
        parameters = engine.conditional_parameters(member)
        locations.append(matrix @ parameters.mean)
        scales.append(parameters.noise_scale / parameters.noise_shape *
                      (1 + np.einsum("ij,jk,ik->i", matrix, parameters.covariance_factor, matrix)))
        support = tuple(sorted(member.structure.basis_terms))
        mass[support] = mass.get(support, 0.) + float(weight)
    location = np.column_stack(locations) @ weights
    # Positive scale surrogate incl. between-member disagreement, not t variance.
    scale = np.sqrt((np.column_stack(scales) + (np.column_stack(locations)-location[:, None])**2) @ weights)
    regions = []
    for axis in axes:
        cut = float(np.median(fit.X[:, axis]))
        for side, mask in (("le", fit.X[:, axis] <= cut), ("gt", fit.X[:, axis] > cut)):
            if np.any(mask):
                regions.append({"id": f"axis{axis}:{side}", "axis": axis, "side": side, "cut": cut})
    editor = StructuralEditingEngine(fit.X.shape[1])
    parents = sorted({c.expression for c in archive.candidates})
    if not parents:
        raise ValueError("empty frozen archive")
    predictions = {p: np.asarray(editor.runtime.predict(p, fit.X)) for p in parents}
    contexts = {}
    for parent in parents:
        v = predictions[parent] - np.mean(predictions[parent])
        redundancy = 0.
        for other, prediction in predictions.items():
            if other == parent: continue
            u = prediction - np.mean(prediction)
            denominator = np.linalg.norm(u) * np.linalg.norm(v)
            if denominator: redundancy = max(redundancy, min(1., abs(float(u @ v / denominator))))
        contexts[parent] = build_region_site_contexts(parent, fit, feedback, regions=regions,
            forbidden_row_ids=forbidden, bank_location=location, bank_scale=scale,
            parent_mass=min(1., mass.get(tuple(sorted(compile_fixed_formula_support(parent, fit.X.shape[1]))), 0.)),
            parent_redundancy=redundancy)
    materials = dict(predictions)
    materials.update({f"x{i}": fit.X[:, i] for i in range(fit.X.shape[1])})
    return feedback, contexts, regions, materials


def run_regional_site_comparison(*, implementation_contract, E0, archive, selection, left, right,
        row_ids, generation_ancestry_ids, design, read_calibration, read_admission, read_reporting,
        task_identity, seed, blocks_per_arm, axes, output_root, operators=INVENTION_OPERATORS,
        frozen_priors=None):
    """A-F equal edit slots; actual work reported, failures never credited zero.

    C is a joint site/operator AST-only ablation, not a replay of the old
    fixed-site baseline. D/E/F share the same local donor schedule. Paired
    search gain is adaptive training feedback, not independent admission.
    """
    if implementation_contract != CONTRACT or type(blocks_per_arm) is not int or blocks_per_arm <= 0:
        raise ValueError("explicit joint-edit contract and positive paired budget required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative seed required")
    operators = tuple(operators)
    # The experiment compares six optional expansions; no post-hoc threshold changes.
    if design.tested_mixture_count < len(MODES):
        raise ValueError("admission design must account for all tested conditions")
    _manifest(selection, left, right, {**row_ids, "evaluation": design.reporting_ids})
    visible_ids = tuple(row_ids["fit"]) + tuple(row_ids["left"]) + tuple(row_ids["right"])
    if (not generation_ancestry_ids or not set(visible_ids).issubset(design.training_ids)
            or not set(generation_ancestry_ids).issubset(design.training_ids)):
        raise ValueError("all upstream/adaptive search rows must be declared training")
    forbidden = design.calibration_ids + design.admission_ids + design.reporting_ids
    priors = frozen_priors or {}
    if set(priors) - set(MODES):
        raise ValueError("unknown arm prior")
    # A shared scoring identity, but independent per-arm online policies/caches.
    scoring_id = search_scoring_identity(E0, selection, left, right, seed,
        {"operators": operators, "budget": {"paired_blocks": blocks_per_arm}})
    policies = {m: RegionalSiteOperatorPolicy(operators=operators, seed=seed,
        scoring_identity=scoring_id, task_identity=task_identity, frozen_prior=priors.get(m),
        intervention_identity=m, utility_identity="standalone-predictive-stacking-crossfit-logscore-v1") for m in MODES}
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    stage = "generation"
    started = time.perf_counter()
    try:
        _publish(root / "PLAN.json", {"contract": CONTRACT, "modes": MODES, "seed": seed,
            "blocks_per_arm": blocks_per_arm, "materialization_slots_per_arm": 2*blocks_per_arm,
            "operators": operators, "axes": axes, "design": asdict(design), "row_ids": row_ids,
            "generation_ancestry_ids": generation_ancestry_ids, "task_identity": task_identity,
            "scoring_identity": scoring_id, "frozen_prior_identities": {m: p.get("identity") for m,p in priors.items()},
            "frontier_enabled": False, "provider_calls": 0})
        feedback, contexts, regions, materials = _prepare(E0, archive, selection, row_ids["fit"], forbidden, axes)
        _publish(root / "CONTEXTS.json", contexts)
        editor = StructuralEditingEngine(selection.development.X.shape[1])
        domains = [selection.development.X, left.X, right.X, selection.acquisition_pool.X]
        parents = sorted(contexts)
        combinations = [(parent, region) for parent in parents for region in regions]
        order = np.random.default_rng(seed).permutation(len(combinations))
        opportunities = [(combinations[int(i)][0], combinations[int(i)][1], contrast)
                         for i in order for contrast in ("operator", "site")]
        _publish(root / "OPPORTUNITIES.json", [{"parent":p, "region":r, "contrast":c}
                                               for p,r,c in opportunities])
        base_supports = {digest(tuple(compile_fixed_formula_support(r["expression"], editor.n_features))) for r in E0}
        generated = {}
        for mode in MODES:
            arm_started = time.perf_counter()
            policy = policies[mode]
            scorer = DirectCrossfitBankScorer(E0, selection, left, right, _ClosedEvaluation(), "shared-regional-scorer", seed)
            seen, candidates, records = set(base_supports), {}, []
            with (root / f"{mode}_ATTEMPTS.jsonl").open("x", encoding="utf-8") as trace:
                for block in range(blocks_per_arm):
                    print(f"  {mode} block {block+1}/{blocks_per_arm}", flush=True)
                    parent, region, contrast = opportunities[block % len(opportunities)]
                    rows = project_contexts(contexts[parent], region["id"], mode, seed + block)
                    mask = selection.development.X[:, region["axis"]] <= region["cut"]
                    if region["side"] == "gt": mask = ~mask
                    if mode in {"A_random", "C_ast_joint"}:
                        ranked = sorted(materials, key=lambda s: digest([seed, s]))
                    else:
                        direction = np.asarray(feedback.location_score)[mask]
                        def rank(s):
                            v = materials[s][mask]
                            denominator = np.linalg.norm(direction) * np.linalg.norm(v)
                            return -abs(float(direction @ v / denominator)) if denominator else 0.
                        ranked = sorted(materials, key=lambda s: (rank(s), s))
                    donor = ranked[(block // len(opportunities)) % len(ranked)]
                    context_id = digest([contexts[parent]["identity"], mode, rows])
                    actions = []
                    for row in rows:
                        allowed = editor.applicable_operators(parent, donor, row["site_path"], seed=seed+block,
                                                             operators=operators, known_support_hashes=seen)
                        actions.extend(SiteOperatorAction(parent, donor, region["id"], tuple(row["site_path"]), op,
                            context_id, tuple(row["features"])) for op in allowed)
                    pair = policy.choose_pair(block, actions, contrast=contrast,
                        mode="random" if mode in {"A_random", "B_local_random"} else "thompson") if actions else None
                    record = {"block": block, "parent": parent, "donor": donor, "region": region["id"],
                              "contrast": contrast, "pair_available": pair is not None,
                              "legal_action_count": len(actions), "slots": [], "learned": False}
                    scored = []
                    for action in pair or ():
                        request = editor.deterministic_request(action.operator, parent, donor, seed=seed+block,
                            site_path=action.site_path, task_identity=f"{task_identity}:{block}",
                            matched_control_of=scored[0][0].identity if scored else None)
                        result = editor.apply(request, registered_domains=domains,
                            known_structural_hashes=set(), known_canonical_hashes=set(), known_support_hashes=seen)
                        gains = None
                        if result.valid and result.novel:
                            # Exceptions are terminal; an invalid scorer contract is not zero gain.
                            gains = tuple(float(v) for v in scorer.crossfit_gain(result.expression))
                            if len(gains) != 2 or not np.all(np.isfinite(gains)):
                                raise ValueError("two finite shared crossfit scores required")
                            candidates.setdefault(result.support_hash, {"expression": result.expression,
                                "source": "regional-joint-edit", "lineage_id": result.lineage_id})
                        record["slots"].append({"action": asdict(action), "request": asdict(request),
                                                "result": result.row(), "fold_gains": gains})
                        scored.append((request, result, gains))
                    distinct = len(scored) == 2 and scored[0][1].support_hash != scored[1][1].support_hash
                    record["distinct_pair_supports"] = distinct
                    if len(scored) == 2 and all(s[2] is not None for s in scored) and distinct:
                        policy.observe_pair(block, primary_request=scored[0][0], control_request=scored[1][0],
                            primary_result=scored[0][1], control_result=scored[1][1], primary_folds=scored[0][2],
                            control_folds=scored[1][2], evidence_id=digest([task_identity, mode, block]), scoring_identity=scoring_id)
                        record["learned"] = True
                    # Pair members share the same pre-pair novelty state.
                    seen.update(s[1].support_hash for s in scored if s[1].valid and s[1].novel)
                    records.append(record)
                    trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    trace.flush()
                    os.fsync(trace.fileno())
            audit = {"mode": mode, "candidates": list(candidates.values()), "policy": policy.audit(),
                     "records": records, "materialization_slots": 2*blocks_per_arm,
                     "materialized_attempts": sum(len(r["slots"]) for r in records),
                     "density_evaluations": scorer.density_evaluations, "optimizer_fits": scorer.optimizer_fits,
                     "model_fits": scorer.model_fits, "component_policy": DIRECT_CANDIDATE_POLICY,
                     "wall_seconds": time.perf_counter()-arm_started, "provider_calls": 0}
            generated[mode] = audit
            _publish(root / f"{mode}_GENERATION.json", audit)
        _publish(root / "GENERATION_FROZEN.json", {m: digest(a) for m,a in generated.items()})
        # No search code runs after this boundary; no independent data can train policies.
        components = {m: FrozenBankComponents(E0, a["candidates"], selection.development,
            fit_row_ids=row_ids["fit"], design=design, component_policy=DIRECT_CANDIDATE_POLICY) for m,a in generated.items()}
        sessions = {m: PredictiveExpansionSession(design, c.component_ids) for m,c in components.items()}
        visible = [selection.development, left, right]
        def read_role(reader, name):
            ids, data = reader()
            if (tuple(ids) != getattr(design, name+"_ids") or len(ids) != len(data.X)
                    or data.role is not DataRole.VALIDATION
                    or any(data.row_fingerprints & d.row_fingerprints for d in visible)):
                raise ValueError("independent role identity/overlap mismatch")
            visible.append(data)
            return ids, data
        stage = "calibration"
        ids, data = read_role(read_calibration, stage)
        for mode, component in components.items():
            values = component.densities(data)
            weights, optimizer = fit_stacking_weights(values, seed=seed, random_starts=4)
            frozen = sessions[mode].freeze(values, weights, row_ids=ids, component_ids=component.component_ids)
            _publish(root / f"{mode}_CALIBRATION_FROZEN.json", {**frozen, "optimizer": optimizer})
        stage = "admission"
        ids, data = read_role(read_admission, stage)
        for mode, component in components.items():
            admission = sessions[mode].admit(component.densities(data), row_ids=ids, component_ids=component.component_ids)
            _publish(root / f"{mode}_ADMISSION_FROZEN.json", admission)
        stage = "reporting"
        ids, data = read_role(read_reporting, stage)
        reports = {m: sessions[m].report(c.densities(data), row_ids=ids, component_ids=c.component_ids) for m,c in components.items()}
        result = {"schema": CONTRACT, "reports": reports, "protocol_complete": True,
                  "wall_seconds": time.perf_counter()-started, "provider_calls": 0, "frontier_enabled": False,
                  "claim_boundary": "Opt-in structural-edit mechanism adapter; no efficacy, transfer or novelty claim from correctness tests."}
        _publish(root / "RESULT.json", result)
        return result
    except Exception as error:
        _publish(root / "FAILURE.json", {"stage": stage, "type": type(error).__name__, "message": str(error),
                                         "protocol_complete": False, "retry_authorized": False})
        raise
