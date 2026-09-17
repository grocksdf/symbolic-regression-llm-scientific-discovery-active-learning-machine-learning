"""Restricted discovery-model transactions; no historical P3M state reuse.

Raw coordinates only. External callers admit already-opened matching responses;
this module never accesses validation/held-out or silently retries an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from hypothesis_mvp.discovery.pcpi_adapter import FrozenDiscoveryModel, FrozenDiscoveryTarget
from hypothesis_mvp.data.oracle import PoolOracle
from .acquisition import predictive_components_for_partition, estimate_class_eig_until_ranked
from .p3j_run_identity import _publish_no_overwrite


def _hash(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("discovery transaction identity collision")
    else:
        _publish_no_overwrite(path, payload)


@dataclass(frozen=True)
class DiscoveryScoringControls:
    minimum_samples: int
    maximum_samples: int
    error_safety_factor: float
    growth_factor: int = 2

    def __post_init__(self):
        if (type(self.minimum_samples) is not int or type(self.maximum_samples) is not int
            or self.minimum_samples < 1 or self.maximum_samples < self.minimum_samples
            or not np.isfinite(self.error_safety_factor) or self.error_safety_factor < 1
            or type(self.growth_factor) is not int or self.growth_factor < 2):
            raise ValueError("invalid discovery scoring controls")


class DiscoveryTransaction:
    """Durable plan/admit/reconstruct coordinator for a frozen Gaussian model."""

    def __init__(self, root: Path, model: FrozenDiscoveryModel,
                 target: FrozenDiscoveryTarget, domain: np.ndarray,
                 controls: DiscoveryScoringControls, *, source_identity: str,
                 query_policy: str = "class_eig", random_seed: int = 0):
        if query_policy not in {"class_eig", "random"}:
            raise ValueError("unsupported registered query policy")
        if type(random_seed) is not int or random_seed < 0:
            raise ValueError("random seed must be a nonnegative integer")
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.target, self.controls = model, target, controls
        self.query_policy, self.random_seed = query_policy, random_seed
        self.domain = np.array(domain, dtype=np.float64, order="C", copy=True)
        if self.domain.ndim != 2 or not len(self.domain) or self.domain.shape[1] != model.n_features:
            raise ValueError("invalid frozen action domain")
        digest = sha256(str(self.domain.shape).encode() + self.domain.tobytes()).hexdigest()
        if digest != target.action_domain_identity or model.stable_hash != target.model_identity:
            raise ValueError("model/action domain crossed frozen target")
        if target.initial_posterior.bank_hash != model.bank.stable_hash:
            raise ValueError("initial posterior crossed frozen bank")
        if not source_identity:
            raise ValueError("source identity required")
        self.domain.setflags(write=False)
        self.engine = model.engine(target.model_identity)
        identity = {"schema": "discovery-gaussian-transaction-v1", "target": target.stable_hash,
                    "controls": vars(controls), "source": source_identity,
                    "query_policy": query_policy, "random_seed": random_seed,
                    "likelihood": "homoscedastic-gaussian-nig", "coordinates": "raw"}
        _publish(self.root / "IDENTITY.json", identity)
        self.identity = _hash(identity)
        self.posterior = target.initial_posterior
        self.prefix_hash = self.identity
        self.receipts: list[dict] = []
        self._recover()

    def _recover(self) -> None:
        paths = sorted(self.root.glob("RECEIPT-*.json"))
        for index, path in enumerate(paths, 1):
            if index > self.target.measurement_budget:
                raise ValueError("receipt prefix exceeds frozen budget")
            if path.name != f"RECEIPT-{index:03d}.json":
                raise ValueError("noncontiguous response prefix")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            decision = self._decision(index)
            expected = self._receipt(decision, receipt["candidate_id"], receipt["action"], receipt["response"])
            if receipt != expected:
                raise ValueError("tampered response receipt")
            self._advance(receipt)

    def _decision(self, index: int) -> dict:
        payload = json.loads((self.root / f"DECISION-{index:03d}.json").read_text(encoding="utf-8"))
        material = {k: v for k, v in payload.items() if k != "hash"}
        if (payload["hash"] != _hash(material) or payload["prefix"] != self.prefix_hash
            or payload["identity"] != self.identity or payload["query"] != index
            or payload["selection_valid"] is not True
            or payload["query_policy"] != self.query_policy
            or payload["certified"] is not (self.query_policy == "class_eig")):
            raise ValueError("invalid durable discovery decision")
        return payload

    def plan(self, candidate_ids: np.ndarray, actions: np.ndarray) -> dict:
        index = len(self.receipts) + 1
        if index > self.target.measurement_budget:
            raise ValueError("frozen measurement budget exhausted")
        ids = np.asarray(candidate_ids)
        values = np.asarray(actions, dtype=float)
        used = {r["candidate_id"] for r in self.receipts}
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
            or values.shape != (len(ids), self.model.n_features)
            or len(set(ids.tolist())) != len(ids) or np.any(ids < 0)
            or any(int(i) in used for i in ids)
            or any(not np.any(np.all(self.domain == row, axis=1)) for row in values)):
            raise ValueError("invalid or repeated candidate domain")
        candidates_hash = _hash({"ids": ids.tolist(), "actions": values.tolist()})
        path = self.root / f"DECISION-{index:03d}.json"
        if path.exists():
            decision = self._decision(index)
            if decision["candidates"] != candidates_hash:
                raise ValueError("resumed candidate domain changed")
            return decision
        if self.query_policy == "random":
            # Query-indexed stream: resuming does not consume a different draw.
            rng = np.random.default_rng(np.random.SeedSequence([self.random_seed, index]))
            leader = int(rng.integers(len(ids)))
            score, errors, certified = None, [], False
        else:
            leader, score, errors = self._rank(values, ids)
            certified = True
        decision = {"identity": self.identity, "prefix": self.prefix_hash, "query": index,
                    "candidates": candidates_hash, "candidate_id": int(ids[leader]),
                    "action": values[leader].tolist(), "score": score,
                    "errors": errors, "certified": certified,
                    "selection_valid": True, "query_policy": self.query_policy}
        decision["hash"] = _hash(decision)
        _publish(path, decision)
        return decision

    def _rank(self, values, ids):
        components = predictive_components_for_partition(self.engine, self.posterior,
                                                         self.target.partition, values)
        c = self.controls
        ranked = estimate_class_eig_until_ranked(components, c.minimum_samples, c.maximum_samples,
            error_safety_factor=c.error_safety_factor, growth_factor=c.growth_factor)
        if not ranked.ranking_certified:
            raise RuntimeError("uncertified class EIG; no response authorized")
        scores = ranked.estimate.scores
        leader = min(range(len(ids)), key=lambda i: (-float(scores[i]), int(ids[i])))
        return leader, float(scores[leader]), ranked.estimate.error_bounds.tolist()

    def _receipt(self, decision: dict, candidate_id: int, action, response: float) -> dict:
        if (isinstance(candidate_id, bool) or candidate_id in {r["candidate_id"] for r in self.receipts}
            or candidate_id != decision["candidate_id"] or list(action) != decision["action"]
            or not np.isfinite(response)):
            raise ValueError("response does not match durable selection")
        payload = {"query": decision["query"], "decision": decision["hash"],
                   "prefix": self.prefix_hash, "candidate_id": int(candidate_id),
                   "action": list(action), "response": float(response)}
        payload["hash"] = _hash(payload)
        return payload

    def _advance(self, receipt: dict) -> None:
        self.posterior = self.engine.update_one(self.posterior, np.asarray(receipt["action"]), receipt["response"])
        self.receipts.append(receipt)
        self.prefix_hash = receipt["hash"]

    def admit(self, candidate_id: int, action: np.ndarray, response: float) -> None:
        index = len(self.receipts) + 1
        if index > self.target.measurement_budget:
            raise ValueError("frozen measurement budget exhausted")
        decision = self._decision(index)
        receipt = self._receipt(decision, candidate_id, action, response)
        # Validate update before publication; crash afterwards reconstructs from receipt.
        self.engine.update_one(self.posterior, np.asarray(action), float(response))
        _publish(self.root / f"RECEIPT-{index:03d}.json", receipt)
        self._advance(receipt)

    def execute_measured_query(self, oracle: PoolOracle, candidate_ids: np.ndarray,
                               actions: np.ndarray) -> dict:
        """Select first, reveal exactly its indexed raw measured-pool response.

        Restricted to the immutable in-memory PoolOracle, not a physical lab
        instrument. Crash before receipt may re-read the same static label;
        recovery never counts it as a second observation or reveals a new ID.
        """
        if type(oracle) is not PoolOracle:
            raise TypeError("only the registered static measured-pool oracle is supported")
        decision = self.plan(candidate_ids, actions)
        selected = decision["candidate_id"]
        if selected >= len(oracle.X_pool) or oracle.X_pool[selected].tolist() != decision["action"]:
            raise ValueError("oracle covariate coordinates do not match selection")
        revealed_x, revealed_y, revealed_ids = oracle.acquire_indices(
            np.asarray([decision["candidate_id"]], dtype=int))
        if (revealed_ids.tolist() != [decision["candidate_id"]]
            or revealed_x.shape != (1, self.model.n_features) or revealed_y.shape != (1,)
            or revealed_x[0].tolist() != decision["action"]):
            raise ValueError("oracle coordinates do not match raw frozen selection")
        self.admit(int(revealed_ids[0]), revealed_x[0], float(revealed_y[0]))
        return self.receipts[-1]

    def run_measured_pool(self, oracle: PoolOracle, candidate_ids: np.ndarray,
                          actions: np.ndarray) -> dict:
        """Execute/recover the frozen budget; no retry or fallback on NO-GO."""
        ids, values = np.asarray(candidate_ids), np.asarray(actions, dtype=float)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or values.shape != (len(ids), self.model.n_features)
            or len(set(ids.tolist())) != len(ids) or len(ids) < self.target.measurement_budget):
            raise ValueError("invalid full measured-pool candidate contract")
        pool_contract = {"identity": self.identity, "ids": ids.tolist(), "actions": values.tolist()}
        _publish(self.root / "POOL_CONTRACT.json", pool_contract)
        for receipt in self.receipts:
            matches = np.flatnonzero(ids == receipt["candidate_id"])
            if len(matches) != 1 or values[int(matches[0])].tolist() != receipt["action"]:
                raise ValueError("recovered response outside registered pool")
        while len(self.receipts) < self.target.measurement_budget:
            used = {r["candidate_id"] for r in self.receipts}
            remaining = np.asarray([int(i) not in used for i in ids], dtype=bool)
            self.execute_measured_query(oracle, ids[remaining], values[remaining])
        manifest = {"schema": "discovery-gaussian-measured-run-v1", "identity": self.identity,
                    "query_policy": self.query_policy,
                    "target": self.target.stable_hash, "prefix": self.prefix_hash,
                    "receipt_hashes": [r["hash"] for r in self.receipts],
                    "completed_queries": len(self.receipts), "heldout_opened": False,
                    "protocol_complete": True, "efficacy_demonstrated": False}
        _publish(self.root / "RUN_MANIFEST.json", manifest)
        return manifest
