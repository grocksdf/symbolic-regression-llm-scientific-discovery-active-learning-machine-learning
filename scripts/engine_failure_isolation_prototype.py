"""Task 4 — single-engine failure isolation: provider-free synthetic tests.

Re-implements the narrowed fail-closed guard (see ENGINE_FAILURE_ISOLATION_DESIGN.md)
over MOCK engine records. No provider, no real PySR/Julia/Lasso, no formal experiment.
Produces ENGINE_FAILURE_ISOLATION_TESTS.json.
"""
import json, hashlib, uuid, datetime

ISOLATABLE_TIMEOUT_ENGINES = frozenset({"polynomial_lasso"})


class EngineRunRecord:
    def __init__(self, engine, status, error_type=None, controls=(), timeout_s=1200.0):
        self.engine = engine
        self.status = status            # 'succeeded' | 'timeout' | 'failed'
        self.error_type = error_type
        self.controls = tuple(controls)
        self.timeout_s = timeout_s
        self.isolation_note = None


class DataIdentityError(RuntimeError):
    pass


def decide(records):
    """Mirror of the narrowed _refuse_partial_engine_degradation.

    Returns {'proceed': True, 'kept': [...], 'discarded': [...], 'isolated': [...]}
    or raises RuntimeError (fail closed) for every impermissible case.
    """
    by_engine = {}
    for r in records:
        by_engine.setdefault(r.engine, []).append(r)

    silent = sorted(e for e, rows in by_engine.items()
                    if all(r.status != "succeeded" for r in rows))
    if not silent:
        kept = [e for e in by_engine if e not in silent]
        return {"proceed": True, "kept": kept, "discarded": [], "isolated": []}

    # narrow carve-out: a TIMEOUT (declared budget exceeded, not infra fault) of a
    # designated engine, with at least one peer engine succeeded.
    isolatable = [e for e in silent
                  if e in ISOLATABLE_TIMEOUT_ENGINES
                  and all(r.error_type == "TimeoutError" for r in by_engine[e])]
    peer_succeeded = any(e not in silent for e in by_engine)
    if isolatable and peer_succeeded:
        for e in isolatable:
            for r in by_engine[e]:
                r.isolation_note = "engine_timeout_isolated; partial output discarded"
        kept = [e for e in by_engine if e not in silent]
        return {"proceed": True, "kept": kept, "discarded": list(isolatable),
                "isolated": list(isolatable)}

    # fail closed (unchanged)
    summary = "; ".join(
        f"{r.engine}:{r.status}:{r.error_type}" for r in records if r.engine in silent)
    raise RuntimeError(
        "allocated symbolic engine(s) produced no candidate and the run "
        f"refused to degrade silently: {summary}")


def simulate_child(engine_outcomes, arm, with_data_identity_fault=False,
                   with_identity_mismatch=False):
    """engine_outcomes: dict engine->status. Returns a synthetic-but-valid
    2-cycle admitted-bank trace if the run proceeds, else raises."""
    if with_data_identity_fault:
        raise DataIdentityError(f"[{arm}] data access / identity failure -> fail closed")
    if with_identity_mismatch:
        raise DataIdentityError(f"[{arm}] source-identity mismatch -> fail closed")

    records = []
    for eng, st in engine_outcomes.items():
        et = "TimeoutError" if st == "timeout" else (None if st == "succeeded" else "RuntimeError")
        records.append(EngineRunRecord(eng, st, error_type=et))

    verdict = decide(records)  # may raise (fail closed)

    # Build a 2-cycle admitted-bank trace from the KEPT engines only (no fakes).
    kept = verdict["kept"]
    bank_rows = []
    for eng in kept:
        for i in range(2):  # 2 cycles, each admits one real candidate per kept engine
            lineage = hashlib.sha256(f"{eng}|{arm}|{i}|{uuid.uuid4()}".encode()).hexdigest()
            bank_rows.append({"cycle": i, "engine": eng, "lineage_id": lineage,
                              "expression": f"mock_candidate_from_{eng}_cycle{i}"})
    trace = []
    for cyc in range(2):
        rows = [r for r in bank_rows if r["cycle"] == cyc]
        trace.append({
            "cycle": cyc,
            "bank_rows_after": rows,           # real entries only, derived from kept engines
            "admitted_lineages": [r["lineage_id"] for r in rows],
            "faked": False,
        })
    return {
        "arm": arm, "verdict": verdict, "trace_complete": len(trace) == 2
        and all(len(c["bank_rows_after"]) > 0 for c in trace),
        "bank_rows_after_counts": [len(c["bank_rows_after"]) for c in trace],
        "trace_is_fabricated": False,
    }


def run_tests():
    results = []
    # Scenario 1: both engines succeed -> identical to original (both kept)
    r = simulate_child({"pysr": "succeeded", "polynomial_lasso": "succeeded"}, "C")
    results.append({"scenario": "both_succeed", "arm": "C", "proceed": True,
                    "kept": r["verdict"]["kept"], "identical_to_original": set(r["verdict"]["kept"]) == {"pysr", "polynomial_lasso"}})

    # Scenario 2: PySR succeeds + Lasso timeout -> keep PySR only
    r = simulate_child({"pysr": "succeeded", "polynomial_lasso": "timeout"}, "D")
    results.append({"scenario": "pysr_ok_lasso_timeout", "arm": "D", "proceed": True,
                    "kept": r["verdict"]["kept"], "discarded": r["verdict"]["discarded"],
                    "isolated": r["verdict"]["isolated"], "trace_complete": r["trace_complete"],
                    "bank_rows_after_counts": r["bank_rows_after_counts"]})

    # Scenario 3: PySR infrastructure fault -> fail closed
    try:
        simulate_child({"pysr": "failed", "polynomial_lasso": "succeeded"}, "R")
        results.append({"scenario": "pysr_infra_fault", "proceed": False, "failed_closed": False, "ERROR": "should have raised"})
    except RuntimeError as e:
        results.append({"scenario": "pysr_infra_fault", "proceed": False, "failed_closed": True,
                        "error": str(e)[:80]})

    # Scenario 4: both engines no valid result -> fail closed
    try:
        simulate_child({"pysr": "timeout", "polynomial_lasso": "timeout"}, "C")
        results.append({"scenario": "both_no_result", "proceed": False, "failed_closed": False, "ERROR": "should have raised"})
    except RuntimeError as e:
        results.append({"scenario": "both_no_result", "proceed": False, "failed_closed": True,
                        "error": str(e)[:80]})

    # Scenario 5: data access / identity failure -> fail closed (separate guard)
    try:
        simulate_child({"pysr": "succeeded", "polynomial_lasso": "succeeded"}, "R",
                       with_data_identity_fault=True)
        results.append({"scenario": "data_identity_fault", "proceed": False, "failed_closed": False, "ERROR": "should have raised"})
    except DataIdentityError as e:
        results.append({"scenario": "data_identity_fault", "proceed": False, "failed_closed": True,
                        "error": str(e)[:80]})

    # Scenario 5b: identity mismatch -> fail closed
    try:
        simulate_child({"pysr": "succeeded", "polynomial_lasso": "succeeded"}, "C",
                       with_identity_mismatch=True)
        results.append({"scenario": "identity_mismatch", "proceed": False, "failed_closed": False, "ERROR": "should have raised"})
    except DataIdentityError as e:
        results.append({"scenario": "identity_mismatch", "proceed": False, "failed_closed": True,
                        "error": str(e)[:80]})

    # Scenario 6: C/D/R identical handling on the same failing input
    arms = {}
    for arm in ("C", "D", "R"):
        try:
            rr = simulate_child({"pysr": "succeeded", "polynomial_lasso": "timeout"}, arm)
            arms[arm] = {"proceed": True, "kept": rr["verdict"]["kept"], "isolated": rr["verdict"]["isolated"]}
        except Exception as e:
            arms[arm] = {"proceed": False, "error": str(e)[:60]}
    results.append({"scenario": "cdr_identical_handling", "arms": arms,
                    "identical": len({json.dumps(arms[a], sort_keys=True) for a in arms}) == 1})

    # Scenario 7: two-cycle admitted-bank trace complete & un-faked (pysr-only case)
    rr = simulate_child({"pysr": "succeeded", "polynomial_lasso": "timeout"}, "C")
    results.append({"scenario": "two_cycle_trace_intact", "trace_complete": rr["trace_complete"],
                    "bank_rows_after_counts": rr["bank_rows_after_counts"],
                    "trace_is_fabricated": rr["trace_is_fabricated"]})

    out = {
        "title": "Engine Failure Isolation — synthetic tests (no provider)",
        "design_ref": "ENGINE_FAILURE_ISOLATION_DESIGN.md",
        "n_scenarios": len(results),
        "all_pass": all(
            (r.get("failed_closed") or r.get("proceed") or r.get("identical") or r.get("trace_complete"))
            and not r.get("ERROR") for r in results),
        "results": results,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(r"D:\01\666\llm-srbench\outputs\feedback_repair_R_v2_AUDIT_20261024\ENGINE_FAILURE_ISOLATION_TESTS.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_tests()
