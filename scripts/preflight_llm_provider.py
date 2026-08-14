#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "bigmodel_glm_5_2.json"),
    )
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    settings = ProviderSettings.from_file(args.config)
    if not settings.routes:
        raise RuntimeError("No LLM routes configured")
    runtime = ProposalRuntime(EquationRuntime(1), 1, settings, candidates_per_island=1)
    results = []
    for index in range(max(1, args.calls)):
        batch = runtime.propose(
            task_name="provider_preflight",
            task_desc="Transport and strict protocol stability check for a one-variable physical law.",
            round_id=index + 1,
            island="nmse",
            parent_hash="preflight-parent",
            island_context={
                "current_expression": "x0",
                "failure_signature": ["validation_error"],
                "instruction": "Return exactly one complete finite equation close to x0. "
                "Use protocol_id='hypothesis-proposal-v1', runtime_id='canonical-real-only-discovery', "
                "action='REPLACE', candidate_id='preflight-1', parent_hash='preflight-parent', "
                "round_id matching the request, island='nmse', and no markdown.",
            },
            library_rows=[],
            ephemeral_refinements=[],
        )
        row = {
            "call": index + 1,
            "protocol_valid": batch.protocol_valid,
            "reason": batch.reason,
            "error": batch.error,
            "candidate_count": len(batch.candidates),
            "telemetry": batch.telemetry,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not batch.protocol_valid:
            raise RuntimeError(f"LLM preflight failed on call {index + 1}: {batch.error or batch.reason}")
        if index + 1 < args.calls:
            time.sleep(max(0.0, args.interval))
    print(json.dumps({
        "status": "ok",
        "successful_calls": len(results),
        "routes": [{
            "provider": route.provider,
            "role": route.role,
            "model": route.model,
            "endpoint": route.endpoint,
        } for route in settings.routes],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
