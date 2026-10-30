"""Create the transparent eight-task amended rerun after numerical failure."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-protocol", type=Path, required=True)
    parser.add_argument("--terminal-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("amended protocol already exists")
    protocol = json.loads(
        args.failed_protocol.read_text(encoding="utf-8"))
    terminal = json.loads(
        args.terminal_audit.read_text(encoding="utf-8"))
    if terminal.get("status") != (
            "PROTOCOL_TERMINAL_NUMERICAL_FAILURE_NO_EFFICACY_RESULT"):
        raise ValueError("source attempt is not the registered terminal failure")
    protocol.update({
        "status": "AMENDED_TECHNICAL_RERUN_AUTHORIZED",
        "confirmation_execution": False,
        "claim_boundary": (
            "Transparent amended eight-task sensitivity rerun after the "
            "one-time confirmation terminated without an efficacy estimand. "
            "The three partially opened tasks are not untouched confirmation; "
            "results are reported separately from all retrospective evidence."),
        "source_terminal_audit": str(args.terminal_audit.resolve()),
        "amended_rerun": {
            "reason": "all-domain core definedness correctness repair",
            "original_confirmation_pass_or_fail_available": False,
            "tasks_replaced": False, "seeds_replaced": False,
            "thresholds_changed": False, "metrics_changed": False,
            "independent_confirmation_claim_authorized": False},
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "coordinate_count": len(protocol["selection"]["coordinates"]),
        "amended_rerun": True,
        "independent_confirmation_claim_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
