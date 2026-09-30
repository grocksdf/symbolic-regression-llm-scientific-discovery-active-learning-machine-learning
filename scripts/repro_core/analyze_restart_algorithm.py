#!/usr/bin/env python3
"""Analyze RESTART source/config and generate an implementation map.

This is not a benchmark runner. It records which RESTART-style components should
be learned, reimplemented, or deliberately excluded from PCPI's LLM search
controller. If --restart-root points to a local clone of https://github.com/Liyunlun/RESTART,
the script scans source/config files for residual refinement, concept/library, and
admission-gate hooks. Without a local clone, it still writes the target component
map so the repo has a concrete algorithm checklist.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "paper_locked_outputs" / "llm_srbench" / "restart_source_analysis"
KEYWORDS = {
    "short_term_targeted_refinement": ["residual", "boost", "refine", "correction", "unexplained"],
    "residual_exploration_function": ["residual_config", "e2e", "exploration", "analyze", "structure analysis"],
    "long_term_structure_library": ["concept", "library", "memory", "snippet", "num_concepts"],
    "improvement_gated_admission": ["fit_score", "min_fit_to_add", "threshold", "improvement", "admission"],
    "llm_search_controller": ["sampler", "prompt", "samples_per_prompt", "global_max_sample_num", "island"],
}

COMPONENTS = {
    "short_term_targeted_refinement": {
        "goal": "Use residual/boosting-style diagnosis to identify what the current equation fails to explain, then propose targeted repairs.",
        "pcpi_target": "residual_repair action type in LLM search controller; no task-specific PO rules.",
        "required_evidence": "candidate-level ledger entries showing current expression, residual summary, proposed repair, validation outcome.",
    },
    "residual_exploration_function": {
        "goal": "Build repair subproblems over [X, f_current(X)] so corrections can depend on both raw variables and current prediction.",
        "pcpi_target": "repair task generator: y or residual target; correction form f_new = f_current + g(f_current, x) or f_new = g(f_current, x).",
        "required_evidence": "ablation table: with vs without residual repair under identical budget.",
    },
    "long_term_structure_library": {
        "goal": "Store only validated structures from successful refinements as reusable motifs/concepts.",
        "pcpi_target": "validated_structure_library.jsonl keyed by operator pattern, variables, improvement, complexity, and domain metadata.",
        "required_evidence": "library admission ledger and library ablation: no library vs gated library.",
    },
    "improvement_gated_admission": {
        "goal": "Admit a refinement only if validation improvement exceeds a threshold after complexity/stability penalties.",
        "pcpi_target": "generalization proxy gate: val_nmse delta, bootstrap stability, coefficient stability, edge split risk, complexity penalty.",
        "required_evidence": "gate accepts/rejects, reasons, before/after validation metrics; no test/OOD metrics in gate.",
    },
    "llm_search_controller": {
        "goal": "LLM emits structured search actions rather than final explanation prose.",
        "pcpi_target": "JSON actions: propose_candidates, repair_residual, mutate_structure, simplify, update_library, reject_region.",
        "required_evidence": "llm_action_ledger.jsonl with redacted prompts/actions and deterministic validator decisions.",
    },
}

EXCLUDED = {
    "test_or_ood_feedback_in_prompt": "Never include problem.test_samples, problem.ood_test_samples, test_extra, or evaluator scores in prompts/gates/tuning.",
    "task_answer_templates": "Do not handwrite benchmark/task-specific solution templates such as PO allowlists/denylists.",
    "llm_as_final_oracle": "LLM should propose actions/candidates; deterministic train/val validator decides admission.",
    "unpaired_budget_changes": "Do not compare LLM and no-LLM with different candidate budgets, seeds, or splits.",
}


def _safe_read(path: Path, max_chars: int = 300_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _iter_source_files(root: Path) -> Iterable[Path]:
    exts = {".py", ".yaml", ".yml", ".md", ".txt", ".sh"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts and ".git" not in path.parts:
            yield path


def _scan_restart(root: Optional[Path]) -> Dict[str, Any]:
    if root is None or not root.exists():
        return {"available": False, "root": str(root) if root else None, "hits": {}, "files_scanned": 0}
    hits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    files_scanned = 0
    for path in _iter_source_files(root):
        files_scanned += 1
        text = _safe_read(path)
        low = text.lower()
        rel = str(path.relative_to(root))
        for comp, kws in KEYWORDS.items():
            matched = [kw for kw in kws if kw.lower() in low]
            if matched:
                # Capture lightweight context lines without copying large source.
                contexts: List[str] = []
                for line in text.splitlines():
                    l = line.lower()
                    if any(kw.lower() in l for kw in matched):
                        stripped = line.strip()
                        if stripped and len(stripped) < 240:
                            contexts.append(stripped)
                    if len(contexts) >= 5:
                        break
                hits[comp].append({"file": rel, "keywords": matched, "context": contexts})
    return {"available": True, "root": str(root), "files_scanned": files_scanned, "hits": hits}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# RESTART-style algorithm analysis for PCPI LLM-SRBench paired ablation")
    lines.append("")
    lines.append(f"Created: {payload['created_at']}")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Stop using legacy closed-loop outputs as the main claim. The new claim is a paired LLM-SRBench evaluation of an LLM search controller that learns RESTART-style adaptive refinement.")
    lines.append("")
    lines.append("## Components to learn/reimplement")
    lines.append("")
    for name, spec in COMPONENTS.items():
        lines.append(f"### {name}")
        lines.append(f"- Goal: {spec['goal']}")
        lines.append(f"- PCPI target: {spec['pcpi_target']}")
        lines.append(f"- Required evidence: {spec['required_evidence']}")
        scan_hits = payload.get("source_scan", {}).get("hits", {}).get(name, [])
        if scan_hits:
            files = ", ".join(sorted({h['file'] for h in scan_hits})[:8])
            lines.append(f"- Local source hints: {files}")
        lines.append("")
    lines.append("## Explicit exclusions")
    lines.append("")
    for name, desc in EXCLUDED.items():
        lines.append(f"- **{name}**: {desc}")
    lines.append("")
    lines.append("## Output contract")
    lines.append("")
    lines.append("The paired runner must produce `paired_deltas.csv`, `paired_protocol_audit.json`, and `llm_action_ledger.jsonl`. The paper table script must use only these paired outputs for the no-LLM vs LLM claim.")
    lines.append("")
    if not payload.get("source_scan", {}).get("available"):
        lines.append("## Source scan note")
        lines.append("")
        lines.append("No local RESTART source tree was provided. Clone the public RESTART repository and rerun with `--restart-root path/to/RESTART` to populate local file-level hints.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--restart-root", default=None, help="Local clone of the RESTART repository to scan.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    root = Path(args.restart_root).resolve() if args.restart_root else None
    source_scan = _scan_restart(root)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "analyze_restart_algorithm.py",
        "source_scan": source_scan,
        "components_to_learn_or_reimplement": COMPONENTS,
        "excluded_components_or_practices": EXCLUDED,
        "paper_claim_rewrite": "We evaluate whether an LLM search controller can learn RESTART-style adaptive refinement for symbolic regression. Under paired no-LLM vs LLM ablations on LLM-SRBench, we measure whether the LLM module improves candidate generation, residual repair, and validation-gated generalization.",
    }
    _write_json(out / "restart_component_map.json", payload)
    (out / "restart_algorithm_notes.md").write_text(_make_markdown(payload), encoding="utf-8")
    print(f"[OK] wrote {out / 'restart_component_map.json'}")
    print(f"[OK] wrote {out / 'restart_algorithm_notes.md'}")
    if source_scan.get("available"):
        print(f"[INFO] scanned {source_scan.get('files_scanned')} files under {source_scan.get('root')}")
    else:
        print("[INFO] no local RESTART source tree provided; wrote target component checklist only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
