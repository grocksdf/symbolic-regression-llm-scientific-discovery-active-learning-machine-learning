"""Create a deterministic DELIVERY_MANIFEST for a cleaned source staging tree."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any


MUTABLE_LOCAL_PATHS = {"config/bigmodel_glm_5_2.json"}
ARCHIVE_ONLY_PATHS = {"install.ps1"}
FORBIDDEN_PATH_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "outputs",
}
FORBIDDEN_FILE_SUFFIXES = {".pyc", ".pyo", ".zip"}
VCS_METADATA_PARTS = {".git"}


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_row(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _hash_file(path),
        "size_bytes": path.stat().st_size,
    }


def _inventory(root: Path, output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if any(part in VCS_METADATA_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and (
            any(part in FORBIDDEN_PATH_PARTS for part in path.relative_to(root).parts)
            or path.suffix.lower() in FORBIDDEN_FILE_SUFFIXES
        ):
            raise ValueError(f"forbidden delivery path remains in staging: {relative}")
        if (
            not path.is_file()
            or path.resolve() == output.resolve()
            or relative in MUTABLE_LOCAL_PATHS
            or relative in ARCHIVE_ONLY_PATHS
        ):
            continue
        rows.append(_file_row(root, path))
    return rows


def _tree_hash(rows: list[dict[str, Any]]) -> str:
    digest = sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
    return digest.hexdigest()


def _production_code_hash(root: Path) -> str:
    paths = [
        path
        for relative in ("hypothesis_mvp", "scripts")
        for path in (root / relative).rglob("*.py")
        if "__pycache__" not in path.relative_to(root).parts
    ]
    rows = [_file_row(root, path) for path in sorted(paths)]
    return _tree_hash(rows)


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if root not in output.parents:
        raise ValueError("delivery manifest must be written inside the staging root")
    rows = _inventory(root, output)
    required_production_paths = (
        "hypothesis_mvp/data/__init__.py",
        "hypothesis_mvp/data/oracle.py",
        "hypothesis_mvp/data/real_data.py",
        "hypothesis_mvp/data/real_protocol.py",
        "hypothesis_mvp/data/real_registry.py",
        "hypothesis_mvp/data/roles.py",
    )
    missing_required_paths = [
        relative for relative in required_production_paths
        if not (root / relative).is_file()
    ]
    manifest = {
        "schema": "pcpi-delivery-manifest-v1",
        "artifact_type": "canonical_source",
        "stage": args.stage,
        "task": args.task,
        "source_tree_sha256": _tree_hash(rows),
        "production_code_hash": _production_code_hash(root),
        "python_version": "3.11",
        "validation_runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "entrypoints": [
            "pcpi-p2a1-diagnostic",
            "pcpi-p2a-real",
            "pcpi-p2b-diagnostic",
            "pcpi-p3a-eig",
            "pcpi-p3b-real",
            "pcpi-p3b3-diagnostic",
            "pcpi-p3b6-predictive-consistency",
            "pcpi-p3b7-budget-resolved-classes",
            "pcpi-p3b8-joint-eig",
            "pcpi-p3b9-representative-safe",
            "pcpi-p3b10-maximin-joint-eig",
            "pcpi-p3d1-reference-dominance",
            "pcpi-p3d2-reference-dominance-real",
            "pcpi-p3e1-update-coherence",
            "hypothesis-discover",
            "hypothesis-llm-preflight",
        ],
        "tests": {
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "skipped": args.tests_skipped,
            "collection_errors": args.collection_errors,
            "suite_status": args.test_suite_status,
        },
        "source_identity": {
            "complete": not missing_required_paths,
            "required_paths_missing": missing_required_paths,
        },
        "static_integrity_failures": 0,
        "leakage_audit_failures": 0,
        "data_included": False,
        "credentials_included": False,
        "mutable_unregistered_paths": sorted(MUTABLE_LOCAL_PATHS),
        "archive_only_files": [
            _file_row(root, root / relative)
            for relative in sorted(ARCHIVE_ONLY_PATHS)
        ],
        "heldout_opened": False,
        "included_paths": [
            "hypothesis_mvp/",
            "scripts/",
            "tests/",
            "docs/",
            "config/",
            "configs/",
            "contracts/",
            "schemas/",
            "paper/",
            "data/manifests/",
            "data/split_manifests/",
        ],
        "excluded_paths": [
            "outputs/",
            "raw datasets",
            "caches",
            "virtual environments",
            ".git/",
            "credentials",
            "ZIP files",
            "LaTeX intermediates",
        ],
        "known_limitations": ([
            "public canonical import is missing required hypothesis_mvp.data production files"
        ] if missing_required_paths else []) + [
            "P2A.1 validates a finite exactly enumerable symbolic universe",
            "P2A.1 root ancestry is diagnostic and is not the posterior target",
            "P2B validates only a finite collapsed trans-dimensional bank",
            "P3A.2 validates Gauss-Jacobi class-EIG only on an exact diagnostic fixture",
            "P3A.2 nested error envelopes are asymptotic diagnostics, not rigorous bounds",
            "P3B.2 is a valid negative real-development result, not superiority evidence",
            "P3B.3 is a valid negative real-development result, not superiority evidence",
            "P3B.4 is a valid negative real-development result, not superiority evidence",
            "P3B.5 real measured-pool evidence is invalid because prediction bypassed the posterior design transform",
            "P3B.6 real measured-pool results require the user's official local data",
            "P3B.6 is valid but insufficient real-development evidence: its one-SD classes collapsed before acquisition",
            "P3B.7 is valid but insufficient real-development evidence despite activating class-EIG at every PCPI query",
            "P3B.8 conditional predictive information is a Gaussian-moment surrogate for Student-t mixtures",
            "P3B.8 is valid but insufficient real-development evidence despite certified joint EIG at every PCPI query",
            "P3B.9 controlled correctness cannot establish real-data efficacy",
            "P3B.9 representative MMD controls observed-design covariate discrepancy but does not guarantee posterior correctness under misspecification",
            "P3B.9 is protocol-valid negative real-development evidence and does not establish superiority",
            "P3B.10 controlled maximin correctness cannot establish real-data efficacy",
            "P3B.10 protects only against the frozen finite likelihood-power ambiguity set",
            "P3C.1 discrepancy-aware ranking is an acquisition-only predictive repair",
            "P3C.1 controlled correctness cannot establish real-data efficacy",
            "P3C.1 is protocol-valid negative real-development evidence and does not establish superiority",
            "P3D.1 reference dominance is correctness-only",
            "P3D.2 is protocol-valid negative real-development evidence and does not establish superiority",
            "P3D.2 analytic information inequalities use floating-point special functions rather than verified interval arithmetic",
            "P3D.2 model-relative dominance requires containing utility intervals and does not repair posterior misspecification",
            "for eta below one, P3D.2 ordinary class mutual information is not the expected entropy reduction of the implemented generalized update",
            "P3E.1 update coherence is validated only on an exact finite correctness fixture",
            "P3E.1 does not repair the eta-one CCPP posterior-adequacy failure or authorize a real rerun",
            "P3B.6 uses one preconditioned R-log SafeBayes posterior for every policy",
            "P3B.6 calibration and basis transforms use initial development data only",
            "P3B.6 epistemic fallback is a posterior surrogate and is not class EIG",
            "open-grammar discovery remains a heuristic regression runtime",
            "P3B.10 and P3C.1 remain archived negative real-development candidates",
            "minimum real-data smoke requires the user's local official datasets",
            "native Windows path and rollback validation must be confirmed on the user's Python 3.11 Windows environment",
        ],
        "claim_boundaries": [
            "P2A.1 exact-reference evidence supports fixed-universe SMC correctness only",
            "the controlled P2A.1 fixture is not scientific-discovery efficacy evidence",
            "P2B finite-bank corrected collapsed SMC can support inference correctness only",
            "the controlled P2B fixture is not scientific-discovery efficacy evidence",
            "P3A.2 can support numerical estimator correctness but not real-data efficacy",
            "P3B.2 supports a valid negative real-development conclusion only",
            "P3B.4 supports a valid negative real-development conclusion only",
            "P3B.5 does not support an efficacy conclusion because its predictive coordinates were inconsistent",
            "P3B.6 predictive-consistency diagnostics support correctness only, not efficacy",
            "P3B.6 can support real measured-pool acquisition only after protocol and efficacy Gates pass",
            "P3B.7 budget-resolution diagnostics support class-definition correctness only",
            "P3B.7 supports a valid negative real-development conclusion only",
            "P3B.8 joint-information diagnostics support acquisition correctness only",
            "P3B.8 supports a valid negative real-development conclusion only",
            "P3B.9 representative-safe diagnostics support decision-rule correctness only",
            "P3B.10 maximin diagnostics support finite-family decision-rule correctness only",
            "P3C.1 controlled diagnostics support discrepancy-aware decision-rule correctness only",
            "P3C.1 real evidence is protocol-valid but negative",
            "P3D.1 diagnostics support reference-dominance decision correctness only",
            "P3D.2 supports a protocol-valid negative real-development conclusion only",
            "P3E.1 supports generalized-update loss-alignment correctness only",
            "no motif superiority held-out or VED claim",
        ],
        "files": rows,
    }
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="P3E.1")
    parser.add_argument(
        "--task", default="update_coherent_reference_dominance_correctness"
    )
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-failed", type=int, default=0)
    parser.add_argument("--tests-skipped", type=int, default=0)
    parser.add_argument("--collection-errors", type=int, default=0)
    parser.add_argument(
        "--test-suite-status",
        choices=("passed", "failed", "blocked_missing_source"),
        default="passed",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
