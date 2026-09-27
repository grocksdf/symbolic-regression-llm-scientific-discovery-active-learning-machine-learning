"""Metadata-only freeze for the AISTATS Decision-Readiness Rate benchmark."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


FAMILY_PATHS = {
    "lsr_transform": "lsr_transform",
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
}


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_names(hdf5_path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "DRR metadata freeze requires the isolated h5py runtime") from exc
    output = {}
    with h5py.File(hdf5_path, "r") as source:
        for family, group_path in FAMILY_PATHS.items():
            group = source[group_path]
            output[family] = sorted(
                str(name) for name, value in group.items()
                if isinstance(value, h5py.Group))
    return output


def _selected_tasks(tasks, protocol):
    selected = {}
    for family, quota in protocol["family_quotas"].items():
        excluded = set(protocol["excluded_tasks"].get(family, ()))
        available = [name for name in tasks[family] if name not in excluded]
        ordered = sorted(available, key=lambda name:
            sha256(
                f"{protocol['selection_seed']}:{family}:{name}".encode()
            ).digest())
        if len(ordered) < quota:
            raise ValueError("DRR family quota exceeds unused task capacity")
        selected[family] = ordered[:quota]
    return selected


def build_drr_benchmark_freeze(project_root, protocol):
    if (protocol.get("schema") != "scientific-aistats-drr-protocol-v1"
            or protocol.get("execution_authorized") is not False
            or sum(protocol["family_quotas"].values()) != 30
            or protocol["primary_metric"]["name"] !=
                "decision_readiness_rate"
            or protocol["primary_metric"][
                "all_registered_tasks_in_denominator"] is not True
            or protocol["primary_metric"][
                "metric_switching_forbidden"] is not True):
        raise ValueError("invalid AISTATS DRR protocol")
    benchmark_root = Path(protocol["benchmark_root"]).resolve()
    benchmark_repo = Path(protocol["benchmark_repository"]).resolve()
    hdf5_path = benchmark_root / protocol["hdf5_member"]
    tasks = _task_names(hdf5_path)
    selected = _selected_tasks(tasks, protocol)
    excluded = {
        family: sorted(set(tasks[family]) - set(selected.get(family, ())))
        for family in tasks}
    data_files = [hdf5_path, *sorted((benchmark_root / "data").glob("*.parquet"))]
    return {
        "schema": "scientific-aistats-drr-benchmark-freeze-v1",
        "source": verify_clean_git_source(project_root),
        "benchmark_source": verify_clean_git_source(benchmark_repo),
        "protocol_sha256": sha256(json.dumps(
            protocol, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest(),
        "data_files": {
            str(path): {"length": path.stat().st_size, "sha256": _sha(path)}
            for path in data_files},
        "available_task_counts": {
            family: len(names) for family, names in tasks.items()},
        "selected_tasks": selected,
        "selected_task_count": sum(map(len, selected.values())),
        "selected_task_identity": sha256(json.dumps(
            selected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "excluded_task_counts": {
            family: len(names) for family, names in excluded.items()},
        "selection_rule": (
            "within-family ascending SHA-256 of "
            "selection_seed:family:task after fixed exclusions"),
        "hdf5_access": "group names and object types only",
        "sample_arrays_accessed": False,
        "equations_accessed": False,
        "test_or_ood_accessed": False,
        "llm_called": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Metadata-only benchmark and primary-metric freeze. No task "
            "arrays, equations, model execution, efficacy, or superiority."),
    }


__all__ = ["build_drr_benchmark_freeze"]
