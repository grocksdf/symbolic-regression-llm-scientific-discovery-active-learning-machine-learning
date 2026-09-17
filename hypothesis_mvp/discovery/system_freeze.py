"""Response-free source/config/runtime identity; never authorizes experiments."""
from hashlib import sha256
import json

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.hypotheses.runtime_environment import (
    runtime_binary_identity, runtime_dependency_snapshot, runtime_dependency_hash,
    dependency_specification_hash,
)


def capture_system_freeze(project_root, config):
    # Source gate first: dirty code fails before any data or output access.
    source = verify_clean_git_source(project_root)
    config_hash = sha256(json.dumps(config, sort_keys=True, allow_nan=False,
                                   separators=(",", ":")).encode()).hexdigest()
    snapshot = runtime_dependency_snapshot()
    return {"schema": "scientific-system-freeze-v1", "source": source,
        "config_sha256": config_hash, "runtime_binaries": runtime_binary_identity(),
        "dependency_sha256": runtime_dependency_hash(snapshot),
        "dependency_specification_sha256": dependency_specification_hash(project_root),
        "real_data_access": False, "heldout_access": False,
        "formal_experiment_authorized": False}


def verify_system_freeze(project_root, config, expected):
    actual = capture_system_freeze(project_root, config)
    if actual != expected:
        raise ValueError("scientific system freeze identity changed")
    return actual
