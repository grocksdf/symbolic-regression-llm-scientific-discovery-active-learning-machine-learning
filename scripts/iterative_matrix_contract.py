"""Response-free identity and config contract for the iterative bank matrix."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import yaml

CONDITION = "three_arm_formula_expanded_candidate"
SCHEMA = "iterative-expanded-bank-matrix-freeze-v1"
FAMILY_PATHS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "lsr_transform": "lsr_transform",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
}


def digest(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def source_identity(benchmark_root: Path, mainline_root: Path):
    """Resolve the imported discovery code against one explicit clean root."""
    from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

    mainline_root = mainline_root.resolve(strict=True)
    if not (mainline_root / "hypothesis_mvp/discovery/__init__.py").is_file():
        raise ValueError("mainline root must contain hypothesis_mvp/discovery")
    configured = os.environ.get("HYPOTHESIS_MVP_ROOT")
    if configured and Path(configured).resolve() != mainline_root:
        raise ValueError("HYPOTHESIS_MVP_ROOT disagrees with frozen mainline root")
    os.environ["HYPOTHESIS_MVP_ROOT"] = str(mainline_root)
    if ensure_mainline().resolve() != (mainline_root / "hypothesis_mvp"):
        raise ValueError("discovery import disagrees with explicit mainline root")
    from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source

    return (verify_clean_git_source(benchmark_root.resolve()),
            verify_clean_git_source(mainline_root))


def validate_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    agent = config["agent_config"]
    if (config.get("condition") != CONDITION
            or config.get("llm_rl_enabled") is not True
            or config.get("use_library") is not False
            or agent.get("engines") != ["pysr", "polynomial_lasso"]
            or agent.get("cycles") != 2
            or agent.get("discovery_rounds") != 2
            or agent.get("iterative_posterior_refinement") is not True
            or agent.get("posterior_gap_directed") is not True
            or agent.get("provider_failure_mode") != "abort"
            or agent.get("refit_policy") != "pcpi-expanded-fixed-inner-v1"
            or agent.get("task_local_memory") is not False
            or agent.get("use_knowledge") is not False):
        raise ValueError("expanded iterative config violates frozen protocol")
    for key in ("engine_budget", "discovery_budget", "candidates_per_island",
                "new_skeleton_quota", "llm_evaluation_reserve",
                "synthesis_evaluation_reserve"):
        if type(agent.get(key)) is not int or agent[key] <= 0:
            raise ValueError(f"invalid registered search budget: {key}")
    if agent["new_skeleton_quota"] > agent["candidates_per_island"]:
        raise ValueError("new-skeleton quota exceeds island width")
    return config


def selected_tasks(value, family: str) -> set[str]:
    """Collect all previously named tasks, including sealed confirmations."""
    output = set()
    if isinstance(value, dict):
        for name in ("selected_tasks", "development_tasks",
                     "untouched_confirmation_tasks", "confirmation_tasks"):
            mapping = value.get(name)
            if isinstance(mapping, dict):
                entries = mapping.get(family, ())
                if isinstance(entries, str):
                    output.add(entries)
                elif isinstance(entries, list):
                    output.update(str(task) for task in entries)
        for child in value.values():
            output.update(selected_tasks(child, family))
    elif isinstance(value, list):
        for child in value:
            output.update(selected_tasks(child, family))
    return output
