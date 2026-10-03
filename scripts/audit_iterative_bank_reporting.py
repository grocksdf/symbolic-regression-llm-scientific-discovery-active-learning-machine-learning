"""Post-generation common-report audit for the expanded iterative bank.

Run only after the generation artifact and its admission trace are frozen.
This reads selected roles from the benchmark *train* dataset; no ground-truth
formula metadata, test, OOD or untouched confirmation paths are inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
ensure_mainline()
from hypothesis_mvp.data.roles import (  # noqa: E402
    AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
    covariate_fingerprint,
)
from hypothesis_mvp.discovery.iterative_reporting import (  # noqa: E402
    paired_iterative_bank_reporting,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior  # noqa: E402


GROUPS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
    "lsr_transform": "lsr_transform",
}


def run(artifact: Path, hdf5: Path, family: str, task: str, seed: int):
    frozen = json.loads(artifact.read_text(encoding="utf-8"))
    if frozen.get("task_name") != task:
        raise ValueError("artifact task identity mismatch")
    report = frozen["scientific_discovery_runtime"]
    if (report.get("drr_condition") != "three_arm_formula_expanded_candidate"
            or not report.get("scientist_agent_system_evaluation", {}).get(
                "iterative_feedback_gate")
            or report.get("test_or_ood_accessed")
            or report.get("heldout_opened")):
        raise ValueError("artifact is not a verified closed-report iterative run")
    indices = report["drr_role_row_indices"]
    pairs = report["iterative_cycle_role_row_indices"]
    if len(pairs) < 2:
        raise ValueError("iterative cycle role identities missing")
    from methods.hypothesis_mvp_pcpi.drr_adapter import (  # noqa: E402
        three_arm_role_indices, iterative_gap_role_indices,
    )
    with h5py.File(hdf5, "r") as handle:
        data = handle[f"{GROUPS[family]}/{task}/train"]
        # Reconstruct the registered seed-to-row partition from indices,
        # without touching any formula metadata or test/OOD dataset.
        expected = three_arm_role_indices(
            data.shape[0], task_name=task, seed=seed)
        if {key: list(value) for key, value in expected.items()} != indices:
            raise ValueError("role indices do not match registered seed")
        if [{"gap_audit": list(a), "gap_admission": list(b)}
                for a, b in iterative_gap_role_indices(expected, len(pairs))] != pairs:
            raise ValueError("cycle role partition changed")

        def opened(row_indices, *, response):
            original = [int(index) for index in row_indices]
            ordered = sorted(original)
            values = np.asarray(data[ordered, :], dtype=float)
            lookup = dict(zip(ordered, values, strict=True))
            values = np.asarray([lookup[index] for index in original])
            return (values[:, 1:], values[:, 0]) if response else values[:, 1:]

        fit = RoleDataset(DataRole.DEVELOPMENT,
                          *opened(indices["discovery_development"], response=True))
        validation = RoleDataset(DataRole.VALIDATION,
            *opened(indices["discovery_validation"], response=True))
        actions = opened(indices["action_covariates"], response=False)
        selection = SelectionData(fit, validation,
            AcquisitionCovariates(DataRole.ACQUISITION_POOL, actions,
                                  covariate_fingerprint(actions)), ())
        cycle_roles = tuple(tuple(RoleDataset(DataRole.VALIDATION,
            *opened(pair[name], response=True))
            for name in ("gap_audit", "gap_admission")) for pair in pairs)
        reporting = RoleDataset(DataRole.VALIDATION,
            *opened(indices["reporting"], response=True))
    result = type("FrozenResult", (), {"system_evaluation":
        report["scientist_agent_system_evaluation"]})()
    return paired_iterative_bank_reporting(
        result, selection, reporting, cycle_roles,
        NormalInverseGammaPrior(), 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "hdf5", "family", "task", "seed", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True,
                            type=Path if name in {"artifact", "hdf5", "output"}
                            else int if name == "seed" else str)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("report output already exists")
    result = run(args.artifact, args.hdf5, args.family, args.task, args.seed)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
