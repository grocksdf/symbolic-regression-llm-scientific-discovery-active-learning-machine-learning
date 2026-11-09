"""Explicit retrospective real-data runner; no provider or engine reruns."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("HYPOTHESIS_MVP_ROOT", ROOT.parent / "hypothesis_mvp"))
sys.path[:0] = [str(ROOT), str(MAIN)]


def hash_file(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def source(root):
    return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"], text=True).strip()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--historical-freeze",type=Path)
    p.add_argument("--blocks-per-arm",type=int,default=16)
    p.add_argument("--coordinate-index",type=int)
    a = p.parse_args()
    if a.prepare:
        if not a.historical_freeze or a.freeze or a.blocks_per_arm < 2:
            p.error("prepare requires historical freeze and >=2 blocks")
        historical = json.loads(a.historical_freeze.read_bytes())
        coordinates = []
        for row in historical["coordinates"]:
            bank = Path(historical["source_output"])/"generation"/row["family"]/row["task"]/f"seed{row['seed']}"/"SHARED_BLOCK.json"
            coordinates.append({**row,"bank":str(bank),"bank_sha256":hash_file(bank)})
        if len(coordinates) != 106 or len({(r['family'],r['task'],r['seed']) for r in coordinates}) != 106:
            raise ValueError("expected all 106 distinct historical coordinates")
        freeze = {"schema":"regional-structural-retrospective-matrix-v1", "coordinates":coordinates,
            "hdf5":historical["hdf5"], "hdf5_sha256":hash_file(historical["hdf5"]),
            "mainline":source(MAIN), "benchmark":source(ROOT), "runner_sha256":hash_file(__file__),
            "python":sys.executable,"blocks_per_arm":a.blocks_per_arm,
            "primary":"E_attributed_joint minus D_local_joint screened_predictive_stacking",
            "mechanism_control":"E_attributed_joint minus F_shuffled_joint",
            "aggregation":"seed means within task, then equal task means",
            "failure_policy":"preserve failures, report incomplete; never discard or retry",
            "retrospective_development":True,"provider_calls":0,"frontier":False}
        write(a.prepare,freeze)
        print(f"Frozen {len(coordinates)} coordinates: {a.prepare}")
        return
    if not a.freeze or not a.output_root: p.error("execution requires freeze and output-root")
    freeze = json.loads(a.freeze.read_bytes())
    if (source(MAIN)!=freeze["mainline"] or source(ROOT)!=freeze["benchmark"]
        or hash_file(__file__)!=freeze["runner_sha256"] or sys.executable!=freeze["python"]
        or hash_file(freeze["hdf5"])!=freeze["hdf5_sha256"]):
        raise ValueError("frozen source/interpreter/data identity mismatch")
    import h5py
    import numpy as np
    from hypothesis_mvp.data.roles import DataRole, RoleDataset, AcquisitionCovariates, covariate_fingerprint
    from hypothesis_mvp.discovery.joint_evidence_gated_expansion import CandidateArchive
    from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
    from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
    from scripts.run_gap_candidate_marginal_value import _opened, DATASET_GROUPS, three_arm_role_indices
    from scripts.regional_structural_search import run_regional_site_comparison, CONTRACT, MODES
    a.output_root.mkdir(parents=True,exist_ok=False)
    write(a.output_root/"FREEZE.json",freeze)
    results=[]
    for i,row in enumerate(freeze["coordinates"]):
        if a.coordinate_index is not None and i != a.coordinate_index:
            continue
        print(f"[{i+1}/106] {row['family']}/{row['task']} seed={row['seed']}",flush=True)
        try:
            if hash_file(row["bank"])!=row["bank_sha256"]: raise ValueError("bank identity changed")
            shared=json.loads(Path(row["bank"]).read_bytes())["shared"]
            if shared["task"]!=row["task"] or shared["seed"]!=row["seed"]: raise ValueError("bank coordinate mismatch")
            key=f"{DATASET_GROUPS[row['family']]}/{row['task']}/train"
            with h5py.File(freeze["hdf5"],"r") as h:
                ds=h[key]
                idx=three_arm_role_indices(int(ds.shape[0]),task_name=row["task"],seed=row["seed"])
                fit=RoleDataset(DataRole.DEVELOPMENT,*_opened(ds,idx["discovery_development"],response=True))
                vx,vy=_opened(ds,idx["discovery_validation"],response=True)
                pool=_opened(ds,idx["action_covariates"],response=False)
            ids=lambda k:tuple(str(v) for v in idx[k])
            half=len(vx)//2
            left=RoleDataset(DataRole.VALIDATION,vx[:half],vy[:half])
            right=RoleDataset(DataRole.VALIDATION,vx[half:],vy[half:])
            validation_ids=ids("discovery_validation")
            ancestry=ids("discovery_development")+validation_ids
            design=ExpansionDesign(ancestry,ids("gap_audit"),ids("gap_admission"),ids("reporting"),
                (0,)*len(idx["gap_admission"]),(0,),tested_mixture_count=len(MODES))
            def reader(role):
                def read():
                    with h5py.File(freeze["hdf5"],"r") as h:
                        values=_opened(h[key],idx[role],response=True)
                    return ids(role),RoleDataset(DataRole.VALIDATION,*values)
                return read
            core=shared["E0_rows"]
            archive=CandidateArchive.freeze(core,n_features=fit.X.shape[1],coefficient_policy=EXPANDED_FORMULA_POLICY)
            result=run_regional_site_comparison(implementation_contract=CONTRACT,E0=core,archive=archive,
                selection=SimpleNamespace(development=fit,validation=left,acquisition_pool=AcquisitionCovariates(
                    DataRole.ACQUISITION_POOL,pool,covariate_fingerprint(pool))),left=left,right=right,
                row_ids={"fit":ids("discovery_development"),"left":validation_ids[:half],"right":validation_ids[half:]},
                generation_ancestry_ids=ancestry,design=design,read_calibration=reader("gap_audit"),
                read_admission=reader("gap_admission"),read_reporting=reader("reporting"),
                task_identity=f"{row['family']}/{row['task']}",seed=row["seed"],blocks_per_arm=freeze["blocks_per_arm"],
                axes=tuple(range(fit.X.shape[1])),output_root=a.output_root/f"coordinate_{i:03d}")
            results.append({"coordinate":row,"result":result})
        except Exception as error:
            results.append({"coordinate":row,"failure":type(error).__name__,"message":str(error)})
        write(a.output_root/f"PROGRESS_{i:03d}.json",results[-1])
    complete=all("result" in r for r in results)
    write(a.output_root/("COMPLETE.json" if complete else "INCOMPLETE.json"),results)
    print(json.dumps({"complete":complete,"completed":sum("result" in r for r in results),"registered":106}))
    if not complete: raise SystemExit(1)


if __name__=="__main__": main()
