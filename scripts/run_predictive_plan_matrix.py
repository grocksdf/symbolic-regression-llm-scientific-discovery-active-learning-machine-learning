"""Fresh training-only LLM proposals on retrospective development tasks."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get('HYPOTHESIS_MVP_ROOT', ROOT.parent/'hypothesis_mvp'))
sys.path[:0] = [str(ROOT), str(MAIN)]
from scripts.run_regional_structural_matrix import hash_file, source, write


def coordinate(freeze,row,output=None):
    import h5py
    import numpy as np
    from hypothesis_mvp.data.roles import DataRole,RoleDataset,AcquisitionCovariates,covariate_fingerprint
    from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
    from hypothesis_mvp.discovery.structural_plan_pool import freeze_plan_pool
    from hypothesis_mvp.discovery.bank_search_feedback import feedback_from_posterior
    from hypothesis_mvp.discovery.pcpi_adapter import freeze_expanded_formula_model,EXPANDED_FORMULA_POLICY
    from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
    from scripts.run_gap_candidate_marginal_value import _opened,DATASET_GROUPS,three_arm_role_indices
    from scripts.predictive_plan_search import run_predictive_plan_comparison,CONTRACT
    key=f"{DATASET_GROUPS[row['family']]}/{row['task']}/train"
    with h5py.File(freeze['hdf5'],'r') as h:
        ds=h[key]; idx=three_arm_role_indices(len(ds),task_name=row['task'],seed=row['seed'])
        fit=RoleDataset(DataRole.DEVELOPMENT,*_opened(ds,idx['discovery_development'],response=True))
        vx,vy=_opened(ds,idx['discovery_validation'],response=True)
        pool=_opened(ds,idx['action_covariates'],response=False)
    ids=lambda k:tuple(str(v) for v in idx[k])
    half=len(vx)//2
    left,right=RoleDataset(DataRole.VALIDATION,vx[:half],vy[:half]),RoleDataset(DataRole.VALIDATION,vx[half:],vy[half:])
    ancestry=ids('discovery_development')+ids('discovery_validation')
    design=ExpansionDesign(ancestry,ids('gap_audit'),ids('gap_admission'),ids('reporting'),(0,)*len(idx['gap_admission']),(0,),tested_mixture_count=2)
    regions=[{'id':'axis0-upper','axis':0,'side':'gt','cut':float(np.median(fit.X[:,0]))}]
    shared=json.loads(Path(row['bank']).read_bytes())['shared']
    if shared['task']!=row['task'] or shared['seed']!=row['seed']: raise ValueError('bank coordinate mismatch')
    if output is None:
        return {'fit_rows':len(fit.X),'features':fit.X.shape[1],'independent_roles_opened':False}
    from scripts.selector_llm_proposals import generate
    plans,audits=generate(freeze,shared,fit,regions,Path(str(output)+'_proposals'))
    model=freeze_expanded_formula_model(shared['E0_rows'],n_features=fit.X.shape[1],prior=NormalInverseGammaPrior(),exploration_identity=hash_file(row['bank']),coefficient_policy=EXPANDED_FORMULA_POLICY,minimum_supports=1)
    engine=model.engine(model.stable_hash); posterior=engine.fit_batch(fit.X,fit.y)
    forbidden=design.calibration_ids+design.admission_ids+design.reporting_ids
    feedback=feedback_from_posterior(engine,posterior,fit,search_row_ids=ids('discovery_development'),forbidden_row_ids=forbidden,bank_identity=model.stable_hash)
    checked=freeze_plan_pool(plans,fit=fit,fit_row_ids=ids('discovery_development'),forbidden_row_ids=forbidden,feedback=feedback,
        bank_locations=np.column_stack([engine.predictive_location_scale(m,fit.X)[0] for m in posterior.members]),
        core_rows=shared['E0_rows'],regions=regions,task_identity=f"{row['family']}/{row['task']}")
    if len(checked['rows'])<32: raise ValueError(f"unique compiled pool too small: {len(checked['rows'])}<32; no replacement")

    def reader(role):
        def read():
            with h5py.File(freeze['hdf5'],'r') as h: values=_opened(h[key],idx[role],response=True)
            return ids(role),RoleDataset(DataRole.VALIDATION,*values)
        return read
    return run_predictive_plan_comparison(implementation_contract=CONTRACT,E0=shared['E0_rows'],plans=plans,proposal_audits=audits,
        selection=SimpleNamespace(development=fit,validation=left,acquisition_pool=AcquisitionCovariates(DataRole.ACQUISITION_POOL,pool,covariate_fingerprint(pool))),
        left=left,right=right,row_ids={'fit':ids('discovery_development'),'left':ids('discovery_validation')[:half],'right':ids('discovery_validation')[half:]},
        generation_ancestry_ids=ancestry,design=design,regions=regions,task_identity=f"{row['family']}/{row['task']}",seed=row['seed'],blocks_per_arm=8,
        read_calibration=reader('gap_audit'),read_admission=reader('gap_admission'),read_reporting=reader('reporting'),output_root=output)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',type=Path); p.add_argument('--freeze',type=Path)
    p.add_argument('--historical-freeze',type=Path); p.add_argument('--output-root',type=Path)
    p.add_argument('--provider-env',type=Path); p.add_argument('--provider-url'); p.add_argument('--provider-model'); a=p.parse_args()
    if a.prepare:
        if not all((a.provider_env,a.provider_url,a.provider_model)): p.error('prepare needs explicit provider identity')
        if not a.provider_url.startswith('https://'): raise ValueError('HTTPS required')
        old=json.loads(a.historical_freeze.read_bytes())
        from scripts.provider_transport_preflight import _key
        if not _key(a.provider_env): raise ValueError('provider credential unavailable')
        tasks=sorted({(r['family'],r['task']) for r in old['coordinates']})[:8]
        rows=[r for r in old['coordinates'] if (r['family'],r['task']) in tasks]
        if len(rows)!=16 or len(tasks)!=8: raise ValueError('registered 8 tasks x2 seeds required')
        f={'schema':'predictive-plan-retrospective-matrix-v1','coordinates':rows,'hdf5':old['hdf5'],
           'hdf5_sha256':hash_file(old['hdf5']),'mainline':source(MAIN),'benchmark':source(ROOT),'python':sys.executable,
           'provider_env':str(a.provider_env.resolve()),'provider_env_sha256':hash_file(a.provider_env),
           'provider_url':a.provider_url,'provider_model':a.provider_model,
           'thinking_type':'enabled','reasoning_effort':'low',
           'retrospective_development':True,'primary':'thompson minus random independently screened reporting log score',
           'blocks_per_arm':8,'minimum_unique_pool':32,'failure_policy':'all failures preserved, no replacement',
           'proposal_calls_per_coordinate':8,'proposals_per_call':4,'frontier':False,
           'bootstrap_seed':20261111,'bootstrap_resamples':20000,
           'pass_rule':'all 16 coordinates succeed and task-mean effect >0; CI separately reported',
           'confirmation_access':False,'test_ood_access':False}
        audits=[]
        for row in rows:
            print('Preflight',row['task'],row['seed'],flush=True)
            audits.append(coordinate(f,row))
        f['preflight']=audits
        f['files']={str(Path(__file__).resolve()):hash_file(__file__),str(ROOT/'scripts/selector_llm_proposals.py'):hash_file(ROOT/'scripts/selector_llm_proposals.py'),
                    **{r['bank']:hash_file(r['bank']) for r in rows}}
        write(a.prepare,f); print('Freeze ready:',a.prepare); return
    f=json.loads(a.freeze.read_bytes())
    if hash_file(f['provider_env'])!=f['provider_env_sha256']: raise ValueError('credential file identity changed')
    if source(MAIN)!=f['mainline'] or source(ROOT)!=f['benchmark'] or sys.executable!=f['python'] or hash_file(f['hdf5'])!=f['hdf5_sha256']:
        raise ValueError('source/interpreter/data mismatch')
    for path,expected in f['files'].items():
        if hash_file(path)!=expected: raise ValueError('frozen input changed: '+path)
    a.output_root.mkdir(parents=True,exist_ok=False); write(a.output_root/'FREEZE.json',f)
    results=[]
    for i,row in enumerate(f['coordinates']):
        print(f"[{i+1}/{len(f['coordinates'])}] {row['task']} {row['seed']}",flush=True)
        try: item={'coordinate':row,'result':coordinate(f,row,a.output_root/f'coordinate_{i:03d}')}
        except Exception as e: item={'coordinate':row,'failure':type(e).__name__,'message':str(e)}
        results.append(item); write(a.output_root/f'PROGRESS_{i:03d}.json',item)
    complete=all('result' in r for r in results)
    write(a.output_root/('COMPLETE.json' if complete else 'INCOMPLETE.json'),results)
    effects={}
    for r in results:
        if 'result' not in r: continue
        reports=r['result']['reports']; task=r['coordinate']['family']+'/'+r['coordinate']['task']
        gain=lambda m:reports[m]['gains']['screened_predictive_stacking']
        effects.setdefault(task,[]).append(gain('thompson')-gain('random'))
    task_means={t:sum(v)/len(v) for t,v in effects.items()}
    interval=None
    if complete:
        import numpy as np
        values=np.asarray(list(task_means.values()))
        rng=np.random.default_rng(f['bootstrap_seed'])
        interval=np.quantile(np.mean(rng.choice(values,size=(f['bootstrap_resamples'],len(values))),axis=1),[.025,.975]).tolist()
    write(a.output_root/'SUMMARY.json',{'complete':complete,'task_means':task_means,
          'task_mean_effect':sum(task_means.values())/len(task_means) if complete else None,
          'task_bootstrap_ci95':interval,
          'passed':complete and sum(task_means.values())/len(task_means)>0,
          'failure_count':sum('failure' in r for r in results),'retrospective_development':True})
    if not complete: raise SystemExit(1)


if __name__=='__main__': main()
