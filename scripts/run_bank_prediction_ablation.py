"""Fixed-engine-bank retrospective prediction ablation, no LLM or search."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('HYPOTHESIS_MVP_ROOT',str(ROOT.parent/'hypothesis_mvp'))]
import numpy as np
from scipy.stats import norm,t
from scipy.special import logsumexp
from scripts.run_regional_structural_matrix import hash_file,source,write,MAIN
from scripts.stacking_diagnostics import fit_stacking_weights,mixture_log_density
from hypothesis_mvp.discovery.pcpi_adapter import freeze_expanded_formula_model,EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.region_site_attribution import digest
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support


def plugin_fit(matrix,y):
    scale=np.max(np.abs(matrix),axis=0);scale=np.where(scale==0,1.,scale)
    coefficients=np.linalg.lstsq(matrix/scale,y,rcond=None)[0]
    return coefficients,scale


def run_case(f,row,folder):
    import h5py
    from scripts.run_gap_candidate_marginal_value import _opened,DATASET_GROUPS,three_arm_role_indices
    shared=json.loads(Path(row['bank']).read_bytes())['shared']
    if shared['task']!=row['task'] or shared['seed']!=row['seed']:raise ValueError('coordinate mismatch')
    with h5py.File(f['hdf5'],'r') as h:
        ds=h[f"{DATASET_GROUPS[row['family']]}/{row['task']}/train"]
        idx=three_arm_role_indices(len(ds),task_name=row['task'],seed=row['seed'])
        X,y=_opened(ds,idx['discovery_development'],response=True)
        C,cy=_opened(ds,idx['gap_audit'],response=True)
    supports={compile_fixed_formula_support(r['expression'],X.shape[1]):r for r in shared['E0_rows']}
    models=[];bayes_cal=[];plugin_cal=[]
    for support,rowbank in supports.items():
        model=freeze_expanded_formula_model([rowbank],n_features=X.shape[1],prior=NormalInverseGammaPrior(),
            exploration_identity=digest(rowbank),coefficient_policy=EXPANDED_FORMULA_POLICY,minimum_supports=1)
        e=model.engine(model.stable_hash);p=e.fit_batch(X,y)
        coefficients,scale=plugin_fit(design_matrix(X,support),y)
        location=(design_matrix(C,support)/scale)@coefficients
        # Estimate plug-in observation variance on calibration, not report data.
        sigma=float(np.linalg.norm(cy-location)/np.sqrt(len(cy)))
        if not np.isfinite(sigma) or sigma<=0:raise ValueError('unidentifiable plug-in Gaussian scale; no clipping')
        bayes_cal.append(e.predictive_logpdf(p,C,cy));plugin_cal.append(norm.logpdf(cy,loc=location,scale=sigma))
        models.append((support,e,p,coefficients,scale,sigma))
    bcal=np.column_stack(bayes_cal);ncal=np.column_stack(plugin_cal)
    bw,ba=fit_stacking_weights(bcal,seed=row['seed'],random_starts=4)
    nw,na=fit_stacking_weights(ncal,seed=row['seed'],random_starts=4)
    folder.mkdir(parents=True,exist_ok=False)
    write(folder/'PREDICTORS_FROZEN.json',{'bayesian_weights':bw.tolist(),'plugin_weights':nw.tolist(),
        'bayesian_optimizer':ba,'plugin_optimizer':na,'supports':list(supports),'reporting_opened':False,
        'plugin_sigmas':[m[-1] for m in models]})
    with h5py.File(f['hdf5'],'r') as h:
        R,ry=_opened(h[f"{DATASET_GROUPS[row['family']]}/{row['task']}/train"],idx['reporting'],response=True)
    blog=[];nlog=[]
    for support,e,p,coefficients,scale,sigma in models:
        blog.append(e.predictive_logpdf(p,R,ry))
        nlog.append(norm.logpdf(ry,loc=(design_matrix(R,support)/scale)@coefficients,scale=sigma))
    blog=np.column_stack(blog);nlog=np.column_stack(nlog);uniform=np.ones(len(models))/len(models)
    scores={name:float(np.mean(mixture_log_density(values,w))) for name,values,w in (
        ('bayes_stacked',blog,bw),('plugin_stacked',nlog,nw),('bayes_uniform',blog,uniform),('plugin_uniform',nlog,uniform))}
    if not all(np.isfinite(v) for v in scores.values()):raise ValueError('nonfinite report score')
    result={'scores':scores,'bayes_minus_plugin_stacked':scores['bayes_stacked']-scores['plugin_stacked'],
        'bayes_minus_plugin_uniform':scores['bayes_uniform']-scores['plugin_uniform'],
        'component_count':len(models),'test_ood_accessed':False,'llm_calls':0}
    write(folder/'RESULT.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--source-freeze',type=Path)
    p.add_argument('--freeze',type=Path);p.add_argument('--output-root',type=Path);a=p.parse_args()
    if a.prepare:
        old=json.loads(a.source_freeze.read_bytes())
        f={'schema':'bank-prediction-ablation-v1','coordinates':old['coordinates'],'hdf5':old['hdf5'],
           'hdf5_sha256':hash_file(old['hdf5']),'mainline':source(MAIN),'benchmark':source(ROOT),'python':sys.executable,
           'runner_sha256':hash_file(__file__),'primary':'bayes_minus_plugin_stacked',
           'secondary':'bayes_minus_plugin_uniform','retrospective_development':True,
           'claim_boundary':'Fixed engine-bank Bayesian predictive package vs least-squares Gaussian plug-in; includes prior shrinkage and parameter uncertainty, not isolation of model-evidence weighting.',
           'bootstrap_seed':20261112,'bootstrap_resamples':20000,'failure_policy':'preserve all; incomplete=no efficacy claim'}
        for row in f['coordinates']:row['bank_sha256']=hash_file(row['bank'])
        write(a.prepare,f);print('Freeze ready:',a.prepare);return
    f=json.loads(a.freeze.read_bytes())
    if (f['mainline']!=source(MAIN) or f['benchmark']!=source(ROOT) or f['python']!=sys.executable
            or f['runner_sha256']!=hash_file(__file__) or f['hdf5_sha256']!=hash_file(f['hdf5'])):raise ValueError('source/data/runtime mismatch')
    a.output_root.mkdir(parents=True,exist_ok=False);write(a.output_root/'FREEZE.json',f);results=[]
    for i,row in enumerate(f['coordinates']):
        print(f"[{i+1}/{len(f['coordinates'])}] {row['task']} {row['seed']}",flush=True)
        try:
            if hash_file(row['bank'])!=row['bank_sha256']:raise ValueError('bank changed')
            entry={'coordinate':row,'result':run_case(f,row,a.output_root/f'coordinate_{i:03d}')}
        except Exception as error:entry={'coordinate':row,'failure':type(error).__name__,'message':str(error)}
        results.append(entry);write(a.output_root/f'PROGRESS_{i:03d}.json',entry)
    complete=all('result'in r for r in results);summary={'complete':complete,'failures':sum('failure'in r for r in results)}
    for metric in ('bayes_minus_plugin_stacked','bayes_minus_plugin_uniform'):
        groups={}
        for r in results:
            if 'result'in r:groups.setdefault(r['coordinate']['family']+'/'+r['coordinate']['task'],[]).append(r['result'][metric])
        means={k:float(np.mean(v)) for k,v in groups.items()}
        summary[metric]={'task_means':means,'task_mean':float(np.mean(list(means.values()))) if complete else None}
        if complete:
            rng=np.random.default_rng(f['bootstrap_seed']);v=np.asarray(list(means.values()))
            summary[metric]['ci95']=np.quantile(rng.choice(v,(f['bootstrap_resamples'],len(v))).mean(axis=1),[.025,.975]).tolist()
    write(a.output_root/'SUMMARY.json',summary)
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
