"""Scheduling-only continuation with verified inheritance and isolated workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_regional_structural_matrix import ROOT, MAIN, hash_file, source, write

THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
ARMS = ('A_random','B_local_random','C_ast_joint','D_local_joint','E_attributed_joint','F_shuffled_joint')


def verify_record(record, row, coordinate):
    if record['coordinate'] != row:
        raise ValueError('coordinate identity mismatch')
    if 'failure' in record:
        return  # Terminal failures are inherited, never silently retried.
    result = json.loads((coordinate/'RESULT.json').read_bytes())
    if result != record.get('result') or not result.get('protocol_complete'):
        raise ValueError('completed result mismatch')
    frozen = json.loads((coordinate/'GENERATION_FROZEN.json').read_bytes())
    from hypothesis_mvp.discovery.region_site_attribution import digest
    if set(frozen) != set(ARMS):
        raise ValueError('generation freeze must contain all six arms')
    for arm in ARMS:
        if digest(json.loads((coordinate/f'{arm}_GENERATION.json').read_bytes())) != frozen[arm]:
            raise ValueError('generation artifact identity mismatch')
        for suffix in ('CALIBRATION_FROZEN','ADMISSION_FROZEN'):
            if not (coordinate/f'{arm}_{suffix}.json').is_file():
                raise ValueError('missing independently frozen stage')


def audit(original, source_root):
    if source(MAIN) != original['mainline'] or hash_file(original['hdf5']) != original['hdf5_sha256']:
        raise ValueError('statistical source/data changed')
    if sys.executable != original['python']:
        raise ValueError('interpreter changed')
    worker = ROOT/'scripts/run_regional_structural_matrix.py'
    if hash_file(worker) != original['runner_sha256']:
        raise ValueError('statistical runner changed')
    # Only this scheduler and its correctness tests may differ from the source
    # that generated the inherited scores; all other tracked content must match.
    paths = subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',
                                     original['benchmark'],'HEAD'],text=True).splitlines()
    allowed = {'scripts/parallel_regional_structural_matrix.py','tests/test_parallel_regional_structural_matrix.py'}
    if set(paths)-allowed:
        raise ValueError(f'non-scheduling source changes: {set(paths)-allowed}')
    for root in (ROOT, MAIN):
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
            raise ValueError(f'dirty source: {root}')
    reused = {}
    for i,row in enumerate(original['coordinates']):
        if hash_file(row['bank']) != row['bank_sha256']:
            raise ValueError('frozen engine bank changed')
        progress = source_root/f'PROGRESS_{i:03d}.json'
        if not progress.exists():
            continue
        record = json.loads(progress.read_bytes())
        coordinate = Path(record.get('artifact_root', source_root/f'coordinate_{i:03d}'))
        verify_record(record,row,coordinate)
        reused[i] = {**record,'artifact_root':str(coordinate),
                     'inherited_from':str(progress),'inherited_record_sha256':hash_file(progress)}
    return reused


def execute(i,row,root,freeze_path,timeout,env):
    child = root/f'worker_{i:03d}'
    start = time.monotonic()
    print(f'[{i+1}] start {row["task"]} seed={row["seed"]}',flush=True)
    with (root/f'worker_{i:03d}.log').open('x',encoding='utf-8') as log:
        try:
            process = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_regional_structural_matrix.py'),
                '--freeze',str(freeze_path),'--output-root',str(child),'--coordinate-index',str(i)],
                stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout)
            progress = child/f'PROGRESS_{i:03d}.json'
            if process.returncode not in (0,1) or not progress.exists():
                raise RuntimeError(f'worker exit={process.returncode}; missing/invalid progress')
            record = json.loads(progress.read_bytes())
            verify_record(record,row,child/f'coordinate_{i:03d}')
            if process.returncode != 0 and 'failure' not in record:
                raise RuntimeError('nonzero worker exit with success record')
        except subprocess.TimeoutExpired:
            record = {'coordinate':row,'failure':'CoordinateTimeout','message':f'{timeout}s ceiling'}
        except Exception as error:
            record = {'coordinate':row,'failure':type(error).__name__,'message':str(error)}
    record.update(artifact_root=str(child/f'coordinate_{i:03d}'),wall_seconds=time.monotonic()-start)
    print(f'[{i+1}] finished: '+('success' if 'result' in record else record['failure']),flush=True)
    return record


def worker_limit(path,default):
    value = int(path.read_text().strip()) if path.exists() else default
    if not 1 <= value <= 4:
        raise ValueError('worker limit must be 1..4')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-output',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,3,4),default=2)
    parser.add_argument('--math-threads',type=int,default=1)
    parser.add_argument('--coordinate-timeout',type=int,default=7200)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    if args.math_threads < 1 or args.coordinate_timeout < 1:
        parser.error('positive thread count and timeout required')
    src = args.source_output
    original = json.loads((src/'FREEZE.json').read_bytes())
    # Re-stat the inherited ledger after verification to detect concurrent edits.
    before = {p.name:hash_file(p) for p in src.glob('PROGRESS_*.json')}
    reused = audit(original,src)
    if before != {p.name:hash_file(p) for p in src.glob('PROGRESS_*.json')}:
        raise RuntimeError('source is still changing; stop old runner first')
    print(json.dumps({'reused':len(reused),'successes':sum('result' in r for r in reused.values()),
        'preserved_failures':sum('failure' in r for r in reused.values()),
        'pending':len(original['coordinates'])-len(reused)}),flush=True)
    if args.check_only:
        return
    root = args.output_root
    root.mkdir(parents=True,exist_ok=False)
    freeze = {**original,'benchmark':source(ROOT),'parallel_scheduling_amendment':True}
    write(root/'FREEZE.json',freeze)
    env = os.environ.copy()
    env.update({key:str(args.math_threads) for key in THREAD_ENV})
    env['OMP_DYNAMIC']='FALSE'
    write(root/'CONTINUATION.json',{'source_output':str(src),'source_freeze_sha256':hash_file(src/'FREEZE.json'),
        'scheduler_sha256':hash_file(__file__),'reused_indices':sorted(reused),
        'workers_initial':args.workers,'math_threads':args.math_threads,'thread_environment':{k:env[k] for k in THREAD_ENV},
        'timeout':args.coordinate_timeout,'scheduling_only':True,'partial_coordinates_restart_from_frozen_inputs':True})
    control = root/'WORKERS.txt'
    control.write_text(str(args.workers),encoding='ascii')
    print(f'Concurrency control: {control}',flush=True)
    records = dict(reused)
    for i,r in reused.items():
        write(root/f'PROGRESS_{i:03d}.json',r)
    pending = iter(i for i in range(len(original['coordinates'])) if i not in reused)
    with ThreadPoolExecutor(max_workers=4) as pool:
        active = {}
        exhausted = False
        previous_limit = None
        while active or not exhausted:
            limit = worker_limit(control,args.workers)
            if limit != previous_limit:
                with (root/'SCHEDULING_EVENTS.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'time':time.time(),'workers':limit,'active':len(active)})+'\n')
                previous_limit = limit
            while not exhausted and len(active)<limit:
                i = next(pending,None)
                if i is None:
                    exhausted=True
                    break
                future=pool.submit(execute,i,original['coordinates'][i],root,root/'FREEZE.json',args.coordinate_timeout,env)
                active[future]=i
            if not active:
                continue
            done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
            for future in done:
                i=active.pop(future)
                records[i]=future.result()
                write(root/f'PROGRESS_{i:03d}.json',records[i])
    ordered=[records[i] for i in range(len(original['coordinates']))]
    complete=all('result' in r for r in ordered)
    write(root/('COMPLETE.json' if complete else 'INCOMPLETE.json'),ordered)
    if not complete:
        raise SystemExit(1)


if __name__=='__main__':
    main()
