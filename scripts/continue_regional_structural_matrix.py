"""Audited post-interruption continuation; original results stay immutable."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

from run_regional_structural_matrix import ROOT, MAIN, hash_file, source, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-output',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--coordinate-timeout',type=int,default=7200)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.coordinate_timeout <= 0: p.error('positive timeout required')
    original=json.loads((a.source_output/'FREEZE.json').read_bytes())
    if source(MAIN)!=original['mainline'] or hash_file(original['hdf5'])!=original['hdf5_sha256']:
        raise ValueError('mainline statistical source or data changed; reuse forbidden')
    if sys.executable!=original['python']: raise ValueError('interpreter changed')
    # Audit all reusable files before starting a new run.
    reused={}
    for i,row in enumerate(original['coordinates']):
        if hash_file(row['bank'])!=row['bank_sha256']: raise ValueError('bank changed')
        progress=a.source_output/f'PROGRESS_{i:03d}.json'
        if not progress.exists(): continue
        record=json.loads(progress.read_bytes())
        if record['coordinate']!=row: raise ValueError('progress coordinate mismatch')
        if 'result' in record:
            coordinate=a.source_output/f'coordinate_{i:03d}'
            result=json.loads((coordinate/'RESULT.json').read_bytes())
            if result!=record['result'] or not result.get('protocol_complete'):
                raise ValueError('completed result mismatch')
            frozen=json.loads((coordinate/'GENERATION_FROZEN.json').read_bytes())
            from hypothesis_mvp.discovery.region_site_attribution import digest
            for arm, identity in frozen.items():
                if digest(json.loads((coordinate/f'{arm}_GENERATION.json').read_bytes()))!=identity:
                    raise ValueError('generation artifact changed')
        elif 'failure' not in record: raise ValueError('invalid progress record')
        reused[i]={**record,'inherited_record_sha256':hash_file(progress),'inherited_from':str(progress)}
    if a.check_only:
        print(json.dumps({'validated_recorded_outcomes':len(reused),
            'successful':sum('result' in r for r in reused.values()),
            'failures_preserved':sum('failure' in r for r in reused.values()),
            'pending':len(original['coordinates'])-len(reused)}))
        return
    a.output_root.mkdir(parents=True,exist_ok=False)
    worker=ROOT/'scripts/run_regional_structural_matrix.py'
    freeze={**original,'benchmark':source(ROOT),'runner_sha256':hash_file(worker)}
    freeze_path=a.output_root/'WORKER_FREEZE.json'
    write(freeze_path,freeze)
    write(a.output_root/'CONTINUATION.json',{'source':str(a.source_output),
        'source_freeze_sha256':hash_file(a.source_output/'FREEZE.json'),
        'post_interruption_development_continuation':True,'original_failures_preserved':True,
        'coordinate_timeout':a.coordinate_timeout,'reuse_indices':sorted(reused),
        'mainline':source(MAIN),'benchmark':source(ROOT),'worker_sha256':hash_file(worker)})
    records=[]
    for i,row in enumerate(original['coordinates']):
        if i in reused:
            print(f'[{i+1}/106] reuse recorded outcome',flush=True)
            record=reused[i]
        else:
            print(f'[{i+1}/106] execute {row["task"]} seed={row["seed"]}',flush=True)
            child=a.output_root/f'worker_{i:03d}'
            started=time.monotonic()
            try:
                # Inherited stdout exposes block progress. Each coordinate has
                # its own process so a symbolic hang cannot stall the matrix.
                process=subprocess.run([sys.executable,'-B',str(worker),'--freeze',str(freeze_path),
                    '--output-root',str(child),'--coordinate-index',str(i)],timeout=a.coordinate_timeout)
                progress=child/f'PROGRESS_{i:03d}.json'
                if not progress.exists(): raise RuntimeError(f'worker exit={process.returncode}, missing progress')
                record=json.loads(progress.read_bytes())
                if record['coordinate']!=row: raise ValueError('worker coordinate mismatch')
            except subprocess.TimeoutExpired:
                record={'coordinate':row,'failure':'CoordinateTimeout','message':f'{a.coordinate_timeout}s ceiling',
                        'partial_artifacts':str(child)}
            except Exception as error:
                record={'coordinate':row,'failure':type(error).__name__,'message':str(error)}
            record['continuation_wall_seconds']=time.monotonic()-started
        write(a.output_root/f'PROGRESS_{i:03d}.json',record)
        records.append(record)
    complete=all('result' in r for r in records)
    write(a.output_root/('COMPLETE.json' if complete else 'INCOMPLETE.json'),records)
    print(json.dumps({'registered':len(records),'successful':sum('result' in r for r in records),
                      'failures_preserved':sum('failure' in r for r in records)}))
    if not complete: raise SystemExit(1)


if __name__=='__main__': main()
